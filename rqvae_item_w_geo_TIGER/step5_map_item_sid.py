import argparse
import configparser
import logging
import math
import os
from collections import Counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

BATCH_ROWS = 65536
OUT_FIELDS = ["item_id", "sence_id", "title", "description", "brand",
              "shop_id", "i_lng", "i_lag"]

# mem-token 前缀，index 从 1 开始（与合并 rank 对应：<m_1> 是第 1 条 BPE 规则）
MEM_PREFIX = "<m_{}>"
# 拼接 gen-token 序列成字符串 key 用的分隔符（token 里不会出现）
_SEP = "\x1f"


# ============================================================
# BPE mem-token 挖掘
# ============================================================
def _merge_seq(word, pair, new_sym):
    """把 word(tuple) 中所有相邻的 pair 合并成 new_sym，返回新的 tuple。"""
    a, b = pair
    out = []
    i = 0
    n = len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_sym)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def learn_bpe(seq_freq, v_mem, min_count):
    """
    在带频次的 gen-token 序列集合上学习标准 BPE。
    seq_freq: dict[tuple[str,...], int]，key 是某个 item 的 gen-token 序列，
              value 是拥有该序列的 item 数（含重复 item）。
    返回:
        rules:  list[(pair, new_sym)]，按合并 rank 升序（rank 0 优先级最高）
        expand: dict[str, str]，每个符号展开成的原始 gen-token 拼接串，用于日志
    合并对选取：频次最高者；频次相同按 pair 字典序取最大以保证可复现。
    """
    words = dict(seq_freq)  # 会被逐步合并，拷贝一份避免污染原始序列
    rules = []
    # 展开表：base token 展开为自身；合成符号展开为其 pair 两侧展开的拼接
    expand = {}

    for k in range(v_mem):
        pair_counts = Counter()
        for word, f in words.items():
            for i in range(len(word) - 1):
                pair_counts[(word[i], word[i + 1])] += f
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        best_count = pair_counts[best_pair]
        if best_count < min_count:
            break

        new_sym = MEM_PREFIX.format(k + 1)
        a, b = best_pair
        expand[new_sym] = expand.get(a, a) + expand.get(b, b)
        rules.append((best_pair, new_sym))

        merged = {}
        for word, f in words.items():
            mw = _merge_seq(word, best_pair, new_sym)
            merged[mw] = merged.get(mw, 0) + f
        words = merged

    return rules, expand


def apply_bpe(seq, rank_of_pair, sym_of_rank):
    """
    对单个 gen-token 序列重放 BPE 合并，收集轨迹上产生过的【全部】中间符号。
    每一步在当前序列中挑选 rank 最小（优先级最高）的相邻对合并，记录其产生的符号，
    直到没有可用规则。返回按产生顺序、去重后的 mem 符号列表。
    """
    seq = list(seq)
    produced = []
    seen = set()
    while len(seq) > 1:
        best_i = -1
        best_rank = None
        for i in range(len(seq) - 1):
            r = rank_of_pair.get((seq[i], seq[i + 1]))
            if r is not None and (best_rank is None or r < best_rank):
                best_rank = r
                best_i = i
        if best_i < 0:
            break
        new_sym = sym_of_rank[best_rank]
        if new_sym not in seen:
            seen.add(new_sym)
            produced.append(new_sym)
        seq[best_i:best_i + 2] = [new_sym]
    return produced


def build_mem_tokens(pf_index, gen_cols, v_mem, min_count):
    """
    第一趟：读 index_file 的 gen-token 列，统计带频次的序列，学习 BPE，
    并为每个唯一序列预计算其 mem_sid 字符串。
    返回:
        seq_to_mem: dict[tuple[str,...], str]，序列 -> mem_sid（多个 mem 符号直接拼接，
                    无 mem 时为 ""）；用于第二趟按行查表。
    """
    if len(gen_cols) < 2:
        logging.warning("[MEM-BPE] gen-token 列数 %d < 2，无法构造相邻对，mem_sid 全为空",
                        len(gen_cols))
        return {}

    # -------- 统计带频次的 gen-token 序列（含重复 item）--------
    seq_counter = Counter()
    for batch in pf_index.iter_batches(batch_size=BATCH_ROWS, columns=gen_cols):
        df = batch.to_pandas().astype(str)
        key = df[gen_cols[0]]
        for c in gen_cols[1:]:
            key = key + _SEP + df[c]
        for k, v in key.value_counts().items():
            seq_counter[k] += int(v)

    seq_freq = {tuple(k.split(_SEP)): v for k, v in seq_counter.items()}
    total_items = sum(seq_freq.values())
    logging.info("[MEM-BPE] corpus items=%d, unique gen sequences=%d, gen cols=%s",
                 total_items, len(seq_freq), gen_cols)

    # -------- 学习 BPE --------
    logging.info("[MEM-BPE] learning BPE: v_mem=%d, min_count=%d", v_mem, min_count)
    rules, expand = learn_bpe(seq_freq, v_mem, min_count)
    stop_reason = "v_mem reached" if len(rules) >= v_mem else "no pair >= min_count"
    logging.info("[MEM-BPE] learned %d mem tokens (stopped by: %s)",
                 len(rules), stop_reason)

    rank_of_pair = {pair: rank for rank, (pair, _) in enumerate(rules)}
    sym_of_rank = {rank: sym for rank, (_, sym) in enumerate(rules)}

    # -------- 每个唯一序列预计算 mem_sid，并统计覆盖度 --------
    seq_to_mem = {}
    mem_item_count = Counter()   # mem 符号 -> 含该符号的 item 数
    items_with_mem = 0
    for seq, f in seq_freq.items():
        produced = apply_bpe(seq, rank_of_pair, sym_of_rank)
        if produced:
            seq_to_mem[seq] = "".join(produced)
            items_with_mem += f
            for m in produced:
                mem_item_count[m] += f
        # 无 mem 的序列不入表，查表时默认 ""

    # -------- 日志：每个 mem token 的数量与占比 --------
    logging.info("[MEM-BPE] mem tokens (by merge rank): pattern / count / ratio")
    for _pair, sym in rules:
        cnt = mem_item_count.get(sym, 0)
        ratio = cnt / total_items if total_items else 0.0
        logging.info("    %-8s = %-24s count=%-10d ratio=%.4f%%",
                     sym, expand.get(sym, sym), cnt, ratio * 100)

    no_mem = total_items - items_with_mem
    logging.info("[MEM-BPE] items with mem token: %d (%.2f%%) | without: %d (%.2f%%)",
                 items_with_mem, 100.0 * items_with_mem / total_items if total_items else 0.0,
                 no_mem, 100.0 * no_mem / total_items if total_items else 0.0)

    return seq_to_mem


def iter_aligned_batches(pf, columns, batch_size):
    """
    以固定 batch_size 重切分 iter_batches 的输出。
    pyarrow 的 iter_batches 在 row group 边界可能吐出小于 batch_size 的批次，
    两个文件的 row group 划分不同时批次会错位；重切分后保证两路严格对齐。
    """
    buf = []
    buffered = 0
    for b in pf.iter_batches(batch_size=batch_size, columns=columns):
        buf.append(b)
        buffered += len(b)
        while buffered >= batch_size:
            t = pa.Table.from_batches(buf)
            yield t.slice(0, batch_size)
            rest = t.slice(batch_size)
            buf = rest.to_batches()
            buffered = rest.num_rows
    if buffered:
        yield pa.Table.from_batches(buf)


def main(args):
    pf_index = pq.ParquetFile(args.index_file)
    pf_item = pq.ParquetFile(args.item_file)

    n_index = pf_index.metadata.num_rows
    n_item = pf_item.metadata.num_rows
    assert n_index == n_item, (
        f"index 行数({n_index}) 与 item 行数({n_item}) 不一致"
    )

    code_cols = [c for c in pf_index.schema_arrow.names if c.startswith("code_")]
    # gen-token = 除最后一位（去重位）外的 code 列；mem-token 只在 gen-token 上挖掘
    gen_cols = code_cols[:-1]
    dedup_col = code_cols[-1] if code_cols else None
    logging.info("Rows: %d, code columns: %s (gen=%s, dedup=%s, excluded from BPE)",
                 n_item, code_cols, gen_cols, dedup_col)

    # -------- 第一趟：学习 BPE + 预计算每个唯一序列的 mem_sid --------
    seq_to_mem = build_mem_tokens(pf_index, gen_cols, args.v_mem, args.min_count)

    item_schema_names = set(pf_item.schema_arrow.names)
    item_read_cols = [f for f in OUT_FIELDS if f in item_schema_names]
    if "geohash" in item_schema_names:
        item_read_cols = item_read_cols + ["geohash"]

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    writer = None
    written = 0
    total_batches = math.ceil(n_item / BATCH_ROWS)

    item_iter = iter_aligned_batches(pf_item, item_read_cols, BATCH_ROWS)
    index_iter = iter_aligned_batches(pf_index, code_cols, BATCH_ROWS)

    # -------- 第二趟：拼 geo_sid（不变）+ 查表补 mem_sid，边算边写 --------
    try:
        for item_tbl, index_tbl in tqdm(zip(item_iter, index_iter),
                                        total=total_batches, desc="Merging",
                                        ncols=100):
            assert len(item_tbl) == len(index_tbl), (
                f"批次行数错位: item={len(item_tbl)}, index={len(index_tbl)}"
            )
            item_df = item_tbl.to_pandas()
            index_df = index_tbl.to_pandas()

            # sid = 各层级 code 字符串直接拼接（与原逐行 "".join(sid) 语义一致，含去重位）
            sid = index_df[code_cols[0]].astype(str)
            for c in code_cols[1:]:
                sid = sid + index_df[c].astype(str)

            if "geohash" in item_df.columns:
                geohash = item_df["geohash"].fillna("").astype(str)
            else:
                geohash = pd.Series([""] * len(item_df))
            geo_sid = "<" + geohash + ">" + sid

            # mem_sid = 该行 gen-token 序列查预计算表；未命中高频组合则为 ""
            if gen_cols and seq_to_mem:
                gen_tuples = zip(*[index_df[c].astype(str) for c in gen_cols])
                mem_sid = [seq_to_mem.get(t, "") for t in gen_tuples]
            else:
                mem_sid = [""] * len(index_df)

            out = pd.DataFrame({
                field: (item_df[field].fillna("").astype(str)
                        if field in item_df.columns
                        else [""] * len(item_df))
                for field in OUT_FIELDS
            })
            out["geo_sid"] = geo_sid.to_numpy()
            out["mem_sid"] = mem_sid

            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(args.output_file, table.schema)
            writer.write_table(table)
            written += len(out)
    finally:
        if writer is not None:
            writer.close()

    assert written == n_item, f"写出行数({written})与输入({n_item})不一致"
    logging.info("Done. Rows: %d, Output: %s", written, args.output_file)


def load_conf_defaults(conf_path):
    """从 common.conf 读取 [mem_token] 段的默认超参。"""
    cp = configparser.ConfigParser()
    cp.read(conf_path)
    v_mem = cp.getint("mem_token", "v_mem", fallback=200)
    min_count = cp.getint("mem_token", "min_count", fallback=10)
    return v_mem, min_count


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    default_conf = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "common.conf"
    )
    conf_v_mem, conf_min_count = load_conf_defaults(default_conf)

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file", default="./item_info/MX_item_recall.index.parquet")
    parser.add_argument("--item_file",  default="./item_info/MX_item_recall.item.parquet")
    parser.add_argument("--output_file", default="./item_info/MX_item_recall.parquet")
    # mem-token 超参默认取自 common.conf，命令行可覆盖
    parser.add_argument("--v_mem", type=int, default=conf_v_mem,
                        help="mem 词表大小上限（BPE 合并规则数上限）")
    parser.add_argument("--min_count", type=int, default=conf_min_count,
                        help="BPE 合并相邻对的最小频次")

    args = parser.parse_args()
    logging.info("mem-token config: v_mem=%d, min_count=%d (conf: %s)",
                 args.v_mem, args.min_count, default_conf)
    main(args)
