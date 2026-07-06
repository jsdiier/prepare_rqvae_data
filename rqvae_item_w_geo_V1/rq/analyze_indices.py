#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断 step4 产出的 index parquet：
  - 每层 codebook 的利用率（用了多少个 code、分布熵、头部 code 占比）
  - 整体 SID 冲突率、冲突桶大小分布
可选 --emb_cache 时再报告 embedding 的范数分布（采样）。

用法:
    python rq/analyze_indices.py --index_file ./item_info/MX_item_recall.index.parquet
    python rq/analyze_indices.py --index_file ... --emb_cache ./item_info/item_emb.parquet.embcache.f16.npy
"""
import argparse
import re

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

CODE_RE = re.compile(r"(\d+)>$")


def load_codes(index_file):
    pf = pq.ParquetFile(index_file)
    code_cols = [c for c in pf.schema_arrow.names if c.startswith("code_")]
    n = pf.metadata.num_rows
    codes = np.empty((n, len(code_cols)), dtype=np.int32)
    offset = 0
    for batch in tqdm(pf.iter_batches(batch_size=262144, columns=code_cols),
                      desc="reading index", ncols=100):
        for j in range(len(code_cols)):
            col = batch.column(j).to_pylist()
            codes[offset:offset + len(col), j] = [
                int(CODE_RE.search(v).group(1)) for v in col
            ]
        offset += batch.num_rows
    assert offset == n
    return codes, code_cols


def level_report(codes, code_cols):
    print("\n========== 每层 codebook 利用率 ==========")
    for j, name in enumerate(code_cols):
        col = codes[:, j]
        counts = np.bincount(col)
        used = int((counts > 0).sum())
        p = counts[counts > 0] / len(col)
        entropy = float(-(p * np.log2(p)).sum())
        max_entropy = np.log2(max(used, 1))
        top5 = np.sort(counts)[::-1][:5]
        print(f"[{name}] 使用 code 数: {used}/{len(counts)}  "
              f"归一化熵: {entropy / max_entropy if max_entropy > 0 else 0:.3f}  "
              f"top5 code 占比: {top5.sum() / len(col) * 100:.1f}%  "
              f"最大单 code 占比: {top5[0] / len(col) * 100:.2f}%")


def collision_report(codes):
    print("\n========== SID 冲突统计 ==========")
    _, counts = np.unique(codes, axis=0, return_counts=True)
    total = len(codes)
    unique = len(counts)
    print(f"总条数: {total}")
    print(f"唯一 SID 数: {unique}")
    print(f"冲突率: {(total - unique) / total * 100:.2f}%")
    top = np.sort(counts)[::-1]
    print(f"最大冲突桶: {top[0]} 条")
    print(f"top10 冲突桶: {top[:10].tolist()}")
    for q in (50, 90, 99):
        print(f"桶大小 P{q}: {int(np.percentile(counts, q))}")
    only_one = int((counts == 1).sum())
    print(f"独占 SID 的 item 数: {only_one} ({only_one / total * 100:.1f}%)")


def dup_report(emb_cache):
    """全量扫描：精确统计完全重复的 embedding 占比（= SID 冲突率的理论下界）"""
    print("\n========== embedding 全量重复统计 ==========")
    arr = np.load(emb_cache, mmap_mode="r")
    seen = set()
    step = 262144
    for i in tqdm(range(0, len(arr), step), desc="hashing rows", ncols=100):
        chunk = np.ascontiguousarray(arr[i:i + step])
        for row in chunk:
            seen.add(hash(row.tobytes()))
    total = len(arr)
    distinct = len(seen)
    print(f"总条数: {total}")
    print(f"唯一 embedding 数: {distinct}")
    print(f"重复占比(冲突率理论下界): {(total - distinct) / total * 100:.2f}%")


def emb_report(emb_cache, sample_n=200_000):
    print("\n========== embedding 范数分布（采样） ==========")
    arr = np.load(emb_cache, mmap_mode="r")
    idx = np.sort(np.random.default_rng(0).choice(
        len(arr), size=min(sample_n, len(arr)), replace=False))
    sample = np.asarray(arr[idx], dtype=np.float32)
    norms = np.linalg.norm(sample, axis=1)
    print(f"shape={arr.shape}, 采样 {len(sample)} 条")
    print(f"norm: min={norms.min():.4f} p50={np.median(norms):.4f} "
          f"mean={norms.mean():.4f} max={norms.max():.4f} std={norms.std():.4f}")
    dup = len(sample) - len(np.unique(sample, axis=0))
    print(f"采样内完全重复向量: {dup} 条 ({dup / len(sample) * 100:.2f}%)"
          f"  (注意: 采样值低估全量重复率)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file", required=True)
    parser.add_argument("--emb_cache", default=None,
                        help="可选，item_emb.parquet.embcache*.npy 路径")
    parser.add_argument("--dup_check", action="store_true",
                        help="全量扫描 embedding 重复占比（需 --emb_cache，几分钟）")
    args = parser.parse_args()

    codes, code_cols = load_codes(args.index_file)
    level_report(codes, code_cols)
    collision_report(codes)
    if args.emb_cache:
        emb_report(args.emb_cache)
        if args.dup_check:
            dup_report(args.emb_cache)
