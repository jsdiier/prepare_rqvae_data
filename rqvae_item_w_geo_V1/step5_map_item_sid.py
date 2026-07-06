import argparse
import logging
import math
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

BATCH_ROWS = 65536
OUT_FIELDS = ["item_id", "sence_id", "title", "description", "brand",
              "shop_id", "i_lng", "i_lag"]


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
    logging.info("Rows: %d, code columns: %s", n_item, code_cols)

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

    try:
        for item_tbl, index_tbl in tqdm(zip(item_iter, index_iter),
                                        total=total_batches, desc="Merging",
                                        ncols=100):
            assert len(item_tbl) == len(index_tbl), (
                f"批次行数错位: item={len(item_tbl)}, index={len(index_tbl)}"
            )
            item_df = item_tbl.to_pandas()
            index_df = index_tbl.to_pandas()

            # sid = 各层级 code 字符串直接拼接（与原逐行 "".join(sid) 语义一致）
            sid = index_df[code_cols[0]].astype(str)
            for c in code_cols[1:]:
                sid = sid + index_df[c].astype(str)

            if "geohash" in item_df.columns:
                geohash = item_df["geohash"].fillna("").astype(str)
            else:
                geohash = pd.Series([""] * len(item_df))
            geo_sid = "<" + geohash + ">" + sid

            out = pd.DataFrame({
                field: (item_df[field].fillna("").astype(str)
                        if field in item_df.columns
                        else [""] * len(item_df))
                for field in OUT_FIELDS
            })
            out["geo_sid"] = geo_sid.to_numpy()

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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file", default="./item_info/MX_item_recall.index.parquet")
    parser.add_argument("--item_file",  default="./item_info/MX_item_recall.item.parquet")
    parser.add_argument("--output_file", default="./item_info/MX_item_recall.parquet")

    args = parser.parse_args()
    main(args)
