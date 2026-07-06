import argparse
import logging
import os

import pandas as pd
from tqdm import tqdm


def main(args):
    logging.info("Loading index parquet...")
    df_index = pd.read_parquet(args.index_file)
    code_cols = [c for c in df_index.columns if c.startswith("code_")]
    logging.info("Index shape: %s, code columns: %s", df_index.shape, code_cols)

    logging.info("Loading item parquet...")
    df_item = pd.read_parquet(args.item_file)
    logging.info("Item shape: %s", df_item.shape)

    assert len(df_index) == len(df_item), (
        f"index 行数({len(df_index)}) 与 item 行数({len(df_item)}) 不一致"
    )

    rows = []
    for idx in tqdm(range(len(df_item)), desc="Merging"):
        item = df_item.iloc[idx]
        index_row = df_index.iloc[idx]

        sid = [index_row[c] for c in code_cols]
        geohash = item.get("geohash", "") or ""
        geo_sid = f"<{geohash}>" + "".join(sid)

        rows.append(
            {
                "item_id": item.get("item_id", ""),
                "sence_id": item.get("sence_id", ""),
                "title": item.get("title", ""),
                "description": item.get("description", ""),
                "brand": item.get("brand", ""),
                "shop_id": item.get("shop_id", ""),
                "i_lng": item.get("i_lng", ""),
                "i_lag": item.get("i_lag", ""),
                "geo_sid": geo_sid,
            }
        )

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    df.to_parquet(args.output_file, index=False)

    logging.info("Done. Rows: %d, Output: %s", len(df), args.output_file)


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