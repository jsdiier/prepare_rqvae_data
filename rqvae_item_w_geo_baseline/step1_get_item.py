#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 Hive 表 soda_international_trade_stg.item_recall_meta_w_geo 对应的 HDFS parquet 路径下
流式读取前 N 条数据，整理成 json 并落盘到 item_info/{country_code}_item_recall.item.json

用法:
    python3 get_item_json.py [common.conf]
"""

import os
import sys
import shutil
import subprocess
import tempfile
import configparser

try:
    import pyarrow.parquet as pq
except ImportError:
    print("[ERROR] 需要 pyarrow，请先 pip install pyarrow", file=sys.stderr)
    sys.exit(1)

try:
    import pygeohash as pgh
except ImportError:
    print("[ERROR] 需要 pygeohash，请先 pip install pygeohash", file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("[ERROR] 需要 pandas，请先 pip install pandas", file=sys.stderr)
    sys.exit(1)


GEOHASH_FIELD = "geohash"
LNG_FIELD = "i_lng"
LAT_FIELD = "i_lag"


def load_config(conf_path: str) -> dict:
    cp = configparser.ConfigParser()
    if not cp.read(conf_path, encoding="utf-8"):
        raise FileNotFoundError(f"找不到配置文件: {conf_path}")

    cfg = {
        "hadoop_bin": cp.get("hadoop", "hadoop_bin"),
        "hdfs_root": cp.get("hive", "hdfs_root"),
        "table": cp.get("hive", "table"),
        "dt": cp.get("hive", "dt"),
        "country_code": cp.get("hive", "country_code"),
        "item_count": cp.getint("output", "item_count"),
        "output_dir": cp.get("output", "output_dir", fallback="item_info"),
        "output_fields": [
            f.strip()
            for f in cp.get("output", "output_fields").split(",")
            if f.strip()
        ],
        "geohash_precision": cp.getint("output", "geohash_precision", fallback=6),
    }
    return cfg


def build_hdfs_dir(cfg: dict) -> str:
    return "/".join([
        cfg["hdfs_root"].rstrip("/"),
        cfg["table"],
        f"dt={cfg['dt']}",
        f"country_code={cfg['country_code']}",
    ])


def list_hdfs_part_files(hadoop_bin: str, hdfs_dir: str) -> list:
    """列出该分区下所有 part 文件，按文件名排序，保证可复现"""
    cmd = [hadoop_bin, "fs", "-ls", hdfs_dir]
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        raise RuntimeError(f"hadoop fs -ls 失败: {ret.stderr}")

    files = []
    for line in ret.stdout.splitlines():
        parts = line.split()
        if len(parts) < 8:
            continue
        path = parts[-1]
        # 只要 part- 开头的数据文件，过滤掉 _SUCCESS 等
        fname = os.path.basename(path)
        if fname.startswith("part-"):
            files.append(path)

    files.sort()
    if not files:
        raise RuntimeError(f"目录下没有找到 part 文件: {hdfs_dir}")
    return files


def download_file(hadoop_bin: str, hdfs_path: str, local_dir: str) -> str:
    local_path = os.path.join(local_dir, os.path.basename(hdfs_path))
    cmd = [hadoop_bin, "fs", "-get", hdfs_path, local_path]
    ret = subprocess.run(cmd, capture_output=True, text=True)
    if ret.returncode != 0:
        raise RuntimeError(f"hadoop fs -get 失败: {ret.stderr}")
    return local_path


def compute_geohash(lng_val, lat_val, precision: int):
    """基于经纬度计算 geohash，缺失或非法值时返回 None"""
    if lng_val is None or lat_val is None:
        return None
    try:
        lng = float(lng_val)
        lat = float(lat_val)
    except (TypeError, ValueError):
        return None
    try:
        return pgh.encode(lat, lng, precision=precision)
    except Exception as e:
        print(f"[WARN] geohash 计算失败 (lng={lng_val}, lat={lat_val}): {e}")
        return None


def stream_read_items(hadoop_bin: str, hdfs_files: list, item_count: int, fields: list,
                       geohash_precision: int = 6) -> list:
    """
    流式读取：逐个 part 文件下载到临时目录，使用 pyarrow 按 row group / batch 读取。
    item_count > 0 时，凑够 item_count 条就立即停止，不必下载/读取全部文件；
    item_count == -1 时，表示读取该分区下的全部数据，不提前停止。
    """
    unlimited = item_count == -1

    items = []
    tmp_dir = tempfile.mkdtemp(prefix="item_recall_")
    geohash_enabled = GEOHASH_FIELD in fields

    # geohash 不是 parquet 中的真实列，读取时要剔除；
    # 但若需要计算 geohash，要保证 i_lng / i_lag 这两列被一并读取出来
    real_fields = [f for f in fields if f != GEOHASH_FIELD]
    if geohash_enabled:
        for geo_col in (LNG_FIELD, LAT_FIELD):
            if geo_col not in real_fields:
                real_fields.append(geo_col)

    read_batch_size = 1000 if unlimited else min(item_count, 1000)

    try:
        for hdfs_path in hdfs_files:
            if not unlimited and len(items) >= item_count:
                break

            local_path = download_file(hadoop_bin, hdfs_path, tmp_dir)
            print(f"[INFO] 已下载 {hdfs_path} -> {local_path}")

            try:
                pf = pq.ParquetFile(local_path)
                schema_names = set(pf.schema_arrow.names)
                valid_fields = [f for f in real_fields if f in schema_names]
                missing = [f for f in real_fields if f not in schema_names]
                if missing:
                    print(f"[WARN] 以下字段在 parquet schema 中不存在，将被忽略: {missing}")

                # 按 batch 流式读取，避免一次性加载整份文件到内存
                for batch in pf.iter_batches(batch_size=read_batch_size,
                                              columns=valid_fields):
                    batch_dict = batch.to_pylist()
                    for row in batch_dict:
                        if geohash_enabled:
                            row[GEOHASH_FIELD] = compute_geohash(
                                row.get(LNG_FIELD), row.get(LAT_FIELD), geohash_precision
                            )
                        items.append(row)
                        if not unlimited and len(items) >= item_count:
                            break
                    if not unlimited and len(items) >= item_count:
                        break
                if unlimited and len(items) % 50000 < read_batch_size:
                    print(f"[INFO] 已累计读取 {len(items)} 条")
            finally:
                # 及时清理已读完的本地文件，节省磁盘
                if os.path.exists(local_path):
                    os.remove(local_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return items if unlimited else items[:item_count]


def to_dataframe(items: list, fields: list) -> "pd.DataFrame":
    rows = []
    for row in items:
        rows.append({
            field: ("" if row.get(field) is None else str(row[field]))
            for field in fields
        })
    return pd.DataFrame(rows, columns=fields)


def main():
    conf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "common.conf"
    )
    cfg = load_config(conf_path)

    hdfs_dir = build_hdfs_dir(cfg)
    print(f"[INFO] 目标 HDFS 目录: {hdfs_dir}")
    item_count_desc = "全部数据" if cfg["item_count"] == -1 else str(cfg["item_count"])
    print(f"[INFO] 期望读取条数: {item_count_desc}")
    print(f"[INFO] 输出字段: {cfg['output_fields']}")

    hdfs_files = list_hdfs_part_files(cfg["hadoop_bin"], hdfs_dir)
    print(f"[INFO] 发现 {len(hdfs_files)} 个 part 文件")

    items = stream_read_items(cfg["hadoop_bin"], hdfs_files, cfg["item_count"],
                               cfg["output_fields"], cfg["geohash_precision"])
    print(f"[INFO] 实际读取到 {len(items)} 条数据")

    df = to_dataframe(items, cfg["output_fields"])

    # ── 过滤经纬度缺失的样本 ──────────────────────────────────────
    total_before = len(df)
    mask_no_geo = (df[LNG_FIELD].str.strip() == "") | (df[LAT_FIELD].str.strip() == "")
    no_geo_cnt = mask_no_geo.sum()
    df = df[~mask_no_geo].reset_index(drop=True)
    total_after = len(df)
    print(f"[INFO] 经纬度筛选: 原始 {total_before} 条 → 缺失经纬度 {no_geo_cnt} 条 → 保留 {total_after} 条"
          f"  (剔除率 {no_geo_cnt / total_before * 100:.1f}%)")

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), cfg["output_dir"]
    )
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"{cfg['country_code']}_item_recall.item.parquet"
    )

    df.to_parquet(output_path, index=False)
    print(f"[INFO] 结果已写入: {output_path}  (shape={df.shape})")


if __name__ == "__main__":
    main()