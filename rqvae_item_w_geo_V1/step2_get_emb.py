#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取 item_info/{country_code}_item_recall.item.parquet，
将每个 item 的 brand/title/description 拼接成一段文本，
过 GME (gme-Qwen2-VL-2B-Instruct) 模型得到 embedding，
打印前 log_sample_count 条样本的调试信息，
按 batch 流式写入 item_info/item_emb.parquet（不在内存中累积全部结果）。

用法:
    python3 get_emb.py [common.conf]
"""

import os
import sys
import logging
import configparser

import torch
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def load_config(conf_path: str) -> dict:
    cp = configparser.ConfigParser()
    if not cp.read(conf_path, encoding="utf-8"):
        raise FileNotFoundError(f"找不到配置文件: {conf_path}")

    base_dir = os.path.dirname(os.path.abspath(conf_path))

    cfg = {
        "model_path": cp.get("embedding", "model_path"),
        "embed_dim": cp.getint("embedding", "embed_dim", fallback=1536),
        "batch_size": cp.getint("embedding", "batch_size", fallback=8),
        "device": cp.get("embedding", "device", fallback="cuda"),
        "max_length": cp.getint("embedding", "max_length", fallback=512),
        "log_sample_count": cp.getint("embedding", "log_sample_count", fallback=5),
        "item_json": cp.get("embedding", "item_json"),
        # 输出改为 parquet 流式写入；为兼容旧配置，若只配置了 output_npy 也可自动改后缀
        "output_parquet": cp.get(
            "embedding", "output_parquet",
            fallback=os.path.splitext(cp.get("embedding", "output_npy", fallback="item_info/item_emb.parquet"))[0] + ".parquet"
        ),
        "write_batch_rows": cp.getint("embedding", "write_batch_rows", fallback=1),
    }

    # 相对路径统一相对于 conf 文件所在目录解析
    for key in ("item_json", "output_parquet"):
        if not os.path.isabs(cfg[key]):
            cfg[key] = os.path.join(base_dir, cfg[key])

    return cfg


def load_items(item_parquet_path: str):
    df = pd.read_parquet(item_parquet_path)

    items = df.to_dict(orient="records")
    ordered_keys = [str(i) for i in range(len(items))]
    # 若原始 item 表里有 item_id，一并带出，方便与 embedding 对齐核对
    item_ids = [str(item.get("item_id", "")) for item in items]

    texts = []
    for item in items:
        brand = (item.get("brand") or "").strip()
        title = (item.get("title") or "").strip()
        description = (item.get("description") or "").strip()
        text = (brand + " " + title + " " + description).strip()
        texts.append(text)

    return ordered_keys, item_ids, items, texts


def load_gme_model(model_path: str, device: str):
    """
    按照 GME (gme-Qwen2-VL-2B-Instruct) 实际可用的方式加载：
    AutoTokenizer + AutoModel(trust_remote_code=True)，fp16 推理。
    """
    import time

    t0 = time.time()
    logging.info(f"[1/3] 开始加载 tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",
        use_fast=False,
    )
    logging.info(f"[1/3] tokenizer 加载完成，耗时 {time.time() - t0:.1f}s")

    t1 = time.time()
    logging.info(
        f"[2/3] 开始加载模型权重: {model_path} "
        f"（首次加载可能需要 1-3 分钟，取决于磁盘/显卡速度，请耐心等待）"
    )
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    logging.info(f"[2/3] 模型权重加载完成，耗时 {time.time() - t1:.1f}s，开始搬运到 {device}")

    t2 = time.time()
    model = model.to(device)
    model.eval()
    logging.info(f"[3/3] 模型已搬运到 {device} 并切换 eval 模式，耗时 {time.time() - t2:.1f}s")

    return tokenizer, model


def encode_texts(tokenizer, model, texts: list, device: str, max_length: int) -> np.ndarray:
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.float().cpu().numpy()

    return emb


def log_debug_samples(texts: list, emb: np.ndarray, printed_cnt: int, log_sample_count: int):
    """按照要求的格式打印调试样本：input / first5 / last5 / norm"""
    remaining = log_sample_count - printed_cnt
    if remaining <= 0:
        return printed_cnt

    cur = min(remaining, len(texts))
    for i in range(cur):
        logging.info(
            f"[Embedding Debug] "
            f"sample={printed_cnt + i}\n"
            f"input={texts[i][:300]}\n"
            f"first5={emb[i][:5]}\n"
            f"last5={emb[i][-5:]}\n"
            f"norm={np.linalg.norm(emb[i]):.4f}\n"
            f"{'-' * 60}"
        )
    return printed_cnt + cur


def main():
    conf_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "common.conf"
    )
    cfg = load_config(conf_path)

    logging.info(f"python 解释器: {sys.executable}")
    logging.info(f"torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logging.info(f"模型路径: {cfg['model_path']}")
    logging.info(f"item json: {cfg['item_json']}")
    logging.info(f"输出 parquet: {cfg['output_parquet']}")

    ordered_keys, item_ids, items, texts = load_items(cfg["item_json"])
    logging.info(f"共读取 {len(texts)} 条 item 文本")

    tokenizer, model = load_gme_model(cfg["model_path"], cfg["device"])

    batch_size = cfg["batch_size"]
    log_sample_count = cfg["log_sample_count"]
    max_length = cfg["max_length"]

    output_dir = os.path.dirname(cfg["output_parquet"])
    os.makedirs(output_dir, exist_ok=True)

    # 若已存在旧文件，先删除，避免 ParquetWriter 追加到脏文件
    if os.path.exists(cfg["output_parquet"]):
        os.remove(cfg["output_parquet"])

    printed_cnt = 0
    total_written = 0
    writer = None

    try:
        for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[start:start + batch_size]
            batch_row_ids = ordered_keys[start:start + batch_size]
            batch_item_ids = item_ids[start:start + batch_size]

            emb = encode_texts(tokenizer, model, batch_texts, cfg["device"], max_length)

            if emb.shape[-1] != cfg["embed_dim"]:
                logging.warning(
                    f"embedding 维度({emb.shape[-1]}) 与配置的 embed_dim({cfg['embed_dim']}) 不一致，"
                    f"请检查模型/配置"
                )

            printed_cnt = log_debug_samples(batch_texts, emb, printed_cnt, log_sample_count)

            # 构造本 batch 的 Arrow Table 并流式写入，不在内存中累积历史 batch
            emb_list_col = pa.array(emb.astype(np.float32).tolist(), type=pa.list_(pa.float32()))
            table = pa.Table.from_arrays(
                [
                    pa.array(batch_row_ids, type=pa.string()),
                    pa.array(batch_item_ids, type=pa.string()),
                    emb_list_col,
                ],
                names=["row_id", "item_id", "embedding"],
            )

            if writer is None:
                writer = pq.ParquetWriter(cfg["output_parquet"], table.schema)
            writer.write_table(table)

            total_written += len(batch_texts)

            # 及时释放本 batch 的引用，帮助 GC
            del emb, table, emb_list_col
    finally:
        if writer is not None:
            writer.close()

    logging.info(f"流式写入完成，共写入 {total_written} 条 embedding")
    logging.info(f"embedding 已保存到: {cfg['output_parquet']}")


if __name__ == "__main__":
    main()