import logging
import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from datasets import EmbDataset
from models.rqvae import RQVAE

PREFIX = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]
WRITE_BATCH_ROWS = 65536


def make_table(start_idx, codes_int, dedup_ranks):
    """codes_int: (n, L) int ndarray，dedup_ranks: (n,) 消歧位序号 → 带前缀字符串列的 Arrow Table"""
    n, num_levels = codes_int.shape
    arrays = [pa.array(np.arange(start_idx, start_idx + n), type=pa.int64())]
    names = ["idx"]
    for lvl in range(num_levels):
        col = [PREFIX[lvl].format(int(c)) for c in codes_int[:, lvl]]
        arrays.append(pa.array(col, type=pa.string()))
        names.append(f"code_{lvl}")
    col = [PREFIX[num_levels].format(int(r)) for r in dedup_ranks]
    arrays.append(pa.array(col, type=pa.string()))
    names.append(f"code_{num_levels}")
    return pa.Table.from_arrays(arrays, names=names)


def compute_dedup_ranks(all_codes):
    """TIGER 消歧位：相同 code 的 item 按原始顺序编 1..k，唯一 item 为 1"""
    _, inverse = np.unique(all_codes, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sorted_inv = inverse[order]
    group_starts = np.flatnonzero(np.r_[True, sorted_inv[1:] != sorted_inv[:-1]])
    group_sizes = np.diff(np.r_[group_starts, len(sorted_inv)])
    starts_full = np.repeat(group_starts, group_sizes)
    ranks = np.empty(len(all_codes), dtype=np.int32)
    ranks[order] = (np.arange(len(all_codes)) - starts_full + 1).astype(np.int32)
    return ranks


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    parser = argparse.ArgumentParser(description="Generate RQVAE indices")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True,
                        help="输出 parquet 路径，如 ./item_info/MX_item_recall.index.parquet")
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    logging.info("Using checkpoint: %s", args.ckpt_path)
    logging.info("Output file: %s", args.output_file)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    # -------------------- 加载 checkpoint --------------------
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt["args"]
    state_dict = ckpt["state_dict"]
    logging.info("Loaded checkpoint. data_path: %s", ckpt_args.data_path)

    # -------------------- 数据集 --------------------
    data = EmbDataset(ckpt_args.data_path)
    logging.info("Loaded dataset with %d samples, embedding dim: %d", len(data), data.dim)

    # -------------------- 模型 --------------------
    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=ckpt_args.num_emb_list,
        e_dim=ckpt_args.e_dim,
        layers=ckpt_args.layers,
        dropout_prob=ckpt_args.dropout_prob,
        bn=ckpt_args.bn,
        loss_type=ckpt_args.loss_type,
        quant_loss_weight=ckpt_args.quant_loss_weight,
        kmeans_init=ckpt_args.kmeans_init,
        kmeans_iters=ckpt_args.kmeans_iters,
        sk_epsilons=ckpt_args.sk_epsilons,
        sk_iters=ckpt_args.sk_iters,
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    logging.info("Model loaded and set to eval mode.")

    # -------------------- DataLoader --------------------
    data_loader = DataLoader(
        data,
        num_workers=ckpt_args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )
    logging.info("DataLoader prepared with batch_size=%d, num_workers=%d",
                 args.batch_size, ckpt_args.num_workers)

    # -------------------- 生成索引：先收集 int code 矩阵 --------------------
    # 消歧位需要全量 code 才能计算组内序号，先收集整型矩阵
    # （9M x 3 的 int32 仅 ~110MB），字符串列在写出阶段按批构建
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    all_codes = np.empty((len(data), 0), dtype=np.int32)  # 首批后按实际层数重建
    filled_rows = 0

    logging.info("Start generating indices...")
    with torch.no_grad():
        for d in tqdm(data_loader, total=len(data_loader)):
            d = d.to(device)
            indices = model.get_indices(d, use_sk=False)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy().astype(np.int32)

            if all_codes.shape[1] != indices.shape[1]:
                all_codes = np.empty((len(data), indices.shape[1]), dtype=np.int32)

            all_codes[filled_rows:filled_rows + len(indices)] = indices
            filled_rows += len(indices)
    logging.info("Finished generating indices. Rows: %d", filled_rows)

    # -------------------- 冲突统计（基于 int 矩阵，精确且省内存） --------------------
    _, counts = np.unique(all_codes, axis=0, return_counts=True)
    total_items = len(all_codes)
    unique_items = len(counts)
    collision_rate = (total_items - unique_items) / total_items
    max_conflicts = int(counts.max())

    logging.info("Total indices: %d", total_items)
    logging.info("Unique indices: %d", unique_items)
    logging.info("Max number of conflicts: %d", max_conflicts)
    logging.info("Collision rate: %.6f", collision_rate)

    # -------------------- TIGER 消歧位 + 流式写 parquet --------------------
    dedup_ranks = compute_dedup_ranks(all_codes)
    logging.info("Dedup level appended as code_%d, max rank: %d",
                 all_codes.shape[1], int(dedup_ranks.max()))

    writer = None
    for start in tqdm(range(0, len(all_codes), WRITE_BATCH_ROWS), desc="Writing parquet"):
        end = min(start + WRITE_BATCH_ROWS, len(all_codes))
        table = make_table(start, all_codes[start:end], dedup_ranks[start:end])
        if writer is None:
            writer = pq.ParquetWriter(args.output_file, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()
    logging.info("Saved indices to: %s", args.output_file)
