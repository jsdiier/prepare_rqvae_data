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


def make_table(start_idx, codes_int):
    """codes_int: (n, L) int ndarray → 带前缀字符串列的 Arrow Table"""
    n, num_levels = codes_int.shape
    arrays = [pa.array(np.arange(start_idx, start_idx + n), type=pa.int64())]
    names = ["idx"]
    for lvl in range(num_levels):
        col = [PREFIX[lvl].format(int(c)) for c in codes_int[:, lvl]]
        arrays.append(pa.array(col, type=pa.string()))
        names.append(f"code_{lvl}")
    return pa.Table.from_arrays(arrays, names=names)


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

    # -------------------- 生成索引：流式写 parquet --------------------
    # 冲突统计只需要整型 code 矩阵（9M x 3 的 int32 仅 ~110MB），
    # 字符串结果按 WRITE_BATCH_ROWS 行一批流式写出，不在内存中累积
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    all_codes = np.empty((len(data), 0), dtype=np.int32)  # 首批后按实际层数重建
    writer = None
    pending = []
    pending_rows = 0
    written_rows = 0

    def flush():
        global writer, pending, pending_rows, written_rows
        if not pending:
            return
        codes_int = np.concatenate(pending, axis=0)
        table = make_table(written_rows, codes_int)
        if writer is None:
            writer = pq.ParquetWriter(args.output_file, table.schema)
        writer.write_table(table)
        written_rows += len(codes_int)
        pending = []
        pending_rows = 0

    logging.info("Start generating indices...")
    with torch.no_grad():
        for d in tqdm(data_loader, total=len(data_loader)):
            d = d.to(device)
            indices = model.get_indices(d, use_sk=False)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy().astype(np.int32)

            if all_codes.shape[1] != indices.shape[1]:
                all_codes = np.empty((len(data), indices.shape[1]), dtype=np.int32)

            row_start = written_rows + pending_rows
            all_codes[row_start:row_start + len(indices)] = indices

            pending.append(indices)
            pending_rows += len(indices)
            if pending_rows >= WRITE_BATCH_ROWS:
                flush()
    flush()
    if writer is not None:
        writer.close()
    logging.info("Finished generating indices. Rows written: %d", written_rows)

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
    logging.info("Saved indices to: %s", args.output_file)
