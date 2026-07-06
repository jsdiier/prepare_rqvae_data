import collections
import logging
import argparse
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

from datasets import EmbDataset
from models.rqvae import RQVAE


def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


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
    logging.info(model)

    # -------------------- DataLoader --------------------
    data_loader = DataLoader(
        data,
        num_workers=ckpt_args.num_workers,
        batch_size=64,
        shuffle=False,
        pin_memory=True
    )
    logging.info("DataLoader prepared with batch_size=64, num_workers=%d", ckpt_args.num_workers)

    # -------------------- 生成索引 --------------------
    all_indices = []
    all_indices_str = []
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]

    logging.info("Start generating indices...")
    for d in tqdm(data_loader, total=len(data_loader)):
        d = d.to(device)
        indices = model.get_indices(d, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        for index in indices:
            code = [prefix[i].format(int(ind)) for i, ind in enumerate(index)]
            all_indices.append(code)
            all_indices_str.append(str(code))
    logging.info("Finished generating indices.")

    all_indices = np.array(all_indices)
    all_indices_str = np.array(all_indices_str)

    # -------------------- 冲突统计 --------------------
    total_items = len(all_indices_str)
    unique_items = len(set(all_indices_str.tolist()))
    collision_rate = (total_items - unique_items) / total_items
    max_conflicts = max(get_indices_count(all_indices_str).values())

    logging.info("Total indices: %d", total_items)
    logging.info("Unique indices: %d", unique_items)
    logging.info("Max number of conflicts: %d", max_conflicts)
    logging.info("Collision rate: %.6f", collision_rate)

    # -------------------- 保存为 parquet --------------------
    # 每个 codebook 层级单独一列: code_0, code_1, code_2, ...
    num_levels = all_indices.shape[1]
    rows = [
        {"idx": idx, **{f"code_{i}": code for i, code in enumerate(codes)}}
        for idx, codes in enumerate(all_indices.tolist())
    ]
    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    df.to_parquet(args.output_file, index=False)
    logging.info("Saved indices to: %s  (shape=%s)", args.output_file, df.shape)
    logging.info("Columns: %s", df.columns.tolist())