import argparse
import random
import torch
import numpy as np
from time import time
import logging

from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
from trainer import  Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=5000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=50, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument('--lr_scheduler_type', type=str, default="constant", help='scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=50, help='warmup epochs')
    parser.add_argument("--data_path", type=str,
                        default="../data/Games/Games.emb-llama-td.npy",
                        help="Input data path.")

    parser.add_argument("--weight_decay", type=float, default=0.0, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument("--kmeans_init_samples", type=int, default=100_000,
                        help="训练前用于逐层 kmeans 初始化码本的样本数；"
                             "0 表示沿用旧行为(用第一个 batch 初始化，样本少易产生死码)")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument("--beta", type=float, default=0.25, help="Beta for commitment loss")
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')

    parser.add_argument('--save_limit', type=int, default=5)
    parser.add_argument("--ckpt_dir", type=str, default="", help="output directory for model")

    return parser.parse_args()


def init_rq_codebooks(model, dataset, sample_n, device):
    """
    训练前用大样本逐层初始化 RQ 码本：抽 sample_n 条 → encoder →
    逐层 kmeans 初始化 + 量化取残差喂给下一层。
    语义与 vq.py 里"第一个 batch 顺手初始化"相同，但样本量大得多——
    batch 级样本(如 1024 条聚 256 类)统计上不足，会从第 0 步就产生大量死码。
    初始化完成后 initted=True，训练时旧的 batch 初始化逻辑自动跳过。
    """
    n = len(dataset)
    sample_n = min(sample_n, n)
    rng = np.random.default_rng(2024)
    idx = np.sort(rng.choice(n, size=sample_n, replace=False))
    emb = np.asarray(dataset.embeddings[idx], dtype=np.float32)

    model.eval()
    with torch.no_grad():
        latents = []
        for s in range(0, len(emb), 8192):
            batch = torch.from_numpy(emb[s:s + 8192]).to(device)
            latents.append(model.encoder(batch))
        residual = torch.cat(latents)
        for i, quantizer in enumerate(model.rq.vq_layers):
            t0 = time()
            quantizer.init_emb(residual)
            x_res, _, _ = quantizer(residual, use_sk=False)
            residual = residual - x_res
            print(f"[kmeans init] level {i}: {sample_n} samples, "
                  f"耗时 {time() - t0:.1f}s")
    model.train()


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    print("=================================================")
    print(args)
    print("=================================================")

    logging.basicConfig(level=logging.DEBUG)

    """build dataset"""
    data = EmbDataset(args.data_path)
    model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  beta=args.beta,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  )
    print(model)
    data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)
    trainer = Trainer(args,model, len(data_loader))
    if args.kmeans_init and args.kmeans_init_samples > 0:
        init_rq_codebooks(model, data, args.kmeans_init_samples, trainer.device)
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("Best Loss",best_loss)
    print("Best Collision Rate", best_collision_rate)

