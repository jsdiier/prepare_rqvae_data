#!/bin/bash
set -e
export PYTHONNOUSERSITE=1

echo "================ 当前执行脚本内容 ================"
cat "$0"
echo "=================================================="

# ------------------------------
# 2️⃣ 动态识别根目录
# ------------------------------
if [ -d "/nfs/dataset-ofs-rank-ssl" ]; then
    DATA_PREFIX="/nfs/dataset-ofs-rank-ssl"
    echo ">>> 检测到训练平台环境 (NFS)，使用路径: ${DATA_PREFIX}"
else
    DATA_PREFIX="/home/luban/rank-ssl"
    echo ">>> 检测到本地 SSH 环境，使用路径: ${DATA_PREFIX}"
fi

# ------------------------------
# 3️⃣ 激活虚拟环境
# ------------------------------
VAE_ENV_PATH="${DATA_PREFIX}/chenpinyuan/miniconda_base/envs/RQ_VAE"
export PATH="${VAE_ENV_PATH}/bin:$PATH"

# ------------------------------
# embedding 缓存策略（rq/datasets.py 读取）
# fp16: 缓存减半(27G)，可整载进 Pod 内存；IN_MEMORY=1: 强制整载，避开 NFS mmap 随机读
# ------------------------------
export EMB_CACHE_DTYPE="${EMB_CACHE_DTYPE:-fp16}"
export EMB_CACHE_IN_MEMORY="${EMB_CACHE_IN_MEMORY:-1}"
echo ">>> EMB_CACHE_DTYPE=${EMB_CACHE_DTYPE}, EMB_CACHE_IN_MEMORY=${EMB_CACHE_IN_MEMORY}"

echo "当前使用的 Python 路径: $(which python)"
echo "开始训练..."

# ------------------------------
# 4️⃣ 启动训练
# ------------------------------
python rq/rqvae.py \
  --data_path ./item_info/item_emb.parquet \
  --ckpt_dir ./rq/rq_model \
  --lr 1e-4 \
  --epochs 5 \
  --batch_size 512