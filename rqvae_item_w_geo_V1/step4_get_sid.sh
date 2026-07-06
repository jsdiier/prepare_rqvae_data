#!/bin/bash
set -e
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------------------------
# 用户可修改参数
# ------------------------
CKPT_PATH="./rq/rq_model/Jul-06-2026_14-54-41/best_collision_model.pth"
OUTPUT_FILE="./item_info/MX_item_recall.index.parquet"

# ===============================
# 自动识别环境
# ===============================
if [ -d "/nfs/dataset-ofs-rank-ssl" ]; then
    DATA_PREFIX="/nfs/dataset-ofs-rank-ssl"
elif [ -d "/tmp-data/prod_soda_trade_strategy/rank-ssl" ]; then
    DATA_PREFIX="/tmp-data/prod_soda_trade_strategy/rank-ssl"
elif [ -d "/home/luban/rank-ssl" ]; then
    DATA_PREFIX="/home/luban/rank-ssl"
else
    echo "❌ 未识别运行环境"
    exit 1
fi

VAE_ENV_PATH="${DATA_PREFIX}/chenpinyuan/miniconda_base/envs/RQ_VAE"
export PATH="${VAE_ENV_PATH}/bin:$PATH"
export PYTHONNOUSERSITE=1

# embedding 缓存策略需与 step3 保持一致（rq/datasets.py 读取）
export EMB_CACHE_DTYPE="${EMB_CACHE_DTYPE:-fp16}"
export EMB_CACHE_IN_MEMORY="${EMB_CACHE_IN_MEMORY:-1}"
echo ">>> EMB_CACHE_DTYPE=${EMB_CACHE_DTYPE}, EMB_CACHE_IN_MEMORY=${EMB_CACHE_IN_MEMORY}"

echo ">>> DATA_PREFIX=${DATA_PREFIX}"
echo ">>> Python=$(which python)"
python --version

cd "${SCRIPT_DIR}"

python rq/generate_indices.py \
    --ckpt_path "$CKPT_PATH" \
    --output_file "$OUTPUT_FILE"