#!/bin/bash
# 调用 get_emb.py，对 item_info 下的 item json 计算 GME embedding
# bash step2_get_emb.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="${1:-${SCRIPT_DIR}/common.conf}"
GPU_ID="${2:-0}"

echo ">>> conf_file=${CONF_FILE}"
echo ">>> gpu_id=${GPU_ID}"

# ===============================
# 0️⃣ 信号捕获：退出时杀死所有子进程
# ===============================
# 当脚本因为任何原因退出（正常、报错或被 kill）时，一并杀死所有后台子进程
trap 'echo "▶ 脚本退出，正在清理子进程..."; kill 0' EXIT

# ===============================
# 1️⃣ 环境变量
# ===============================
export NCCL_IB_DISABLE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"


# ===============================
# 2️⃣ 自动识别环境
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

echo ">>> DATA_PREFIX=${DATA_PREFIX}"


# ===============================
# 3️⃣ conda环境
# ===============================
VENV_PATH="${DATA_PREFIX}/chenpinyuan/miniconda_base/envs/gme"

export PATH="${VENV_PATH}/bin:$PATH"
export CONDA_PREFIX="${VENV_PATH}"

echo ">>> Python=$(which python)"
python --version


# ===============================
# 4️⃣ 执行
# ===============================
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/get_emb_$(date +%Y%m%d_%H%M%S).log"
echo ">>> log_file=${LOG_FILE}"

cd "${SCRIPT_DIR}"

python -u step2_get_emb.py "${CONF_FILE}" 2>&1 | tee "${LOG_FILE}"

echo ">>> step2_get_emb.py 执行完成，日志已保存到 ${LOG_FILE}"