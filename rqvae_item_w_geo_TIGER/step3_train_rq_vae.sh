#!/bin/bash
set -e
export PYTHONNOUSERSITE=1
# nohup 重定向到文件时禁用 stdout 块缓冲，否则 sklearn 等库的逐行日志
# 会攒满 4KB 才批量刷出，看起来像卡住
export PYTHONUNBUFFERED=1

echo "================ 当前执行脚本内容 ================"
cat "$0"
echo "=================================================="

# ------------------------------
# 1️⃣ 读取 common.conf 中 [rqvae_train] 段的训练参数
# ------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="${1:-${SCRIPT_DIR}/common.conf}"
echo ">>> conf_file=${CONF_FILE}"

get_conf() {
    awk -F '=' -v sec="$1" -v key="$2" '
        /^\[/ { in_sec = ($0 == "[" sec "]") }
        in_sec && !/^[ \t]*[#;]/ && /=/ {
            k = $1; gsub(/^[ \t]+|[ \t]+$/, "", k)
            if (k == key) {
                v = substr($0, index($0, "=") + 1)
                gsub(/^[ \t]+|[ \t]+$/, "", v)
                print v; exit
            }
        }
    ' "${CONF_FILE}"
}

DATA_PATH="$(get_conf rqvae_train data_path)"
CKPT_DIR="$(get_conf rqvae_train ckpt_dir)"
LR="$(get_conf rqvae_train lr)"
EPOCHS="$(get_conf rqvae_train epochs)"
WARMUP_EPOCHS="$(get_conf rqvae_train warmup_epochs)"
EVAL_STEP="$(get_conf rqvae_train eval_step)"
SK_EPSILONS="$(get_conf rqvae_train sk_epsilons)"
E_DIM="$(get_conf rqvae_train e_dim)"
LAYERS="$(get_conf rqvae_train layers)"
BATCH_SIZE="$(get_conf rqvae_train batch_size)"

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

# ------------------------------
# FULL_MEMORY=1（默认）: 27G 全量整载内存训练，速度最快，需要 Pod 有足够空闲内存；
# FULL_MEMORY=0        : 顺序分块读 + shuffle buffer 流式训练，内存 ~2G，
#                        邻居任务挤占内存、整载被 OOM kill 时用这个。
# 用法: FULL_MEMORY=0 nohup bash step3_train_rq_vae.sh > train_rqvae.log 2>&1 &
# ------------------------------
FULL_MEMORY="${FULL_MEMORY:-1}"
EXTRA_ARGS=""
if [ "${FULL_MEMORY}" = "0" ]; then
    EXTRA_ARGS="--streaming"
    echo ">>> FULL_MEMORY=0，启用流式分块训练（低内存）"
else
    echo ">>> FULL_MEMORY=1，整载内存训练"
fi

echo "当前使用的 Python 路径: $(which python)"
echo "开始训练..."

# ------------------------------
# 4️⃣ 启动训练
# ------------------------------
# 把 PID 写入文件，方便直接终止训练：kill $(cat train_rqvae.pid)
# exec 会让 python 替换当前 bash 进程，因此该 PID 就是训练进程本身，
# 杀掉它即可（DataLoader worker 检测到主进程退出会自动跟随退出）
echo $$ > train_rqvae.pid
echo ">>> 训练 PID: $$ (已写入 train_rqvae.pid，终止: kill \$(cat train_rqvae.pid))"

# 训练参数统一在 common.conf 的 [rqvae_train] 段配置
# SK_EPSILONS / LAYERS 是空格分隔的多值参数，不能加引号，需依赖 word-splitting
exec python rq/rqvae.py \
  --data_path "${DATA_PATH}" \
  --ckpt_dir "${CKPT_DIR}" \
  --lr "${LR}" \
  --epochs "${EPOCHS}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --eval_step "${EVAL_STEP}" \
  --sk_epsilons ${SK_EPSILONS} \
  --e_dim "${E_DIM}" \
  --layers ${LAYERS} \
  --batch_size "${BATCH_SIZE}" \
  ${EXTRA_ARGS}