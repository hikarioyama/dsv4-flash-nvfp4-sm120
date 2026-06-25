#!/usr/bin/env bash
# B12X (native SM120) ビルドで DSv4-Flash を TP2 起動 + MTP 有効。
# パッチ:
#  1. patches/nvfp4.py  — oracle gate で B12X+SILU clamp を許可
#  2. patches/quant_config.py — MTP draft experts を Mxfp4MoEMethod に路由
#     (MTP checkpoint は MXFP4 group_size=32/e8m0, NVFP4 group_size=16/e4m3 と不整合)
#
# Defaults serve a single 1,048,576-token (1M) context: MNBT=512 MAXLEN=1048576
# UTIL=0.93 (~19 min warmup). Why these values: docs/CONTEXT_LENGTH.md.
# Override for more VRAM headroom, e.g. balanced 256K:
#   UTIL=0.94 MAXLEN=262144 MNBT=512 ./serve_b12x_tp2.sh
# Old short-context behavior: MAXLEN=3072 MNBT=4096 ./serve_b12x_tp2.sh
set -uo pipefail
IMG="voipmonitor/vllm:chthonic-consecration-f1190eab-b12x0ff2847-pr20-cu132"
MODEL_DIR="${MODEL_DIR:-/mnt/data/models/DeepSeek-V4-Flash-NVFP4}"
PORT="${PORT:-8000}"; NAME="${NAME:-dsv4b12x}"
SPEC="${SPEC:-1}"   # 1=MTP有効, 0=なし
DIR="$(dirname "$0")"
INNER="$DIR/b12x_inner.sh"
PATCH_NVFP4="$DIR/patches/nvfp4.py"
PATCH_QC="$DIR/patches/quant_config.py"

docker rm -f "$NAME" 2>/dev/null

docker run -d --name "$NAME" --gpus all --runtime nvidia --ipc host --shm-size 32g --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODEL_DIR":/model:ro \
  -v "$INNER":/inner.sh:ro \
  -v "$PATCH_NVFP4":/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:ro \
  -v "$PATCH_QC":/opt/venv/lib/python3.12/site-packages/vllm/models/deepseek_v4/quant_config.py:ro \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUTE_DSL_ARCH=sm_120a \
  -e NCCL_IB_DISABLE=1 -e NCCL_P2P_LEVEL=SYS -e NCCL_PROTO=LL,LL128,Simple \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096 \
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_BREAKABLE_CUDAGRAPH=0 -e VLLM_USE_MEGA_AOT_ARTIFACT=1 \
  -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 -e B12X_MHC_MAX_TOKENS=16384 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_USE_B12X_WO_PROJECTION=1 -e VLLM_USE_B12X_MHC=1 \
  -e VLLM_USE_B12X_FP8_GEMM=1 -e VLLM_USE_B12X_MOE=1 -e VLLM_USE_B12X_SPARSE_INDEXER=1 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 -e VLLM_PCIE_ALLREDUCE_BACKEND=b12x -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e B12X_MLA_SM120_UNIFIED=1 -e USES_B12X=True -e B12X_DENSE_SPLITK_TURBO=1 -e B12X_W4A16_TC_DECODE=1 \
  -e SPEC="$SPEC" -e PORT="$PORT" -e UTIL="${UTIL:-0.93}" -e MAXLEN="${MAXLEN:-1048576}" -e MNBT="${MNBT:-512}" \
  --entrypoint /bin/bash "$IMG" /inner.sh

echo "==> launched $NAME (TP2, B12X, MTP=$SPEC, util=${UTIL:-0.93}, maxlen=${MAXLEN:-1048576}, mnbt=${MNBT:-512}) on :$PORT"
