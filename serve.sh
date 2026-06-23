#!/usr/bin/env bash
# DeepSeek-V4-Flash (NVFP4) を SM120 (RTX PRO 6000) で serve する。
#
# 退役後に run_dsv4.sh が壊れていた(image名/ENTRYPOINT/MODEL_DIR/MoE backend が全部ズレ)ため、
# 動作する起動を一から再構成したもの。詳細な経緯は README.md。
#
# 核心の差分(なぜこの設定でないと動かないか):
#   - REAPなしの nvidia/DeepSeek-V4-Flash-NVFP4 (256 experts) を使う。REAP版(180B=160/162B=144)は
#     image の MoE kernel が "Unsupported expert number" で弾く。
#   - image dsv4-flash-sm120:local の ENTRYPOINT は ["vllm serve"] なので CMD は `serve /model ...` を
#     `--entrypoint vllm` で渡す(`vllm serve /model` だと二重になり死ぬ)。
#   - SM120 には NVFP4 MoE の native backend が無い(FLASHINFER_TRTLLM/CUTEDSL は family-100 gate)。
#     VLLM_TEST_FORCE_FP8_MARLIN=1 で MARLIN backend を強制 → weight-only FP4 dequant で動く(遅め)。
#   - --enable-expert-parallel は付けない(EP有無に関わらず MoE backend 問題は MARLIN で解決)。
#
# usage: ./serve.sh           # fp8 KV で起動
#        KV=fp8 ./serve.sh
set -uo pipefail

MODEL_DIR="${MODEL_DIR:-./models/DeepSeek-V4-Flash-NVFP4}"
IMAGE="${IMAGE:-dsv4-flash-sm120:local}"
PORT="${PORT:-8000}"
NAME="${NAME:-dsv4}"
KV="${KV:-fp8}"            # この image は nvfp4 KV 非対応(CacheConfig は fp8 まで)
MAXLEN="${MAXLEN:-131072}"
UTIL="${UTIL:-0.95}"

if [ ! -f "$MODEL_DIR/model.safetensors.index.json" ]; then
  echo "!! model weights not found in $MODEL_DIR" >&2
  echo "   hf download nvidia/DeepSeek-V4-Flash-NVFP4 --local-dir $MODEL_DIR" >&2
  exit 1
fi

docker rm -f "$NAME" 2>/dev/null

docker run -d --name "$NAME" \
  --gpus all --ipc=host --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_TRITON_MLA_SPARSE=1 \
  -e VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE=256 \
  -e VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE=128 \
  -e VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0 \
  -e VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE=4 \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 \
  -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo \
  -e NCCL_DEBUG=WARN -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "$MODEL_DIR":/model:ro \
  --network host \
  --entrypoint vllm "$IMAGE" \
  serve /model \
    --served-model-name deepseek-v4-flash \
    --host 0.0.0.0 --port "$PORT" \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --disable-custom-all-reduce \
    --kv-cache-dtype "$KV" \
    --block-size 256 \
    --gpu-memory-utilization "$UTIL" \
    --max-model-len "$MAXLEN" \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4

echo "==> launched '$NAME' on :$PORT  (KV=$KV, MoE backend=MARLIN, model=$MODEL_DIR)"
echo "    ready 待ち:  ./ready.sh    疎通:  ./smoke.sh    停止:  ./stop.sh"
