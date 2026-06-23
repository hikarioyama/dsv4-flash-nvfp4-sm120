#!/usr/bin/env bash
# serve.sh の MTP (Multi-Token Prediction / 投機デコード) 有効版。
# DSv4 は内蔵 MTP module (num_nextn_predict_layers=1, mtp.0.* weight 同梱) を draft に使う。
#
# vLLM の method は 'mtp'(旧 'deepseek_mtp' は deprecated→自動置換)。
# num_speculative_tokens=K を SPEC_K で変える(MTP module は1層なので K>1 は同じ head を多段適用)。
#
# usage: SPEC_K=1 ./serve-mtp.sh      # default K=1
#
# STATUS: this boots cleanly, but the MTP DRAFT IS BROKEN on SM120 — ~0%
# acceptance (garbage draft logits). Full debugging log, including why the VRAM
# concern was a misdiagnosis and the current int8-vs-uint8 root-cause lead, is in
# docs/MTP_INVESTIGATION.md. Use serve.sh for real work.
#
# The config below is the *fixed* one (the old script set
# VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0, which reserved zero memory for CUDA
# graphs and OOM'd at capture time — read as "MTP ate my VRAM"). With the fix
# (cudagraph estimate on, max-num-batched-tokens 2048, expandable_segments) it
# boots with KV = 546,872 tokens @ util 0.95, *above* the no-MTP baseline.
set -uo pipefail
MODEL_DIR="${MODEL_DIR:-./models/DeepSeek-V4-Flash-NVFP4}"
IMAGE="${IMAGE:-dsv4-flash-sm120:local}"
PORT="${PORT:-8000}"; NAME="${NAME:-dsv4}"; KV="${KV:-fp8}"
MAXLEN="${MAXLEN:-131072}"; UTIL="${UTIL:-0.95}"; SPEC_K="${SPEC_K:-1}"
MAXBATCH="${MAXBATCH:-2048}"   # MTP の activation peak を抑えて KV/cudagraph に余地を作る(8192→2048)

[ -f "$MODEL_DIR/model.safetensors.index.json" ] || { echo "!! weights not found: $MODEL_DIR" >&2; exit 1; }
docker rm -f "$NAME" 2>/dev/null

docker run -d --name "$NAME" --gpus all --ipc=host --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e VLLM_TEST_FORCE_FP8_MARLIN=1 \
  -e VLLM_TRITON_MLA_SPARSE=1 -e VLLM_TRITON_MLA_SPARSE_TOPK_CHUNK_SIZE=256 \
  -e VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE=128 -e VLLM_TRITON_MLA_SPARSE_MATMUL_DECODE=0 \
  -e VLLM_TRITON_MLA_SPARSE_HEAD_BLOCK_SIZE=4 -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_P2P_DISABLE=1 -e NCCL_IB_DISABLE=1 -e NCCL_SOCKET_IFNAME=lo -e GLOO_SOCKET_IFNAME=lo \
  -e NCCL_DEBUG=WARN -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -v "$MODEL_DIR":/model:ro --network host \
  --entrypoint vllm "$IMAGE" \
  serve /model --served-model-name deepseek-v4-flash \
    --host 0.0.0.0 --port "$PORT" --trust-remote-code \
    --tensor-parallel-size 2 --disable-custom-all-reduce \
    --kv-cache-dtype "$KV" --block-size 256 \
    --gpu-memory-utilization "$UTIL" --max-model-len "$MAXLEN" \
    --max-num-batched-tokens "$MAXBATCH" \
    --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
    --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${SPEC_K}}"

echo "==> launched '$NAME' MTP K=$SPEC_K on :$PORT (KV=$KV, MoE=MARLIN)"
echo "    ready: ./ready.sh   throughput: ./bench_throughput.sh   accept率: ./mtp_accept.sh"
