#!/usr/bin/env bash
set +B
unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS
SPEC_ARG=""
if [ "${SPEC:-1}" = "1" ]; then
  SPEC_ARG='--speculative-config {"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic","moe_backend":"b12x","use_local_argmax_reduction":true}'
fi
exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve /model \
  --served-model-name DeepSeek-V4-Flash --host 0.0.0.0 --port "${PORT:-8000}" \
  --kv-cache-dtype fp8 --block-size 256 --load-format safetensors \
  --tensor-parallel-size 2 --moe-backend b12x --linear-backend b12x \
  --gpu-memory-utilization "${UTIL:-0.93}" --max-model-len "${MAXLEN:-1048576}" --max-num-seqs 64 \
  --async-scheduling --no-scheduler-reserve-full-isl \
  --max-num-batched-tokens "${MNBT:-512}" --max-cudagraph-capture-size 192 \
  --attention-backend B12X_MLA_SPARSE --enable-chunked-prefill --enable-prefix-caching \
  --compilation-config='{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
  --enable-flashinfer-autotune \
  $SPEC_ARG
