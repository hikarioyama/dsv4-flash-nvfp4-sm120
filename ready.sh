#!/usr/bin/env bash
# DSv4 サーバの ready 待ち + KV pool / 起動健全性の表示。
set -uo pipefail
PORT="${PORT:-8000}"; NAME="${NAME:-dsv4}"
for i in $(seq 1 90); do
  if curl -s -m5 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q deepseek-v4-flash; then
    echo "READY at $(date +%H:%M:%S)"
    docker logs "$NAME" 2>&1 | grep -oiE "GPU KV cache size: [0-9,]+ tokens" | tail -1
    docker logs "$NAME" 2>&1 | grep -oiE "Maximum concurrency for [0-9,]+ tokens.*" | tail -1
    exit 0
  fi
  if ! docker ps --filter "name=$NAME" --format '{{.Names}}' | grep -q "$NAME"; then
    echo "!! container DIED"; docker logs "$NAME" 2>&1 | grep -iE "error:|runtimeerror|notimpl|cuda|out of mem|assert|nan|marlin|expert" | grep -v "888]   " | tail -12
    exit 1
  fi
  sleep 10
done
echo "!! timeout"; docker logs "$NAME" 2>&1 | tail -15; exit 1
