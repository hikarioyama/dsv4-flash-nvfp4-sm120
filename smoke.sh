#!/usr/bin/env bash
# DSv4 疎通テスト(1リクエスト)。content と reasoning_content を表示。
set -uo pipefail
PORT="${PORT:-8000}"
curl -s -m120 "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Write a Python one-liner that returns the nth Fibonacci number. Then say DONE."}],"max_tokens":2048,"temperature":0.3}' \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print("content:",repr(m.get("content"))[:400]); print("reasoning:", "(yes)" if m.get("reasoning_content") else "(none)"); print("usage:",d.get("usage"))'
