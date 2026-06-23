#!/usr/bin/env bash
# DSv4 throughput sweep (single + concurrent). Works against either server
# (with or without MTP) — it just hits the OpenAI-compatible endpoint on :8000.
# Self-contained: only needs bash + curl + python3.
#
# usage: ./bench_throughput.sh "1 4 8"   [max_tokens]
set -uo pipefail
NS="${1:-1 4 8}"; MAXTOK="${2:-512}"
PORT="${PORT:-8000}"; MODEL="${MODEL:-deepseek-v4-flash}"
EP="http://127.0.0.1:${PORT}/v1/chat/completions"
PROMPT="${PROMPT:-Write a detailed technical explanation of how a B-tree database index works, including insertion and rebalancing.}"

req() {  # $1=result file
  local t0 t1 body ct
  t0=$(date +%s.%N)
  body=$(curl -s -m300 "$EP" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAXTOK,\"temperature\":0.0}")
  t1=$(date +%s.%N)
  ct=$(printf '%s' "$body" | python3 -c "import sys,json;print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
  printf '%s %s %s\n' "$ct" "$t0" "$t1" > "$1"
}

echo "=== DSv4 throughput (max_tokens=$MAXTOK) ==="
for N in $NS; do
  tmp=$(mktemp -d)
  for i in $(seq 1 "$N"); do req "$tmp/r_$i" & done
  wait
  python3 - "$tmp" "$N" <<'PY'
import sys, glob, os
tmp, N = sys.argv[1], int(sys.argv[2])
tok=[]; t0=[]; t1=[]
for f in glob.glob(os.path.join(tmp, "r_*")):
    parts = open(f).read().split()
    if len(parts) == 3:
        tok.append(int(parts[0])); t0.append(float(parts[1])); t1.append(float(parts[2]))
total = sum(tok)
span = (max(t1) - min(t0)) if t0 else 0
agg = total/span if span > 0 else 0
per = sum(t/(b-a) for t,a,b in zip(tok,t0,t1) if b>a)/max(len(tok),1)
print(f"  N={N}: aggregate {agg:6.1f} t/s | per-stream {per:5.1f} t/s | total {total} tok / {span:.1f}s")
PY
  rm -rf "$tmp"
done
