#!/usr/bin/env bash
# MTP の acceptance rate を /metrics から取得。
# 先に数リクエスト投げて統計を溜めてから読む。
set -uo pipefail
PORT="${PORT:-8000}"; EP="http://127.0.0.1:${PORT}/v1/chat/completions"
# warmup: 生成を数回回して spec decode 統計を溜める
for i in 1 2 3; do
  curl -s -m120 "$EP" -H 'Content-Type: application/json' \
    -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Write a detailed 300-word explanation of how TCP congestion control works."}],"max_tokens":400,"temperature":0.0}' >/dev/null 2>&1
done
echo "=== spec decode / MTP metrics ==="
curl -s "http://127.0.0.1:${PORT}/metrics" 2>/dev/null | grep -iE "spec_decode|accept|draft" | grep -v '^#' | head -20
echo "=== acceptance rate 計算 ==="
# 注意: vllm の *_created メトリクスは Unix タイムスタンプ(値が 1.7e9)。合算すると偽の100%になる。
# accepted/draft は _total サフィックスだけを使う。
curl -s "http://127.0.0.1:${PORT}/metrics" 2>/dev/null | python3 -c "
import sys,re
acc=draft=0.0
for ln in sys.stdin:
    if ln.startswith('#'): continue
    m=re.match(r'(\S+?)(\{[^}]*\})?\s+([0-9.e+-]+)\s*\$', ln)
    if not m: continue
    k,v=m.group(1),float(m.group(3))
    if k=='vllm:spec_decode_num_accepted_tokens_total': acc+=v
    if k=='vllm:spec_decode_num_draft_tokens_total':    draft+=v
if draft>0: print(f'  accepted={acc:.0f} draft={draft:.0f} -> acceptance {acc/draft*100:.1f}%  (mean accept length ~ {1+acc/draft:.2f}x)')
else: print('  (spec decode metrics 未取得 or 0)')
"
echo "  ※ vLLM 自身の窓計測も参照: docker logs --since 60s dsv4 | grep 'SpecDecoding metrics'"
