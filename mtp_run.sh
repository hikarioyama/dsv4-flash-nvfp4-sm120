#!/usr/bin/env bash
# Ensure the dsv4b12x server is in the requested MTP mode, then stream a single
# ~1000-token completion live and print TTFT + tok/s on the final line.
#
# Usage: mtp_run.sh <on|off>
#   on  -> SPEC=1 (MTP / speculative decoding enabled)
#   off -> SPEC=0 (plain decode, same B12X build, apples-to-apples baseline)
#
# If the server is already in the requested mode and healthy, it streams
# immediately. Otherwise it restarts the server in the right mode first
# (~10 min: weight load + torch.compile + warmup + speculator capture).
set -uo pipefail

MODE="${1:-}"
case "$MODE" in
  on)  WANT=1; TAG="MTP ON " ;;
  off) WANT=0; TAG="MTP OFF" ;;
  *)   echo "usage: $(basename "$0") <on|off>" >&2; exit 2 ;;
esac

DIR="$(cd "$(dirname "$0")" && pwd)"
NAME="${NAME:-dsv4b12x}"
PORT="${PORT:-8000}"
SERVE="$DIR/serve_b12x_tp2.sh"

cur="$(docker inspect "$NAME" --format '{{range .Config.Env}}{{println .}}{{end}}' 2>/dev/null | sed -n 's/^SPEC=//p')"
running="$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || echo false)"
health="$(curl -s -m3 "http://localhost:$PORT/health" -o /dev/null -w '%{http_code}' 2>/dev/null || echo 000)"

if [ "$running" = "true" ] && [ "$cur" = "$WANT" ] && [ "$health" = "200" ]; then
  echo ">> server already in [$TAG] mode (SPEC=$cur), healthy -- streaming now."
else
  echo ">> server not in [$TAG] mode (running=$running SPEC=${cur:-?} health=$health)."
  echo ">> restarting in [$TAG] mode (SPEC=$WANT). This takes ~10 min..."
  SPEC="$WANT" PORT="$PORT" NAME="$NAME" "$SERVE" >/dev/null
  printf ">> waiting for health "
  ok=0
  for i in $(seq 1 180); do
    code="$(curl -s -m3 "http://localhost:$PORT/health" -o /dev/null -w '%{http_code}' 2>/dev/null || echo 000)"
    if [ "$code" = "200" ]; then echo " ready (~$((i*5))s)"; ok=1; break; fi
    printf "."; sleep 5
  done
  if [ "$ok" != "1" ]; then
    echo; echo "!! server did not become healthy in ~15 min. Check: docker logs $NAME" >&2
    exit 1
  fi
fi

python3 "$DIR/mtp_stream.py" --port "$PORT" --max-tokens "${MAXTOK:-1000}" --tag "$TAG"

if [ "$WANT" = "1" ]; then
  m="$(docker logs --since 120s "$NAME" 2>&1 | grep -i 'SpecDecoding metrics' | tail -1)"
  [ -n "$m" ] && echo " acceptance ->${m#*SpecDecoding metrics:}"
fi
