#!/usr/bin/env bash
# Stream a ~1000-token completion with MTP ON (speculative decoding enabled).
# Restarts the server into SPEC=1 first if it isn't already.
exec "$(cd "$(dirname "$0")" && pwd)/mtp_run.sh" on
