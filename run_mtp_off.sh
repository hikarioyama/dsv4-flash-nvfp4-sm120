#!/usr/bin/env bash
# Stream a ~1000-token completion with MTP OFF (plain decode, same B12X build).
# Restarts the server into SPEC=0 first if it isn't already (~10 min).
exec "$(cd "$(dirname "$0")" && pwd)/mtp_run.sh" off
