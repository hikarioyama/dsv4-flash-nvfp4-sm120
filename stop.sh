#!/usr/bin/env bash
NAME="${NAME:-dsv4}"
docker rm -f "$NAME" 2>/dev/null && echo "stopped $NAME" || echo "$NAME not running"
