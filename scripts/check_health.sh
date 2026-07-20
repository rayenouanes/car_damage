#!/usr/bin/env bash
set -euo pipefail

HOST=${1:-http://127.0.0.1:8000}
HEALTH_URL="$HOST/health"

echo "Checking API health at $HEALTH_URL"
if command -v jq >/dev/null 2>&1; then
  curl -s "$HEALTH_URL" | jq .
else
  curl -sS "$HEALTH_URL" || echo "Failed to reach $HEALTH_URL"
fi

# Basic status check
STATUS=$(curl -sS "$HEALTH_URL" | python -c "import sys,json;print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "unknown")
echo "status: $STATUS"
if [ "$STATUS" != "ok" ]; then
  echo "Health check reports non-ok status." >&2
  exit 1
fi

echo "API health OK."