#!/bin/sh
set -eu

root="/logs/wtw_hier_nav/go2_wtw_hier_nav_frozen_walk"
echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "root=$root"
if [ ! -d "$root" ]; then
  echo "status=missing_root"
  find /logs -maxdepth 3 -type d 2>/dev/null | sort | tail -80 || true
  exit 0
fi

run="$(find "$root" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -1 || true)"
if [ -z "$run" ]; then
  echo "status=no_runs"
  exit 0
fi

echo "status=ok"
echo "latest_run=$run"
echo "files:"
find "$run" -maxdepth 3 -type f 2>/dev/null | sort | tail -80
if [ -f "$run/heartbeat.json" ]; then
  echo "heartbeat:"
  cat "$run/heartbeat.json"
fi
if [ -f "$run/progress.log" ]; then
  echo "progress_tail:"
  tail -20 "$run/progress.log"
fi
if [ -f "$run/metrics.jsonl" ]; then
  echo "metrics_tail:"
  tail -20 "$run/metrics.jsonl"
fi
echo "checkpoints:"
find "$run/checkpoints" -maxdepth 1 -type f 2>/dev/null | sort | tail -30 || true
