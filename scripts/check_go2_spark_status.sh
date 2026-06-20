#!/usr/bin/env bash
set -euo pipefail

device="${GO2_SPARK_DEVICE:-spark-3011.local}"
fallback_device="${GO2_SPARK_FALLBACK_DEVICE:-192.168.0.24}"
port="${GO2_STATUS_PORT:-8769}"
stamp="$(date -u '+%Y-%m-%dT%H-%M-%SZ')"
out_dir="${GO2_STATUS_DIR:-/private/tmp/go2-spark-status-$stamp}"
tar_path="$out_dir/logs-listing.tar"

mkdir -p "$out_dir"

echo "device=$device"
echo "out_dir=$out_dir"
echo "utc=$stamp"
echo

echo "== app status =="
if ! wendy --device "$device" device ps --json > "$out_dir/device-ps.json"; then
  if [ -n "$fallback_device" ] && [ "$fallback_device" != "$device" ]; then
    echo "primary device lookup failed; retrying with fallback_device=$fallback_device" >&2
    device="$fallback_device"
    wendy --device "$device" device ps --json > "$out_dir/device-ps.json"
  else
    exit 1
  fi
fi
cat "$out_dir/device-ps.json"
echo

echo "== pulling /logs listing =="
nc -l "$port" > "$tar_path" &
listener_pid=$!
sleep 1
wendy run --device "$device" --prefix _tmp_spark_current_listing --dockerfile Dockerfile -y
wait "$listener_pid"

mkdir -p "$out_dir/extracted"
tar -xf "$tar_path" -C "$out_dir/extracted"

echo
echo "== training summary =="
python3 scripts/summarize_go2_listing.py "$out_dir/extracted/logs-listing.txt"
