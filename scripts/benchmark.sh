#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

load_benchmark_config() {
    eval "$(PYTHONPATH="$REPO_ROOT" python3 - <<'PY'
import shlex

from src.benchmark.config import CLOUD_INFER_URL, EDGE_INFER_URL

values = {
    'EDGE_INFER_URL': EDGE_INFER_URL,
    'CLOUD_INFER_URL': CLOUD_INFER_URL,
}

for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"
}

resolve_iface() {
    if [[ -n "${BENCHMARK_IFACE:-}" ]]; then
        printf '%s\n' "$BENCHMARK_IFACE"
        return
    fi

    ip route | awk '/default/ {print $5; exit}'
}

health_url() {
    local infer_url="$1"
    printf '%s/health\n' "${infer_url%/infer}"
}

load_benchmark_config

cleanup() {
    echo ""
    echo "=== Cleanup: resetting network ==="
    sudo bash "$REPO_ROOT/scripts/set_netem.sh" reset || true
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "=== Syntax checks ==="
bash -n "$REPO_ROOT/scripts/set_netem.sh"
bash -n "$REPO_ROOT/scripts/run_grid.sh"

echo ""
echo "=== Dry run ==="
DRY_RUN_OUTPUT=$(bash "$REPO_ROOT/scripts/run_grid.sh" --dry-run)
printf '%s\n' "$DRY_RUN_OUTPUT"

EXPECTED_RUNS=$(printf '%s\n' "$DRY_RUN_OUTPUT" | sed -n 's/^Total measured runs: \([0-9]\+\)$/\1/p')

if [[ -z "$EXPECTED_RUNS" || "$EXPECTED_RUNS" -eq 0 ]]; then
    echo "ERROR: dry-run did not report a positive number of measured runs" >&2
    exit 1
fi

PLANNED_LINES=$(printf '%s\n' "$DRY_RUN_OUTPUT" | grep -c "strategy=.* latency=.* run=" || true)

if [[ "$PLANNED_LINES" != "$EXPECTED_RUNS" ]]; then
    echo "ERROR: dry-run planned $PLANNED_LINES runs but reported $EXPECTED_RUNS" >&2
    exit 1
fi

if [[ "$DRY_RUN_OUTPUT" != *"strategy=device latency=0ms"* ]]; then
    echo "ERROR: dry-run did not include device runs at latency=0ms" >&2
    exit 1
fi

if [[ "$DRY_RUN_OUTPUT" == *"strategy=device latency=100ms"* ]]; then
    echo "ERROR: dry-run included device runs at latency=100ms" >&2
    exit 1
fi

if [[ "$DRY_RUN_OUTPUT" == *"strategy=device latency=300ms"* ]]; then
    echo "ERROR: dry-run included device runs at latency=300ms" >&2
    exit 1
fi

echo ""
echo "=== Health checks ==="
EDGE_HEALTH_URL=$(health_url "$EDGE_INFER_URL")
CLOUD_HEALTH_URL=$(health_url "$CLOUD_INFER_URL")
curl -fsS "$EDGE_HEALTH_URL"
printf '\n'
curl -fsS "$CLOUD_HEALTH_URL"
printf '\n'

echo ""
echo "=== Verifying baseline qdisc ==="
sudo bash "$REPO_ROOT/scripts/set_netem.sh" set 0 0
IFACE=$(resolve_iface)

if [[ -z "$IFACE" ]]; then
    echo "ERROR: could not determine default network interface" >&2
    exit 1
fi

QDISC_0=$(tc qdisc show dev "$IFACE")
printf '%s\n' "$QDISC_0"

if [[ "$QDISC_0" == *"netem"* ]]; then
    echo "ERROR: baseline qdisc still contains netem" >&2
    exit 1
fi

echo ""
echo "=== Verifying 100ms qdisc ==="
sudo bash "$REPO_ROOT/scripts/set_netem.sh" set 100 0
QDISC_100=$(tc qdisc show dev "$IFACE")
printf '%s\n' "$QDISC_100"

if [[ "$QDISC_100" != *"netem"* ]]; then
    echo "ERROR: 100ms qdisc does not contain netem" >&2
    exit 1
fi

if [[ "$QDISC_100" != *"delay 100ms"* && "$QDISC_100" != *"delay 100.0ms"* ]]; then
    echo "ERROR: 100ms qdisc does not report delay 100ms" >&2
    exit 1
fi

echo ""
echo "=== Resetting network ==="
sudo bash "$REPO_ROOT/scripts/set_netem.sh" reset

echo ""
echo "=== Running official benchmark ==="
bash "$REPO_ROOT/scripts/run_grid.sh"

RUN_DIR=$(ls -td "$REPO_ROOT"/results/benchmark_run_* 2>/dev/null | head -1)

if [[ -z "$RUN_DIR" ]]; then
    echo "ERROR: could not determine benchmark result directory" >&2
    exit 1
fi

CSV_COUNT=$(find "$RUN_DIR" -maxdepth 1 -name "*.csv" | wc -l | tr -d '[:space:]')

if [[ "$CSV_COUNT" != "$EXPECTED_RUNS" ]]; then
    echo "ERROR: expected $EXPECTED_RUNS CSV files, found $CSV_COUNT" >&2
    exit 1
fi

for REQUIRED_FILE in "$RUN_DIR/manifest.txt" "$RUN_DIR/netem_log.txt" "$RUN_DIR/system_log.txt"; do
    if [[ ! -f "$REQUIRED_FILE" ]]; then
        echo "ERROR: missing required file: $REQUIRED_FILE" >&2
        exit 1
    fi
done

echo ""
echo "=== Aggregating results ==="
PYTHONPATH="$REPO_ROOT" python3 -m src.benchmark.aggregate --input-dir "$RUN_DIR"

echo "Benchmark complete."
echo "Result directory: $RUN_DIR"
echo "CSV files: $CSV_COUNT"
echo "Manifest: $RUN_DIR/manifest.txt"
echo "Netem log: $RUN_DIR/netem_log.txt"
echo "System log: $RUN_DIR/system_log.txt"
