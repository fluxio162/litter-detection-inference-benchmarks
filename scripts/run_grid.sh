#!/bin/bash

set -euo pipefail

LATENCIES=(0 100 300)
STRATEGIES=(device edge cloud hi-edge hi-cloud)
N_RUNS=5
DRY_RUN=0

usage() {
    echo "Usage: bash scripts/run_grid.sh [--dry-run]" >&2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 1
            ;;
    esac
    shift
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

load_benchmark_config() {
    eval "$(PYTHONPATH="$REPO_ROOT" python3 - <<'PY'
import shlex
from pathlib import Path

from src.benchmark.config import CLOUD_INFER_URL, EDGE_INFER_URL, HI_CONFIDENCE_THRESHOLD, MODEL_PATH, TEST_IMAGES_DIR, TEST_LABELS_DIR

image_dir = Path(TEST_IMAGES_DIR)
image_count = 0
if image_dir.exists():
    image_count = len([p for p in image_dir.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}])

values = {
    'DEVICE_MODEL_PATH': MODEL_PATH,
    'TEST_IMAGES_DIR': TEST_IMAGES_DIR,
    'TEST_LABELS_DIR': TEST_LABELS_DIR,
    'EDGE_INFER_URL': EDGE_INFER_URL,
    'CLOUD_INFER_URL': CLOUD_INFER_URL,
    'HI_CONFIDENCE_THRESHOLD': HI_CONFIDENCE_THRESHOLD,
    'TEST_IMAGE_COUNT': image_count,
}

for key, value in values.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

    EDGE_MODEL_PATH="$REPO_ROOT/models/yolo26s.pt"
    CLOUD_MODEL_PATH="$REPO_ROOT/models/yolo26m.pt"
}

resolve_iface() {
    if [[ -n "${BENCHMARK_IFACE:-}" ]]; then
        printf '%s\n' "$BENCHMARK_IFACE"
        return
    fi

    ip route | awk '/default/ {print $5; exit}'
}

load_benchmark_config

cleanup() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        return
    fi

    echo ""
    echo "=== Cleanup: resetting network ==="
    sudo bash "$REPO_ROOT/scripts/set_netem.sh" reset || true
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

RUN_ID="benchmark_run_$(date +%s)"
RESULTS_DIR="$REPO_ROOT/results/${RUN_ID}"

if [[ "$DRY_RUN" -eq 0 ]]; then
    mkdir -p "$RESULTS_DIR"

    {
        echo "run_id=$RUN_ID"
        echo "date=$(date -Is)"
        echo "repo_root=$REPO_ROOT"
        echo "hostname=$(hostname)"
        echo "kernel=$(uname -a)"
        echo "python=$(python3 --version 2>&1)"
        echo "ultralytics=$(python3 -c 'import ultralytics; print(ultralytics.__version__)' 2>/dev/null || echo unknown)"
        echo "torch=$(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)"
        echo "git_commit=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "benchmark_iface=${BENCHMARK_IFACE:-auto}"
        echo "latencies=${LATENCIES[*]}"
        echo "strategies=${STRATEGIES[*]}"
        echo "n_runs=$N_RUNS"
        echo "device_model_path=$DEVICE_MODEL_PATH"
        echo "edge_model_path=$EDGE_MODEL_PATH"
        echo "cloud_model_path=$CLOUD_MODEL_PATH"
        echo "edge_infer_url=$EDGE_INFER_URL"
        echo "cloud_infer_url=$CLOUD_INFER_URL"
        echo "hi_confidence_threshold=$HI_CONFIDENCE_THRESHOLD"
        echo "test_images_dir=$TEST_IMAGES_DIR"
        echo "test_labels_dir=$TEST_LABELS_DIR"
        echo "test_image_count=$TEST_IMAGE_COUNT"
        echo ""
        echo "ip route:"
        ip route
    } > "$RESULTS_DIR/manifest.txt"

    python3 -m pip freeze > "$RESULTS_DIR/pip_freeze.txt" 2>/dev/null || true
fi

log_system_state() {
    local label="$1"

    {
        echo ""
        echo "=== $(date -Is) ${label} ==="
        echo "hostname=$(hostname)"
        echo "loadavg=$(cat /proc/loadavg)"
        echo "temp=$(vcgencmd measure_temp 2>/dev/null || echo unavailable)"
        echo "throttled=$(vcgencmd get_throttled 2>/dev/null || echo unavailable)"
        free -h || true
    } >> "$RESULTS_DIR/system_log.txt"
}

log_network_state() {
    local latency="$1"
    local iface

    iface=$(resolve_iface)

    {
        echo ""
        echo "=== $(date -Is) latency=${latency}ms iface=${iface:-unknown} ==="
        ip route
        if [[ -n "$iface" ]]; then
            tc qdisc show dev "$iface" || true
            tc -s qdisc show dev "$iface" || true
        else
            echo "ERROR: could not determine default network interface"
        fi
    } | tee -a "$RESULTS_DIR/netem_log.txt"
}

set_network_for_latency() {
    local latency="$1"

    echo ""
    echo "=== Setting network: latency=${latency}ms ==="

    if [[ "$DRY_RUN" -eq 1 ]]; then
        if [[ "$latency" -eq 0 ]]; then
            echo "[dry-run] would reset network to native/default state"
        else
            echo "[dry-run] would set netem delay to ${latency}ms"
        fi
        return
    fi

    if [[ "$latency" -eq 0 ]]; then
        sudo bash "$REPO_ROOT/scripts/set_netem.sh" reset
    else
        sudo bash "$REPO_ROOT/scripts/set_netem.sh" set "$latency" 0
    fi

    log_network_state "$latency"
    sleep 2
}

total_runs=0
for LAT in "${LATENCIES[@]}"; do
    for STRATEGY in "${STRATEGIES[@]}"; do
        if [[ "$STRATEGY" == "device" && "$LAT" -gt 0 ]]; then
            continue
        fi
        total_runs=$(( total_runs + N_RUNS ))
    done
done

echo "Run ID: $RUN_ID"
echo "Results directory: $RESULTS_DIR"
echo "Total measured runs: $total_runs"

current=0

for LAT in "${LATENCIES[@]}"; do
    set_network_for_latency "$LAT"

    for STRATEGY in "${STRATEGIES[@]}"; do
        if [[ "$STRATEGY" == "device" && "$LAT" -gt 0 ]]; then
            echo "  [skip] device is network-agnostic - already covered at latency=0ms"
            continue
        fi

        for RUN in $(seq 1 "$N_RUNS"); do
            current=$(( current + 1 ))
            OUTFILE="$RESULTS_DIR/${STRATEGY}_lat${LAT}ms_run${RUN}.csv"

            if [[ "$DRY_RUN" -eq 1 ]]; then
                echo "[${current}/${total_runs}] strategy=${STRATEGY} latency=${LAT}ms run=${RUN} output=${OUTFILE}"
                continue
            fi

            echo "[${current}/${total_runs}] strategy=${STRATEGY} latency=${LAT}ms run=${RUN}"
            log_system_state "before strategy=${STRATEGY} latency=${LAT}ms run=${RUN}"

            PYTHONPATH="$REPO_ROOT" python3 -m src.benchmark.runner \
                --strategy "$STRATEGY" \
                --output "$OUTFILE" \
                --run-index "$RUN" \
                --network-latency "$LAT"

            log_system_state "after strategy=${STRATEGY} latency=${LAT}ms run=${RUN}"
        done
    done
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo ""
    echo "Dry run complete. Planned results directory: $RESULTS_DIR"
    exit 0
fi

echo "Grid complete. Results in $RESULTS_DIR"
