#!/bin/bash

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 set <latency_ms> <bandwidth_mbps|0> OR $0 reset" >&2
    exit 1
fi

IFACE=${BENCHMARK_IFACE:-$(ip route | awk '/default/ {print $5; exit}')}

if [ -z "$IFACE" ]; then
    echo "ERROR: could not determine default network interface" >&2
    exit 1
fi

if [ "$1" = "reset" ]; then
    tc qdisc del dev "$IFACE" root 2>/dev/null || true
    echo "netem reset on $IFACE"

elif [ "$1" = "set" ]; then
    if [ $# -lt 3 ]; then
        echo "Usage: $0 set <latency_ms> <bandwidth_mbps|0>" >&2
        exit 1
    fi

    LATENCY=$2
    BW=$3

    if ! [[ "$LATENCY" =~ ^[0-9]+$ ]]; then
        echo "ERROR: latency must be a non-negative integer, got: $LATENCY" >&2
        exit 1
    fi

    if ! [[ "$BW" =~ ^[0-9]+$ ]]; then
        echo "ERROR: bandwidth must be a non-negative integer, got: $BW" >&2
        exit 1
    fi

    if [ "$LATENCY" = "0" ] && [ "$BW" = "0" ]; then
        tc qdisc del dev "$IFACE" root 2>/dev/null || true
        echo "netem reset: baseline/no configured delay on $IFACE"
        exit 0
    fi

    tc qdisc del dev "$IFACE" root 2>/dev/null || true

    if [ "$BW" = "0" ]; then
        tc qdisc add dev "$IFACE" root netem delay "${LATENCY}ms"
        echo "netem set: latency=${LATENCY}ms bandwidth=unlimited on $IFACE"
    else
        tc qdisc add dev "$IFACE" root handle 1: netem delay "${LATENCY}ms"
        tc qdisc add dev "$IFACE" parent 1: handle 2: tbf rate "${BW}mbit" burst 32kbit latency 400ms
        echo "netem set: latency=${LATENCY}ms bandwidth=${BW}Mbps on $IFACE"
    fi

else
    echo "Usage: $0 set <latency_ms> <bandwidth_mbps|0> OR $0 reset" >&2
    exit 1
fi
