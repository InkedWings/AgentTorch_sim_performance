#!/usr/bin/env python3
"""Sample per-GPU utilization, memory, power, temperature, and clocks via nvidia-smi.

Captures a wide field set so downstream analysis can study GPU utilization,
memory footprint, power draw, thermals, and clock behavior over a run.
One CSV row per GPU per sample.
"""

import argparse
import csv
import datetime as dt
import socket
import subprocess
import sys
import time


QUERY_FIELDS = [
    "index",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "power.draw",
    "temperature.gpu",
    "clocks.sm",
    "clocks.mem",
    "pstate",
]
CSV_FIELDS = [
    "gpu_index",
    "gpu_util_pct",
    "mem_util_pct",
    "mem_used_mib",
    "mem_total_mib",
    "power_w",
    "temp_c",
    "clock_sm_mhz",
    "clock_mem_mhz",
    "pstate",
]


def iso_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--host", default=socket.gethostname())
    args = parser.parse_args()

    writer = csv.writer(sys.stdout)
    writer.writerow(["epoch", "iso_time", "host", *CSV_FIELDS])
    sys.stdout.flush()

    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]

    while True:
        now = time.time()
        stamp = iso_now()
        try:
            output = subprocess.check_output(
                command, universal_newlines=True, stderr=subprocess.STDOUT
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"nvidia-smi sample failed: {exc}", file=sys.stderr, flush=True)
            time.sleep(args.interval)
            continue

        for line in output.splitlines():
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != len(QUERY_FIELDS):
                continue
            writer.writerow([f"{now:.6f}", stamp, args.host, *fields])
        sys.stdout.flush()
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
