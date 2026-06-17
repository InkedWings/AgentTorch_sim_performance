#!/usr/bin/env python3
"""Sample node-level CPU utilization, frequency, and package power."""

import argparse
import csv
import datetime as dt
import glob
import os
import socket
import sys
import time
from pathlib import Path


def read_cpu_times():
    with open("/proc/stat", "r", encoding="utf-8") as handle:
        fields = handle.readline().split()

    values = [int(value) for value in fields[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return idle, total


def cpu_util(prev, cur):
    prev_idle, prev_total = prev
    cur_idle, cur_total = cur
    total_delta = cur_total - prev_total
    idle_delta = cur_idle - prev_idle
    if total_delta <= 0:
        return None
    return max(0.0, min(100.0, 100.0 * (1.0 - idle_delta / total_delta)))


def read_cpu_freq_mhz():
    freqs = []
    for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"):
        try:
            freqs.append(float(Path(path).read_text().strip()) / 1000.0)
        except (OSError, ValueError):
            pass

    if freqs:
        return sum(freqs) / len(freqs)

    cpuinfo_freqs = []
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("cpu mhz"):
                    cpuinfo_freqs.append(float(line.split(":", 1)[1].strip()))
    except (OSError, ValueError):
        pass

    if cpuinfo_freqs:
        return sum(cpuinfo_freqs) / len(cpuinfo_freqs)
    return None


def rapl_energy_paths():
    paths = []
    for energy_path in glob.glob("/sys/class/powercap/*/energy_uj"):
        path = Path(energy_path)
        name_path = path.parent / "name"
        try:
            name = name_path.read_text(encoding="utf-8").strip().lower()
        except OSError:
            name = ""

        # Prefer package-level domains to avoid double-counting nested DRAM/core domains.
        if "package" in name or path.parent.name.count(":") == 1:
            paths.append(path)
    return sorted(set(paths))


def read_rapl_energy(paths):
    if not paths:
        return None

    total = 0
    found = False
    for path in paths:
        try:
            total += int(path.read_text().strip())
            found = True
        except (OSError, ValueError):
            continue
    return total if found else None


def iso_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def fmt(value):
    return "" if value is None else f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--host", default=socket.gethostname())
    args = parser.parse_args()

    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "epoch",
            "iso_time",
            "host",
            "cpu_util_pct",
            "cpu_freq_mhz",
            "cpu_power_w",
        ]
    )
    sys.stdout.flush()

    rapl_paths = rapl_energy_paths()
    prev_cpu = read_cpu_times()
    prev_energy = read_rapl_energy(rapl_paths)
    prev_time = time.time()

    while True:
        time.sleep(args.interval)
        now = time.time()
        cur_cpu = read_cpu_times()
        cur_energy = read_rapl_energy(rapl_paths)

        util = cpu_util(prev_cpu, cur_cpu)
        freq = read_cpu_freq_mhz()
        power = None
        if prev_energy is not None and cur_energy is not None and now > prev_time:
            energy_delta = cur_energy - prev_energy
            # Handle counter wrap by skipping one sample instead of guessing max range.
            if energy_delta >= 0:
                power = energy_delta / 1_000_000.0 / (now - prev_time)

        writer.writerow([f"{now:.6f}", iso_now(), args.host, fmt(util), fmt(freq), fmt(power)])
        sys.stdout.flush()

        prev_cpu = cur_cpu
        prev_energy = cur_energy
        prev_time = now


if __name__ == "__main__":
    raise SystemExit(main())
