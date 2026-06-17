#!/usr/bin/env python3
"""Poll a vLLM /metrics (Prometheus) endpoint into a long-format CSV time series.

vLLM exposes a rich Prometheus endpoint. Confirmed metric families on the
deployed version include (vllm: prefix):
  gauges    : num_requests_running, num_requests_waiting, kv_cache_usage_perc,
              gpu_cache_usage_perc (older), engine_sleep_state
  counters  : prompt_tokens_total, generation_tokens_total,
              iteration_tokens_total, request_success_total,
              num_preemptions_total, prefix_cache_{hits,queries}_total
  histograms: time_to_first_token_seconds, inter_token_latency_seconds,
              request_queue_time_seconds, request_prefill_time_seconds,
              request_decode_time_seconds, request_inference_time_seconds,
              e2e/request latency, time_per_output_token_seconds, ...

Rather than hard-code names (which drift across vLLM versions), this sampler
captures EVERY ``vllm:`` series each poll, including histogram buckets, sums,
and counts. Histograms are emitted as their _sum / _count / _bucket samples so
that average latencies (sum/count) and percentiles (buckets) are reconstructable
offline. This is the "capture everything" approach the user asked for.

Output: long-format CSV to stdout with columns:
  epoch, iso_time, host, metric, labels, value
where ``metric`` is the full Prometheus series name (e.g.
``vllm:time_to_first_token_seconds_bucket``) and ``labels`` is the raw label
set string (e.g. ``le="0.1",model_name="..."``) or empty.

Use --raw-dir to ALSO dump the full raw /metrics text each poll (timestamped),
so nothing is ever lost even if parsing misses something.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import re
import socket
import sys
import time
import urllib.request

# Matches:  name{labels} value   OR   name value
_LINE = re.compile(r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>.+?)\s*$")


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")


def fetch(url: str, timeout: float) -> str | None:
    try:
        # Bypass any proxy; intra-cluster traffic must go direct.
        proxy_handler = urllib.request.ProxyHandler({})
        opener = urllib.request.build_opener(proxy_handler)
        with opener.open(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - sampler must never die on one bad poll
        print(f"# metrics fetch failed: {exc}", file=sys.stderr, flush=True)
        return None


def parse_metrics(text: str, prefix: str):
    """Yield (name, labels, value) for matching series, skipping comments."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        name = m.group("name")
        if prefix and not name.startswith(prefix):
            continue
        labels = m.group("labels") or ""
        value = m.group("value")
        # Prometheus values are floats; keep as-is text but validate numeric.
        try:
            float(value)
        except ValueError:
            continue
        yield name, labels, value


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", required=True, help="e.g. http://127.0.0.1:8000/metrics")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--host", default=socket.gethostname())
    ap.add_argument(
        "--prefix",
        default="vllm:",
        help="Only emit series whose name starts with this (\"\" = all).",
    )
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument(
        "--raw-dir",
        default=None,
        help="If set, also write full raw /metrics text per poll here.",
    )
    args = ap.parse_args()

    if args.raw_dir:
        os.makedirs(args.raw_dir, exist_ok=True)

    writer = csv.writer(sys.stdout)
    writer.writerow(["epoch", "iso_time", "host", "metric", "labels", "value"])
    sys.stdout.flush()

    while True:
        now = time.time()
        stamp = iso_now()
        text = fetch(args.url, args.timeout)
        if text is not None:
            if args.raw_dir:
                raw_path = os.path.join(args.raw_dir, f"metrics_{now:.3f}.prom")
                try:
                    with open(raw_path, "w", encoding="utf-8") as rf:
                        rf.write(text)
                except OSError as exc:
                    print(f"# raw dump failed: {exc}", file=sys.stderr, flush=True)
            for name, labels, value in parse_metrics(text, args.prefix):
                writer.writerow([f"{now:.6f}", stamp, args.host, name, labels, value])
            sys.stdout.flush()
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
