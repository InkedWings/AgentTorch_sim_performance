#!/usr/bin/env python3
"""Aggregate a single-instance sweep into tidy tables for analysis.

Walks an OUT_ROOT produced by sweep_single_instance.sh and, for every run dir
(one per sweep point), joins:
  * results.json         -> config + timing + summary + per-step times
  * cpu_*/gpu_*.csv       -> node CPU/GPU utilization, power, mem (time-aligned
                             to the run window via results.json markers)
  * vllm_metrics.csv      -> vLLM gauges/counters/histograms over the run window
  * vllm_metrics_before/after.prom -> counter deltas (tokens, requests) for the run

Produces, under OUT_ROOT/_analysis/:
  runs_summary.csv   -> one row per run: every config knob + headline metrics
  steps_long.csv     -> one row per (run, step): per-step + per-substep times
  README.txt         -> column dictionary

Headline metrics per run include:
  end-to-end elapsed, init time, per-step mean/median/p95,
  sim-GPU peak mem, LLM trace lines (request count proxy),
  vLLM tokens (prompt/generation) consumed during run (counter delta),
  vLLM requests succeeded during run, mean num_requests_running/waiting,
  mean/peak kv_cache_usage, mean TTFT / inter-token-latency (sum/count delta),
  vLLM-node + sim-node GPU util & power means, CPU util means.

Pure stdlib; no pandas dependency required.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
from pathlib import Path


# ----------------------------- small helpers -------------------------------
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pctl(values, q):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = q * (len(vals) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < len(vals):
        return vals[lo] * (1 - frac) + vals[lo + 1] * frac
    return vals[lo]


def read_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def window_filter(rows, t0, t1, ts_field="epoch"):
    out = []
    for r in rows:
        t = _f(r.get(ts_field))
        if t is None:
            continue
        if (t0 is None or t >= t0) and (t1 is None or t <= t1):
            out.append(r)
    return out


def mean_of(rows, field):
    vals = [_f(r.get(field)) for r in rows]
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def peak_of(rows, field):
    vals = [_f(r.get(field)) for r in rows]
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


# ----------------------- prometheus snapshot parsing -----------------------
_PROM = re.compile(r"^(?P<name>vllm:[a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-0-9.eE+]+)\s*$")


def parse_prom(path: Path):
    """Return dict: name -> summed value across label sets (counters/gauges)."""
    agg: dict[str, float] = {}
    if not path.exists():
        return agg
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _PROM.match(line)
        if not m:
            continue
        v = _f(m.group("value"))
        if v is None:
            continue
        agg[m.group("name")] = agg.get(m.group("name"), 0.0) + v
    return agg


def counter_delta(before, after, name):
    b = before.get(name)
    a = after.get(name)
    if a is None and b is None:
        return None
    return (a or 0.0) - (b or 0.0)


# --------------------------- vllm metrics (sampled) ------------------------
def vllm_metric_series(rows, metric_name):
    """Sum value across label sets per timestamp, return list of (epoch, value)."""
    by_ts: dict[str, float] = {}
    for r in rows:
        if r.get("metric") != metric_name:
            continue
        v = _f(r.get("value"))
        if v is None:
            continue
        by_ts[r.get("epoch")] = by_ts.get(r.get("epoch"), 0.0) + v
    return [(_f(k), v) for k, v in by_ts.items()]


def vllm_gauge_mean_peak(rows, metric_name):
    series = vllm_metric_series(rows, metric_name)
    vals = [v for _, v in series if v is not None]
    if not vals:
        return None, None
    return statistics.fmean(vals), max(vals)


# --------------------------------- main ------------------------------------
def process_run(run_dir: Path) -> dict | None:
    res_path = run_dir / "results.json"
    if not res_path.exists():
        return None
    res = json.loads(res_path.read_text(encoding="utf-8"))

    cfg = res.get("config", {})
    timing = res.get("timing", {})
    markers = res.get("markers", {})
    eps = res.get("episodes", [])

    # run window from markers (fall back to first/last episode)
    t0 = _f(markers.get("wall_start_epoch"))
    t1 = _f(markers.get("wall_end_epoch"))

    # per-step times across episodes
    step_times = [s for ep in eps for s in ep.get("step_times_s", [])]

    row: dict = {
        "run_label": res.get("run_label", run_dir.name),
        "run_dir": run_dir.name,
        # config knobs
        "num_agents": cfg.get("num_agents"),
        "steps": cfg.get("steps"),
        "episodes": cfg.get("episodes"),
        "grouping_mode": cfg.get("llm_grouping_mode"),
        "history_k": cfg.get("llm_history_k"),
        "max_groups": cfg.get("llm_max_archetype_memory_groups"),
        "max_concurrency": cfg.get("llm_max_concurrency"),
        "max_tokens": cfg.get("max_tokens"),
        "population_dir": cfg.get("population_dir"),
        "cuda_sync": cfg.get("cuda_sync"),
        # end-to-end timing
        "init_elapsed_s": timing.get("init_elapsed_s"),
        "total_elapsed_s": timing.get("total_elapsed_s"),
        "n_steps_recorded": len(step_times),
        "step_mean_s": statistics.fmean(step_times) if step_times else None,
        "step_median_s": statistics.median(step_times) if step_times else None,
        "step_p95_s": pctl(step_times, 0.95),
        "step_max_s": max(step_times) if step_times else None,
        "sim_gpu_peak_mem_mb": res.get("sim_gpu_peak_mem_mb"),
        "llm_trace_lines_total": res.get("llm_trace_lines_total"),
    }

    # episode summary (use last episode's summary as representative)
    if eps:
        summ = eps[-1].get("summary", {})
        row["infected_sum"] = summ.get("infected_sum")
        row["deaths_sum"] = summ.get("deaths_sum")
        row["isolation_rate"] = summ.get("last_isolation_decision_mean")

    # ---- vLLM counter deltas (before/after prom) ----
    before = parse_prom(run_dir / "vllm_metrics_before.prom")
    after = parse_prom(run_dir / "vllm_metrics_after.prom")
    row["vllm_prompt_tokens_delta"] = counter_delta(before, after, "vllm:prompt_tokens_total")
    row["vllm_generation_tokens_delta"] = counter_delta(before, after, "vllm:generation_tokens_total")
    row["vllm_requests_success_delta"] = counter_delta(before, after, "vllm:request_success_total")
    # mean TTFT / ITL over the run from histogram sum/count deltas
    ttft_sum = counter_delta(before, after, "vllm:time_to_first_token_seconds_sum")
    ttft_cnt = counter_delta(before, after, "vllm:time_to_first_token_seconds_count")
    row["vllm_ttft_mean_s"] = (ttft_sum / ttft_cnt) if ttft_sum is not None and ttft_cnt else None
    itl_sum = counter_delta(before, after, "vllm:inter_token_latency_seconds_sum")
    itl_cnt = counter_delta(before, after, "vllm:inter_token_latency_seconds_count")
    row["vllm_itl_mean_s"] = (itl_sum / itl_cnt) if itl_sum is not None and itl_cnt else None
    qt_sum = counter_delta(before, after, "vllm:request_queue_time_seconds_sum")
    qt_cnt = counter_delta(before, after, "vllm:request_queue_time_seconds_count")
    row["vllm_queue_time_mean_s"] = (qt_sum / qt_cnt) if qt_sum is not None and qt_cnt else None

    # ---- vLLM sampled gauges over run window ----
    vrows = window_filter(read_csv_rows(run_dir / "vllm_metrics.csv"), t0, t1)
    run_m, run_pk = vllm_gauge_mean_peak(vrows, "vllm:num_requests_running")
    wait_m, wait_pk = vllm_gauge_mean_peak(vrows, "vllm:num_requests_waiting")
    kv_name = "vllm:kv_cache_usage_perc"
    kv_m, kv_pk = vllm_gauge_mean_peak(vrows, kv_name)
    if kv_m is None:
        kv_m, kv_pk = vllm_gauge_mean_peak(vrows, "vllm:gpu_cache_usage_perc")
    row["vllm_running_mean"] = run_m
    row["vllm_running_peak"] = run_pk
    row["vllm_waiting_mean"] = wait_m
    row["vllm_waiting_peak"] = wait_pk
    row["vllm_kv_usage_mean"] = kv_m
    row["vllm_kv_usage_peak"] = kv_pk

    # ---- node CPU/GPU over run window ----
    # find sampler files by role
    for role in ("vllm", "sim"):
        gpu_files = list(run_dir.glob(f"gpu_{role}_*.csv"))
        cpu_files = list(run_dir.glob(f"cpu_{role}_*.csv"))
        grows = []
        for gf in gpu_files:
            grows += window_filter(read_csv_rows(gf), t0, t1)
        crows = []
        for cf in cpu_files:
            crows += window_filter(read_csv_rows(cf), t0, t1)
        row[f"{role}_gpu_util_mean"] = mean_of(grows, "gpu_util_pct")
        row[f"{role}_gpu_util_peak"] = peak_of(grows, "gpu_util_pct")
        row[f"{role}_gpu_power_mean_w"] = mean_of(grows, "power_w")
        row[f"{role}_gpu_mem_used_peak_mib"] = peak_of(grows, "mem_used_mib")
        row[f"{role}_cpu_util_mean"] = mean_of(crows, "cpu_util_pct")
        row[f"{role}_cpu_power_mean_w"] = mean_of(crows, "cpu_power_w")

    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_root", help="Sweep OUT_ROOT containing per-run dirs")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    analysis = out_root / "_analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(
        d for d in out_root.iterdir() if d.is_dir() and (d / "results.json").exists()
    )
    if not run_dirs:
        print(f"No run dirs with results.json under {out_root}")
        return 1

    rows = []
    steps_long = []
    for d in run_dirs:
        row = process_run(d)
        if row is None:
            continue
        rows.append(row)
        res = json.loads((d / "results.json").read_text(encoding="utf-8"))
        for ep in res.get("episodes", []):
            names = ep.get("substep_names", [])
            for si, st in enumerate(ep.get("step_times_s", [])):
                rec = {
                    "run_label": row["run_label"],
                    "num_agents": row["num_agents"],
                    "grouping_mode": row["grouping_mode"],
                    "history_k": row["history_k"],
                    "episode": ep.get("episode"),
                    "step": si,
                    "step_time_s": st,
                }
                subs = ep.get("substep_times_s", [])
                if si < len(subs):
                    for i, sv in enumerate(subs[si]):
                        nm = names[i] if i < len(names) else str(i)
                        rec[f"substep_{i}_{nm}_s"] = sv
                steps_long.append(rec)

    # write runs_summary.csv
    summary_path = analysis / "runs_summary.csv"
    cols = list({k for r in rows for k in r.keys()})
    # stable ordering: config first
    lead = [
        "run_label", "run_dir", "num_agents", "grouping_mode", "history_k",
        "steps", "episodes", "max_concurrency", "max_tokens", "cuda_sync",
        "total_elapsed_s", "init_elapsed_s",
        "step_mean_s", "step_median_s", "step_p95_s", "step_max_s",
        "sim_gpu_peak_mem_mb", "llm_trace_lines_total",
        "vllm_prompt_tokens_delta", "vllm_generation_tokens_delta",
        "vllm_requests_success_delta", "vllm_ttft_mean_s", "vllm_itl_mean_s",
        "vllm_queue_time_mean_s",
        "vllm_running_mean", "vllm_running_peak", "vllm_waiting_mean",
        "vllm_waiting_peak", "vllm_kv_usage_mean", "vllm_kv_usage_peak",
        "vllm_gpu_util_mean", "vllm_gpu_util_peak", "vllm_gpu_power_mean_w",
        "vllm_gpu_mem_used_peak_mib", "vllm_cpu_util_mean",
        "sim_gpu_util_mean", "sim_gpu_power_mean_w", "sim_gpu_mem_used_peak_mib",
        "sim_cpu_util_mean",
        "infected_sum", "deaths_sum", "isolation_rate",
    ]
    ordered = [c for c in lead if c in cols] + [c for c in sorted(cols) if c not in lead]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ordered)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # write steps_long.csv
    steps_cols = list({k for r in steps_long for k in r.keys()})
    steps_lead = ["run_label", "num_agents", "grouping_mode", "history_k", "episode", "step", "step_time_s"]
    steps_ordered = [c for c in steps_lead if c in steps_cols] + [
        c for c in sorted(steps_cols) if c not in steps_lead
    ]
    steps_path = analysis / "steps_long.csv"
    with steps_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=steps_ordered)
        w.writeheader()
        for r in steps_long:
            w.writerow(r)

    (analysis / "README.txt").write_text(
        "runs_summary.csv : one row per sweep run; config knobs + headline "
        "end-to-end / per-step / vLLM / node-resource metrics.\n"
        "steps_long.csv   : one row per (run, episode, step); per-step and "
        "per-substep wall times.\n"
        "Per-run raw data stays in each run dir (results.json, *.csv, *.prom, "
        "vllm_metrics_raw/).\n",
        encoding="utf-8",
    )

    print(f"[aggregate] {len(rows)} runs -> {summary_path}")
    print(f"[aggregate] {len(steps_long)} step rows -> {steps_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
