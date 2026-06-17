#!/usr/bin/env python3
"""Generate a clean sub-sampled AgentTorch population directory.

The COVID model loads agent attributes from fixed-size files and asserts that
tensor shapes match ``num_agents`` (see
``agent_torch/models/covid/substeps/utils.py:read_from_file`` — a hard
``assert values.shape == tuple(shape)``).  Therefore the population sweep
cannot just pass ``--num-agents < 37518``; it needs a self-consistent
population directory whose every file has exactly N rows.

This tool builds such a directory from a source population (default: astoria)
by:
  * taking the first N agents (deterministic, contiguous slice) for every
    per-agent attribute pickle and the disease_stages.csv,
  * building a DENSITY-PRESERVING mobility network for the N-agent subset:
    keep the induced subgraph (source edges with both endpoints < N), then
    top up with random in-subset edges so the edges-per-agent ratio matches
    the source population. This avoids the sharp density collapse of plain
    induced-subgraph sampling (a prefix of N agents keeps only edges internal
    to that prefix, which is ~N/N_src of the per-agent degree), which would
    otherwise suppress transmission and therefore agent-state differentiation.
  * copying mapping.json and any auxiliary calibration pickles verbatim.

Contiguous-prefix sampling keeps node indices identical to the source, so
disease-stage / attribute alignment is preserved exactly; only the network is
re-densified to hold edge density constant across sweep sizes ("vary scale,
hold density"). Use --no-densify for the plain induced subgraph instead.

Usage:
  python make_subpopulation.py --source <dir> --out <dir> --size N
  python make_subpopulation.py --source astoria_abspath --out-root <dir> --sizes 1000 2000 5000 ...
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Per-agent attribute pickles that must be sliced to N rows.
ATTR_PICKLES = [
    "age.pickle",
    "area.pickle",
    "county.pickle",
    "ethnicity.pickle",
    "gender.pickle",
    "region.pickle",
    "soc_code.pickle",
]

# Files copied verbatim (not per-agent-sized).
VERBATIM = [
    "mapping.json",
    "kings_county_processed_UI_claims.pkl",
    "processed_kings_county_monthly_cases.pkl",
]


def _slice_series(src_path: Path, dst_path: Path, n: int) -> int:
    s = pd.read_pickle(src_path)
    if len(s) < n:
        raise ValueError(
            f"{src_path.name} has {len(s)} rows < requested {n}; cannot upsample."
        )
    out = s.iloc[:n].reset_index(drop=True)
    out.to_pickle(dst_path)
    return len(out)


def _slice_disease_stages(src_path: Path, dst_path: Path, n: int) -> int:
    df = pd.read_csv(src_path, header=None)
    if len(df) < n:
        raise ValueError(
            f"disease_stages.csv has {len(df)} rows < requested {n}."
        )
    df.iloc[:n].to_csv(dst_path, header=False, index=False)
    return n


def _build_network(
    src_path: Path, dst_path: Path, n: int, densify: bool, seed: int
) -> dict:
    """Build the N-agent network.

    densify=True  -> induced subgraph + random in-subset top-up to match the
                     source edges-per-agent ratio (density-preserving).
    densify=False -> plain induced subgraph (edges with both endpoints < n).
    """
    net = pd.read_csv(src_path, header=None)
    src_edges = len(net)
    src_nodes = int(max(net[0].max(), net[1].max())) + 1
    edges_per_agent = src_edges / max(src_nodes, 1)

    induced = net[(net[0] < n) & (net[1] < n)].copy()
    kept = len(induced)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not densify:
        induced.to_csv(dst_path, header=False, index=False)
        return {
            "edges": kept, "edges_src": src_edges, "added": 0,
            "edges_per_agent": round(kept / max(n, 1), 4),
        }

    target = int(round(edges_per_agent * n))
    need = max(0, target - kept)

    rng = np.random.default_rng(seed)
    added_rows = []
    if need > 0:
        # Sample random directed edges within [0, n), skip self-loops and
        # collisions with existing edges (best-effort; tiny dup rate tolerated).
        existing = set(map(tuple, induced.values.tolist()))
        # over-sample then trim to account for rejections
        batch = int(need * 1.3) + 16
        while len(added_rows) < need:
            src = rng.integers(0, n, size=batch)
            dst = rng.integers(0, n, size=batch)
            for a, b in zip(src.tolist(), dst.tolist()):
                if a == b:
                    continue
                e = (a, b)
                if e in existing:
                    continue
                existing.add(e)
                added_rows.append(e)
                if len(added_rows) >= need:
                    break

    out = pd.concat(
        [induced, pd.DataFrame(added_rows, columns=[0, 1])], ignore_index=True
    ) if added_rows else induced
    out.to_csv(dst_path, header=False, index=False)
    return {
        "edges": len(out), "edges_src": src_edges, "added": len(added_rows),
        "induced_kept": kept, "target": target,
        "edges_per_agent": round(len(out) / max(n, 1), 4),
        "src_edges_per_agent": round(edges_per_agent, 4),
    }


def build_one(source: Path, out: Path, n: int, densify: bool = True, seed: int = 0) -> dict:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    info: dict = {
        "size": n, "source": str(source), "out": str(out),
        "densify": densify, "seed": seed,
    }

    for name in ATTR_PICKLES:
        src = source / name
        if src.exists():
            info[name] = _slice_series(src, out / name, n)

    ds_src = source / "disease_stages.csv"
    if ds_src.exists():
        info["disease_stages_rows"] = _slice_disease_stages(
            ds_src, out / "disease_stages.csv", n
        )
        # report initial infection count in the subsample
        ds = pd.read_csv(out / "disease_stages.csv", header=None)
        info["initial_nonzero_stage"] = int((ds[0] != 0).sum())

    # mobility networks (may be several: 0.csv, 1.csv, ...)
    net_dir = source / "mobility_networks"
    if net_dir.is_dir():
        for net_file in sorted(net_dir.glob("*.csv")):
            net_info = _build_network(
                net_file, out / "mobility_networks" / net_file.name, n,
                densify=densify, seed=seed,
            )
            for k, v in net_info.items():
                info[f"net_{net_file.stem}_{k}"] = v

    for name in VERBATIM:
        src = source / name
        if src.exists():
            shutil.copy2(src, out / name)

    # all_county_mix dir if present (copied verbatim; not per-agent indexed)
    mix = source / "all_county_mix"
    if mix.is_dir():
        shutil.copytree(mix, out / "all_county_mix", dirs_exist_ok=True)

    with (out / "subpopulation_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--source",
        required=True,
        help="Source population directory (must have N>=requested rows).",
    )
    ap.add_argument(
        "--out-root",
        help="Directory under which size-named subdirs are created "
        "(e.g. <root>/astoria_n2000).",
    )
    ap.add_argument("--out", help="Explicit single output dir (use with --size).")
    ap.add_argument("--size", type=int, help="Single size (use with --out).")
    ap.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        help="Multiple sizes; outputs go to <out-root>/<prefix>_n<size>.",
    )
    ap.add_argument(
        "--prefix",
        default="astoria",
        help="Name prefix for generated dirs under --out-root.",
    )
    ap.add_argument(
        "--no-densify",
        action="store_true",
        help="Use the plain induced subgraph (do NOT top up edges to match "
        "source density). Default is density-preserving.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=20260617,
        help="RNG seed for density top-up edge sampling (reproducible).",
    )
    args = ap.parse_args()

    source = Path(args.source).resolve()
    if not source.is_dir():
        print(f"Source not found: {source}", file=sys.stderr)
        return 2

    jobs: list[tuple[Path, int]] = []
    if args.sizes:
        if not args.out_root:
            print("--sizes requires --out-root", file=sys.stderr)
            return 2
        root = Path(args.out_root).resolve()
        for n in args.sizes:
            jobs.append((root / f"{args.prefix}_n{n}", n))
    elif args.out and args.size:
        jobs.append((Path(args.out).resolve(), args.size))
    else:
        print("Provide either --sizes+--out-root or --out+--size", file=sys.stderr)
        return 2

    for out, n in jobs:
        info = build_one(
            source, out, n, densify=not args.no_densify, seed=args.seed
        )
        print(
            f"[ok] N={n:>7}  net0_edges={info.get('net_0_edges', 'NA')}"
            f"  (induced={info.get('net_0_induced_kept', 'NA')}"
            f"+added={info.get('net_0_added', 'NA')},"
            f" deg={info.get('net_0_edges_per_agent', 'NA')})"
            f"  init_infected={info.get('initial_nonzero_stage', 'NA')}"
            f"  -> {out}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
