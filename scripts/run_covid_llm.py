#!/usr/bin/env python
import argparse
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import torch
import yaml

from agent_torch.core.helpers import read_config
from agent_torch.core.llm.archetype import Archetype
from agent_torch.core.llm.backend import LangchainLLM
from agent_torch.models import covid
from agent_torch.models.covid.llm import CovidIsolationTemplate
from agent_torch.models.covid.substeps.new_transmission.action import (
    MakeIsolationDecision,
)


def load_config(config_path: Path, population_dir: Path, args: argparse.Namespace):
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    metadata = config["simulation_metadata"]
    metadata["population_dir"] = str(population_dir)
    metadata["num_agents"] = args.num_agents
    metadata["num_episodes"] = args.episodes
    metadata["num_steps_per_episode"] = args.steps
    metadata["EXECUTION_MODE"] = "llm"
    metadata["device"] = args.device
    metadata["memory_dir"] = str(args.memory_dir)
    metadata["LLM_HISTORY_K"] = args.llm_history_k
    metadata["LLM_MAX_ARCHETYPE_MEMORY_GROUPS"] = args.llm_max_archetype_memory_groups
    metadata["LLM_GROUPING_MODE"] = args.llm_grouping_mode
    if args.llm_trace_path:
        metadata["LLM_TRACE_PATH"] = str(args.llm_trace_path)

    tmp_config = args.output_config
    tmp_config.parent.mkdir(parents=True, exist_ok=True)
    with tmp_config.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return read_config(str(tmp_config))


def build_llm(args: argparse.Namespace):
    parsed = urlparse(args.base_url)
    if parsed.hostname:
        for name in ("NO_PROXY", "no_proxy"):
            current = os.environ.get(name, "")
            entries = [item for item in current.split(",") if item]
            if parsed.hostname not in entries:
                entries.append(parsed.hostname)
            os.environ[name] = ",".join(entries)

    return LangchainLLM(
        openai_api_key=args.api_key,
        agent_profile="",
        model=args.model,
        temperature=args.temperature,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_concurrency=args.llm_max_concurrency,
    )


def summarize_state(state):
    daily_infected = state["environment"]["daily_infected"].detach().cpu()
    daily_deaths = state["environment"]["daily_deaths"].detach().cpu()
    stages = (
        state["agents"]["citizens"]["disease_stage"]
        .detach()
        .flatten()
        .to(torch.int64)
        .cpu()
    )
    vals, counts = torch.unique(stages, return_counts=True)
    summary = {
        "infected_sum": float(daily_infected.sum().item()),
        "deaths_sum": float(daily_deaths.sum().item()),
        "stage_counts": {int(v): int(c) for v, c in zip(vals, counts)},
    }
    citizens = state["agents"]["citizens"]
    if "last_isolation_decision" in citizens:
        summary["isolation_rate"] = float(
            citizens["last_isolation_decision"].detach().float().mean().item()
        )
    if "isolation_streak_days" in citizens:
        summary["mean_isolation_streak"] = float(
            citizens["isolation_streak_days"].detach().float().mean().item()
        )
    if "num_isolation_days" in citizens:
        summary["total_isolation_days"] = float(
            citizens["num_isolation_days"].detach().float().sum().item()
        )
    if "awaiting_test_result" in citizens:
        summary["awaiting_test_rate"] = float(
            citizens["awaiting_test_result"].detach().float().mean().item()
        )
    if "positive_test_result" in citizens:
        summary["positive_test_results_today"] = float(
            citizens["positive_test_result"].detach().float().sum().item()
        )
    if "num_tests_taken" in citizens:
        summary["total_tests_taken"] = float(
            citizens["num_tests_taken"].detach().float().sum().item()
        )
    if "num_positive_tests" in citizens:
        summary["total_positive_tests"] = float(
            citizens["num_positive_tests"].detach().float().sum().item()
        )
    if "is_quarantined" in citizens:
        summary["quarantine_rate"] = float(
            citizens["is_quarantined"].detach().float().mean().item()
        )
    if "num_quarantine_days" in citizens:
        summary["total_quarantine_days"] = float(
            citizens["num_quarantine_days"].detach().float().sum().item()
        )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run AgentTorch COVID with LLM behavior.")
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible API base URL, e.g. http://node:8000/v1",
    )
    parser.add_argument("--model", default="agenttorch-qwen3-32b")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--llm-history-k", type=int, default=0)
    parser.add_argument("--llm-max-archetype-memory-groups", type=int, default=4096)
    parser.add_argument("--llm-max-concurrency", type=int, default=8)
    parser.add_argument(
        "--llm-grouping-mode",
        default="age_memory",
        choices=["age_week", "age_memory", "state_memory"],
    )
    parser.add_argument(
        "--llm-trace-path", type=Path, default=Path("logs/covid_llm_trace.jsonl")
    )
    parser.add_argument("--append-trace", action="store_true")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=21)
    parser.add_argument("--num-agents", type=int, default=37518)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--memory-dir", type=Path, default=Path("logs/covid_llm_memory"))
    parser.add_argument("--output-config", type=Path, default=Path("/tmp/agenttorch_covid_llm.yaml"))
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    population_dir = repo_root / "agent_torch/populations/astoria"
    config_path = repo_root / "agent_torch/models/covid/yamls/config.yaml"
    if not args.memory_dir.is_absolute():
        args.memory_dir = repo_root / args.memory_dir
    if not args.output_config.is_absolute():
        args.output_config = repo_root / args.output_config
    if args.llm_trace_path and not args.llm_trace_path.is_absolute():
        args.llm_trace_path = repo_root / args.llm_trace_path
    if args.llm_trace_path:
        args.llm_trace_path.parent.mkdir(parents=True, exist_ok=True)
        if not args.append_trace:
            args.llm_trace_path.write_text("", encoding="utf-8")

    config = load_config(config_path, population_dir, args)
    llm = build_llm(args)
    template = CovidIsolationTemplate()
    template.memory_size = args.llm_max_archetype_memory_groups
    isolation_archetype = Archetype(
        prompt=template,
        llm=llm,
        n_arch=1,
    )
    isolation_archetype.broadcast(str(population_dir))
    MakeIsolationDecision.set_behavior(isolation_archetype._behavior)

    print("AgentTorch COVID LLM run")
    print(f"base_url={args.base_url}")
    print(f"model={args.model}")
    print(f"agents={args.num_agents} episodes={args.episodes} steps={args.steps}")
    print(f"llm_grouping_mode={args.llm_grouping_mode}")
    print(f"llm_history_k={args.llm_history_k}")
    print(f"llm_max_archetype_memory_groups={args.llm_max_archetype_memory_groups}")
    if args.llm_trace_path:
        print(f"llm_trace_path={args.llm_trace_path}")
    print(
        f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"devices={torch.cuda.device_count()}"
    )

    runner = covid.get_runner(config, covid.registry)
    total_start = time.perf_counter()
    runner.init()
    print(f"init_ok device={runner.config['simulation_metadata']['device']}")

    for episode in range(args.episodes):
        episode_start = time.perf_counter()
        runner.step(args.steps)
        summary = summarize_state(runner.state)
        print(
            f"episode_ok {episode + 1}/{args.episodes} "
            f"elapsed={time.perf_counter() - episode_start:.3f} "
            f"infected_sum={summary['infected_sum']:.6f} "
            f"deaths_sum={summary['deaths_sum']:.6f} "
            f"stage_counts={summary['stage_counts']} "
            f"isolation_rate={summary.get('isolation_rate', 0.0):.6f} "
            f"mean_isolation_streak={summary.get('mean_isolation_streak', 0.0):.6f} "
            f"total_isolation_days={summary.get('total_isolation_days', 0.0):.0f} "
            f"awaiting_test_rate={summary.get('awaiting_test_rate', 0.0):.6f} "
            f"total_tests_taken={summary.get('total_tests_taken', 0.0):.0f} "
            f"total_positive_tests={summary.get('total_positive_tests', 0.0):.0f} "
            f"quarantine_rate={summary.get('quarantine_rate', 0.0):.6f} "
            f"total_quarantine_days={summary.get('total_quarantine_days', 0.0):.0f}"
        )
        if episode + 1 < args.episodes:
            runner.reset()

    if torch.cuda.is_available():
        print(f"max_cuda_mem_MB={torch.cuda.max_memory_allocated() / 1024**2:.2f}")
    print(f"full_ok total_elapsed={time.perf_counter() - total_start:.3f}")


if __name__ == "__main__":
    main()
