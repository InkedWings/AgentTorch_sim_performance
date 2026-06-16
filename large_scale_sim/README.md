# Large-Scale LLM-Driven COVID Transmission Simulation — Implementation Notes

> This document explains the LLM-decision + agent-based transmission simulation we built on top of
> AgentTorch: how it is implemented, which files make up the complete example, and how to run it on
> the ALCF Polaris HPC system.
>
> **The code has not been moved** — it still lives inside the `agent_torch/` package (it depends
> heavily on `agent_torch.core`, so pulling it out would break imports). This document is just a
> "map + operations manual." All paths are relative to the repo root
> `/lus/eagle/projects/lc-mpi/ZhijingYe/AgentTorch`.

---

## 0. One-line summary

37,518 Astoria (Queens, NYC) resident agents evolve each day through 4 substeps
(transmission → disease progression → testing → quarantine). The **"do I isolate at home today?"
behavioral decision is made by an LLM** (Qwen3-32B served via vLLM), while the disease transmission,
testing, and quarantine mechanics are implemented as PyTorch tensors + graph message passing. The LLM
runs on one node and the simulation on another, communicating over an OpenAI-compatible HTTP API.

Most recent successful large-scale run: **37,518 agents × 21 days, completed in 362.7 seconds**
(`logs/covid_llm_runs/covid_llm_x3005c0s13b0n0_to_x3005c0s37b1n0_20260513_023529.log`).

---

## 1. Overall architecture (two-node topology)

```
┌────────────────────────────┐     HTTP POST /v1/chat/completions    ┌─────────────────────────────┐
│  Sim node (RUN_NODE)        │  ───────────────────────────────────▶ │  vLLM node (VLLM_NODE)       │
│                            │     a batch of group prompts (~tens-hundreds) │                         │
│  scripts/run_covid_llm.py  │                                        │  Qwen/Qwen3-32B             │
│  37518 agents / 1 GPU      │                                        │  tensor-parallel-size=4     │
│  PyTorch ABM (4 substeps)  │  ◀───────────────────────────────────  │  apptainer container :8000  │
│  EXECUTION_MODE=llm        │     {"probability":x,"reason":...}      │  OpenAI-compatible API      │
└────────────────────────────┘                                        └─────────────────────────────┘
        started via                                                          managed via
        run_covid_llm_remote.sh (ssh)                                        manage_vllm.sh
```

- **Why split across two nodes**: 32B inference saturates 4 GPUs (TP=4); the simulation tensors
  themselves are tiny (peak ~200 MB). Keeping them on separate nodes avoids resource contention, and
  the LLM service can be reused across multiple simulation runs.
- **Communication**: standard OpenAI Chat Completions protocol. The sim side uses LangChain's
  `ChatOpenAI` client with `base_url` pointing at the vLLM node. **`NO_PROXY` must include the vLLM
  node name**, otherwise the ALCF proxy intercepts intra-cluster requests and they time out.

---

## 2. The simulation loop: 4 substeps per day

Each simulation step = 1 day. The substep order is set by the `0/1/2/3` dict keys under `substeps:` in
`config.yaml` (Python 3.7+ preserves insertion order). Each substep follows AgentTorch's standard
three-phase pattern: **observe → act (policy) → progress (transition)**.

| # | Substep | What it does | Registered classes |
|---|---------|--------------|--------------------|
| **0** | **Transmission** | **LLM decides who isolates** → infection spreads over the mobility network | policy: `MakeIsolationDecision`; transition: `NewTransmission` + `UpdateIsolationMemory` |
| **1** | **Disease Progression** | SEIRM state advance: S→E→I→R/M | transition: `SEIRMProgression` |
| **2** | **Testing** | who gets tested, results after 3 days, true/false positives | obs: `GetTestingState`; policy: `AcceptTest`; transition: `UpdateTestStatus` |
| **3** | **Quarantine** | comply with quarantine after a positive, 12 days, possibly break | obs: `GetQuarantineState`; policy: `StartCompliance`/`BreakCompliance`; transition: `UpdateQuarantineStatus` |

SEIRM state encoding: `S=0, E=1, I=2, R=3, M(death)=4`.

### Substep 0: Transmission (the core; the only place the LLM is called)

**(a) LLM isolation decision** — `MakeIsolationDecision.forward()` (`new_transmission/action.py:316`)

1. Read each agent's **observable state**: age, consecutive isolation days, cumulative isolation days,
   whether currently quarantined, whether awaiting a test result, latest test result, testing/quarantine
   history, etc. (`action.py:226-272`). These continuous quantities are **bucketed into text**
   (e.g. isolation streak → "0 days" / "1 to 2 days" / "3 to 7 days" / "8+ days") to curb combinatorial
   explosion.
2. The 37,518 people are grouped by `grouping_logic` (default: **age band × state context × week**) into
   a set of **groups**, and **only one representative agent per group sends one prompt**
   (`template.py:379 get_grouped_prompts`). This is the key that drops LLM calls from ~787K/episode to a
   few hundred.
3. The prompt template lives in `covid/llm.py` (`CovidIsolationTemplate`): it gives the LLM age, location,
   outbreak context, and personal observable state, and asks for
   `{"probability": <0–1>, "reason": "..."}`.
4. `LangchainLLM.prompt()` (`core/llm/backend.py:182`) uses a thread pool (`max_concurrency`, default 8)
   to fire this batch of group prompts concurrently at vLLM.
5. Parse the returned `probability` (`behavior.py:43 _parse_behavior_value`, with multi-layer JSON/regex
   fallbacks) → broadcast the group probability to every agent in the group → `torch.bernoulli` samples
   the 0/1 `will_isolate`.
6. **Quarantined people are forced to isolate**: `will_isolate = max(will_isolate, is_quarantined)`
   (`action.py:358`).

> **heuristic mode** (`EXECUTION_MODE=heuristic`, no LLM): age-based Bernoulli isolation
> (`action.py:274`). `run_covid_llm.py` **force-overrides the mode to `llm`** at runtime
> (`run_covid_llm.py:30`).

**(b) Transmission** — `NewTransmission.forward()` (`new_transmission/transition.py:131`)

- Uses `torch_geometric` **MessagePassing** to aggregate infection pressure over the agent-agent contact
  network (`populations/astoria/mobility_networks/0.csv`).
- Per-edge transmission rate `_lam()` (`transition.py:52`) =
  `R · susceptibility(by age) · infectiousness · edge weight · γ-integral / mean interactions`.
- Infection probability `1 - exp(-Σ transmission pressure)`, sampled with **straight-through Bernoulli**
  (differentiable, for calibrating R2).
- **Effect of isolation**: `potentially_exposed_today *= (1 - will_isolate)` (`transition.py:231`) — an
  isolating agent **cannot be infected**.
  ⚠️ **Known implementation detail**: the term that would make isolators "no longer infect others" is
  **commented out** in `_lam()` (`transition.py:78-81`), so isolation currently only blocks *incoming*
  infection and does not reduce *outgoing* infectiousness. Documented faithfully for future fixing.

**(c) Isolation memory update** — `UpdateIsolationMemory.forward()` (`new_transmission/memory.py`)

- Writes today's isolation decision back to state: updates `last_isolation_decision`,
  `isolation_streak_days` (consecutive days, reset to 0 if not isolating), `num_isolation_days`
  (cumulative).
- This is the **feedback loop**: today's decision becomes the "personal history" in tomorrow's LLM
  prompt, letting decisions differentiate over time.

### Substep 1: Disease progression `SEIRMProgression` (`seirm_progression/transition.py`)
- Due E→I (after `EXPOSED_TO_INFECTED_TIME=5` days), due I→R/M (after `INFECTED_TO_RECOVERED_TIME=5` days).
- Deaths drawn from the day's exits at the learned mortality rate `M` (default 0.12).

### Substep 2: Testing `AcceptTest` + `UpdateTestStatus` (`testing/`)
- Testable population: E or I and not quarantined; accept a test with `test_compliance_prob=0.95`.
- Results arrive `test_result_delay_days=3` days later; true-positive rate `0.8`, false-positive rate `0.3`.
- On the result day, `positive_test_result=1` (held for one step only, consumed by the next substep);
  cannot re-test within `test_ineligible_days=2` days.

### Substep 3: Quarantine `StartCompliance`/`BreakCompliance` + `UpdateQuarantineStatus` (`quarantine/`)
- Positive result and not yet quarantined → start quarantine with `quarantine_start_prob=0.7`.
- Already quarantined → may break with `quarantine_break_prob=0.1`.
- Quarantine lasts `quarantine_days=12` days, then lifts; updates `quarantine_streak_days`,
  `num_quarantine_days`.

---

## 3. Complete file inventory

### A. Entry points / orchestration scripts (`scripts/`, repo root)
| File | Role |
|------|------|
| `scripts/run_covid_llm.py` | **Simulation main entry**. Parse CLI → rewrite config (force `EXECUTION_MODE=llm`, inject LLM metadata) → build `LangchainLLM` + `Archetype` + template → `MakeIsolationDecision.set_behavior(...)` wires in the LLM → `runner.init()` → `runner.step()` per episode → print summary metrics. |
| `scripts/run_covid_llm_remote.sh` | Runs the simulation on **RUN_NODE** via ssh: `conda activate AgentTorch`, set `NO_PROXY`, point base-url at VLLM_NODE, background nohup (or `COVID_LLM_FOREGROUND=1` for foreground). |
| `manage_vllm.sh` (repo root) | **vLLM service lifecycle management**: `start/stop/status/test/logs/endpoint/config`. Auto-discovers the node via `qstat -n $JOBID` → ssh in → apptainer launches `vllm serve Qwen/Qwen3-32B --tensor-parallel-size 4 --port 8000`. |

### B. COVID model (`agent_torch/models/covid/`)
| File | Role |
|------|------|
| `__init__.py` | `get_registry()` registers all substep classes and init functions; `get_runner(config, registry)` builds the Runner. |
| `llm.py` ⭐(new) | `CovidIsolationTemplate` — the LLM prompt template, asks for `{"probability","reason"}`; `grouping_logic = [age, agent_state_context, week]`. |
| `yamls/config.yaml` ⭐(changed) | The full 709-line config: metadata, agent/environment/network state tensor definitions, wiring of the 4 substeps' inputs/outputs. |
| `simulator.py` / `main.py` / `trainer.py` | registry assembly / minimal entry / calibration-training entry. |
| `substeps/new_transmission/action.py` ⭐(heavily changed) | `MakeIsolationDecision`: state bucketing, grouping, LLM call, parsing, trace writing (`_write_trace` emits `step/week/group_key/prompt/response/parsed_decision`). |
| `substeps/new_transmission/transition.py` ⭐(changed) | `NewTransmission`: graph message passing for transmission. |
| `substeps/new_transmission/memory.py` ⭐(new) | `UpdateIsolationMemory`: isolation-history feedback loop. |
| `substeps/seirm_progression/transition.py` | `SEIRMProgression`: SEIRM disease progression. |
| `substeps/testing/{action,observation,transition}.py` ⭐(changed) | Testing substep. |
| `substeps/quarantine/{action,observation,transition}.py` ⭐(changed) | Quarantine substep. |
| `substeps/utils.py` | Init functions (read network, γ-integrals, mean interactions, load population attributes, etc.). |
| `calibration/utils/{data,feature,llm,misc,neighborhood}.py` | Dependencies of action.py: real case data, `AgeGroup` text, epiweek conversion, neighborhood-name mapping. |

### C. Modified AgentTorch core LLM layer (`agent_torch/core/llm/`)
| File | Role |
|------|------|
| `backend.py` ⭐(changed) | `LangchainLLM`: `ChatOpenAI` pointed at vLLM; thread-pool concurrency; `MockLLM`/`DspyLLM` alternatives. |
| `archetype.py` ⭐(changed) | `Archetype` (facade) + `LLMArchetype`: broadcast binds the population, calls by group, optional memory. |
| `behavior.py` ⭐(changed) | `Behavior.sample()`: template grouping flow, `_parse_behavior_value` parsing, scatters group results back to each agent. |
| `template.py` ⭐(changed) | `Template`: `get_grouped_prompts` (the grouping core), `render`/`assemble_data`, P3O learnable slots. |
| `core/environment.py` ⭐(changed) | Environment assembly. |

### D. Data (`agent_torch/populations/astoria/`, ~30 MB)
`age/area/county/ethnicity/gender/region/soc_code.pickle` (population attributes), `disease_stages.csv`
(initial disease stage), `mobility_networks/0.csv` (contact network), `mapping.json` (attribute→text
mapping), real-case pkls (for calibration).

### E. Artifacts (`logs/`)
- `logs/vllm/` — vLLM service logs (throughput / KV-cache / tokens-per-second stats).
- `logs/covid_llm_runs/` — per-run stdout logs + pid for each simulation.
- `logs/covid_llm_trace.jsonl` ⭐ — **the final, sensible run's decision trace** (one line per group: prompt + response + probability).

⭐ = files we implemented/changed in this work (corresponding to the modified / untracked entries in `git status`).

---

## 4. On "decision sensibility": which run was the good one

We tallied the decision distribution across all 24 trace files under `logs/`. The progression is clear:

| Stage | Representative trace | Output format | Decision behavior |
|-------|----------------------|---------------|-------------------|
| Early (homogeneous) | `covid_llm_trace_21.jsonl`, etc. | Yes/No text | **Every group's decision identical** (all 0 or all 1), no differentiation |
| Mid (probability introduced) | `covid_llm_prob_full_trace.jsonl` | `{"probability"}` | 6 distinct levels appear |
| **Final (sensible) ✅** | **`covid_llm_trace.jsonl`** | `{"probability","reason"}` | **668 distinct groups, probabilities span [0.35, 0.92], differentiate by age/history** |

**Evidence the final version is sensible** (from `covid_llm_trace.jsonl`, 2,624 records):
- step 0: all 6 groups at 0.35 — sensible, because all agents start in identical state (no isolation/testing history).
- As the simulation advances, state differentiates: step 5 has 81 groups, step 14 has 166, step 20 has 217.
- **Older agents isolate with clearly higher probability over time** (step 20: 65+ mean 0.606 vs ~0.54-0.55
  for younger), consistent with the "more dangerous to older people" framing.

> **Conclusion: the code that produced this sensible result = the current working-tree version**
> (the ⭐ files in `git status`). Basis: the trace fields in `action.py:304-312` and the probability-JSON
> output format in `llm.py:23-24` exactly match `covid_llm_trace.jsonl` (the early version emitted
> `archetype_key/system_prompt/user_prompt` + Yes/No, now superseded).

---

## 5. How to run

Prerequisites: on ALCF Polaris, a conda env `AgentTorch` exists, and the vLLM container is at
`../Agentic/containers/vllm-openai-v0.19.1.sif`.

### ① Get a PBS allocation (2 nodes: one vLLM, one sim)
```bash
qstat -u $USER                 # check whether the job moved from Q (queued) to R (running)
qstat -n <JOBID>               # once R, read the two allocated node names (x300...)
# To request an interactive allocation when you have none (adjust -A/-q/-l as needed):
# qsub -I -A lc-mpi -q debug -l select=2:ncpus=64:ngpus=4 -l walltime=01:00:00 -l filesystems=home:eagle
```

### ② Start Qwen3-32B on the vLLM node (the script auto-discovers the node via qstat and ssh's in)
```bash
cd /lus/eagle/projects/lc-mpi/ZhijingYe/AgentTorch
VLLM_PBS_JOBID=<JOBID> ./manage_vllm.sh start
VLLM_PBS_JOBID=<JOBID> ./manage_vllm.sh status    # wait for "Port 8000 listening"
VLLM_PBS_JOBID=<JOBID> ./manage_vllm.sh logs -f   # watch loading (~100s load + ~30s torch.compile)
```

### ③ Verify the endpoint
```bash
VLLM_PBS_JOBID=<JOBID> ./manage_vllm.sh test       # should return "OK"
VLLM_PBS_JOBID=<JOBID> ./manage_vllm.sh endpoint   # prints http://<vllm_node>:8000/v1
```

### ④ Run the simulation on the other node (substitute the two node names from qstat -n)
```bash
./scripts/run_covid_llm_remote.sh <RUN_NODE> <VLLM_NODE> -- \
  --num-agents 37518 --steps 21 --episodes 1 \
  --llm-grouping-mode age_memory --llm-max-concurrency 8 \
  --llm-trace-path logs/covid_llm_trace_repro.jsonl
# To stream output in the foreground: prefix the command with COVID_LLM_FOREGROUND=1
```

### ⑤ Collect results
```bash
tail -f logs/covid_llm_runs/covid_llm_<RUN>_to_<VLLM>_*.log
# Watch for: episode_ok ... elapsed=... infected_sum=... isolation_rate=...
# Decision-level trace (per-group prompt + response + probability): logs/covid_llm_trace_repro.jsonl
```
When done, stop vLLM: `VLLM_PBS_JOBID=<JOBID> ./manage_vllm.sh stop`

> Quick smoke test (no LLM, results in seconds): `--num-agents 2000 --steps 3`, or stay in heuristic mode
> and run `python agent_torch/models/covid/main.py` directly.

---

## 6. Key parameters at a glance

**Simulation scale / disease progression (`config.yaml` simulation_metadata)**
| Parameter | Value | Meaning |
|-----------|-------|---------|
| `num_agents` | 37518 | Astoria population |
| `num_steps_per_episode` | 21 | days per episode |
| `num_episodes` | 5 (script defaults to 1) | episode count |
| `initial_infection_ratio` | 0.04 | initial infected fraction |
| `EXPOSED_TO_INFECTED_TIME` | 5 | E→I days |
| `INFECTED_TO_RECOVERED_TIME` | 5 | I→R/M days |
| `quarantine_days` | 12 | quarantine length |
| `test_result_delay_days` | 3 | test result delay |
| `NEIGHBORHOOD` / `START_WEEK` / `NUM_WEEKS` | Astoria / 202048 / 3 | location and time window |

**Behavioral probabilities (`config.yaml` environment)**
| Parameter | Value |
|-----------|-------|
| `quarantine_start_prob` | 0.7 |
| `quarantine_break_prob` | 0.1 |
| `test_compliance_prob` | 0.95 |
| `test_true_positive_prob` | 0.8 |
| `test_false_positive_prob` | 0.3 |
| `mortality_rate` | 0.2 |

**Learnable parameters**: transmission rate `R2` default 4.75 (shape `[NUM_WEEKS,1]`), mortality rate `M` default 0.12.

**LLM CLI knobs (`run_covid_llm.py`)**
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--base-url` | required | vLLM's `http://node:8000/v1` |
| `--model` | `agenttorch-qwen3-32b` | served model name |
| `--temperature` / `--max-tokens` | 0.0 / 64 | sampling temperature / max generation |
| `--llm-grouping-mode` | `age_memory` | `age_week`/`age_memory`/`state_memory` |
| `--llm-max-concurrency` | 8 | concurrent request threads |
| `--llm-history-k` | 0 | >0 includes conversation history (inflates the prompt) |
| `--llm-max-archetype-memory-groups` | 4096 | upper bound on group count |
| `--llm-trace-path` | `logs/covid_llm_trace.jsonl` | decision trace output |

**vLLM environment variables (`manage_vllm.sh`)**
| Variable | Default |
|----------|---------|
| `VLLM_MODEL` / `VLLM_SERVED_MODEL_NAME` | `Qwen/Qwen3-32B` / `agenttorch-qwen3-32b` |
| `VLLM_TENSOR_PARALLEL_SIZE` | 4 |
| `VLLM_PORT` | 8000 |
| `VLLM_MAX_MODEL_LEN` | 24576 |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.90 |
| `VLLM_PBS_JOBID` | used by `qstat -n` to discover the node |

---

## 7. Existing performance baseline (one 37518×21 run)

From `logs/covid_llm_runs/...20260513_023529.log` and the corresponding vLLM log:
- **End-to-end**: 362.7 s / episode (including init); sim-side peak memory only ~200 MB
  (⚠️ this is the *sim* node, **not** the vLLM node).
- **LLM request count**: ~hundreds of requests over 21 days (per group, not per agent), vs
  37518×21 ≈ 787.8K agent-steps — roughly **3 orders of magnitude of compression**.
- **vLLM throughput / KV-cache / tokens-per-second**: see `logs/vllm/*.log`.
- Example terminal state: cumulative infections ≈1425, deaths ≈32.6, isolation rate ≈0.50.

For a proper HPC scaling study, still to be added: weak/strong scaling over agent count, a serving
sweep over `--llm-max-concurrency` and `VLLM_TENSOR_PARALLEL_SIZE`, and a request-count vs fidelity
trade-off across the three grouping modes. (None of these have been done yet.)
