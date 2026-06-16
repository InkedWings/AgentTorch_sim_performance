import json
import os
import re

import torch

from agent_torch.core.decorators import with_behavior
from agent_torch.core.helpers import get_by_path
from agent_torch.core.substep import SubstepAction

from ...calibration.utils.data import get_data
from ...calibration.utils.feature import Feature
from ...calibration.utils.llm import AgeGroup, construct_user_prompt
from ...calibration.utils.misc import name_to_neighborhood, week_num_to_epiweek


@with_behavior
class MakeIsolationDecision(SubstepAction):
    """Policy that produces the per-citizen isolation decision.

    In LLM mode this stays inside the AgentTorch behavior interface:
    dynamic simulation state is passed as sample-time kwargs, the core
    Behavior groups archetypes, queries the LLM, parses Yes/No to 1/0,
    and broadcasts group decisions back to citizens.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.mode = self.config["simulation_metadata"]["EXECUTION_MODE"]
        self.num_agents = self.config["simulation_metadata"]["num_agents"]

        self.neighborhood = name_to_neighborhood(
            self.config["simulation_metadata"]["NEIGHBORHOOD"]
        )
        self.epiweek_start = week_num_to_epiweek(
            self.config["simulation_metadata"]["START_WEEK"]
        )
        self.include_week_count = self.config["simulation_metadata"].get(
            "INCLUDE_WEEK_COUNT", True
        )
        self.num_weeks = self.config["simulation_metadata"].get("NUM_WEEKS", 1)
        self.cases_data = get_data(
            self.neighborhood,
            self.epiweek_start,
            self.num_weeks,
            [Feature.CASES, Feature.CASES_4WK_AVG],
        )

        self.age_text_by_value = self._load_age_text_by_value()
        self.trace_path = self.config["simulation_metadata"].get("LLM_TRACE_PATH")
        self.llm_history_k = int(
            self.config["simulation_metadata"].get("LLM_HISTORY_K", 0)
        )
        self.llm_grouping_mode = self.config["simulation_metadata"].get(
            "LLM_GROUPING_MODE", "age_memory"
        )
        self.quarantine_days = self.config["simulation_metadata"].get(
            "quarantine_days", 12
        )
        self.test_result_delay_days = self.config["simulation_metadata"].get(
            "test_result_delay_days", 3
        )

        self.streak_text_by_bucket = {
            0: "0 days",
            1: "1 to 2 days",
            2: "3 to 7 days",
            3: "8 or more days",
        }
        self.total_days_text_by_bucket = {
            0: "0 days",
            1: "1 to 3 days",
            2: "4 to 10 days",
            3: "11 or more days",
        }
        self.test_count_text_by_bucket = {0: "0", 1: "1", 2: "2 or more"}

    def _load_age_text_by_value(self):
        mapping_path = os.path.join(
            self.config["simulation_metadata"]["population_dir"], "mapping.json"
        )
        label_to_text = {
            "U19": AgeGroup.UNDER_19.text,
            "20t29": AgeGroup.BETWEEN_20_29.text,
            "30t39": AgeGroup.BETWEEN_30_39.text,
            "40t49": AgeGroup.BETWEEN_40_49.text,
            "50t64": AgeGroup.BETWEEN_50_64.text,
            "65A": AgeGroup.ABOVE_65.text,
        }
        if os.path.exists(mapping_path):
            with open(mapping_path, "r", encoding="utf-8") as f:
                age_labels = json.load(f).get("age", [])
            return {
                idx: label_to_text.get(label, str(label))
                for idx, label in enumerate(age_labels)
            }
        return {age_group.value: age_group.text for age_group in AgeGroup}

    def _bucket_streak(self, values):
        values = values.squeeze().long()
        buckets = torch.zeros_like(values)
        buckets[(values >= 1) & (values <= 2)] = 1
        buckets[(values >= 3) & (values <= 7)] = 2
        buckets[values >= 8] = 3
        return buckets

    def _bucket_total_days(self, values):
        values = values.squeeze().long()
        buckets = torch.zeros_like(values)
        buckets[(values >= 1) & (values <= 3)] = 1
        buckets[(values >= 4) & (values <= 10)] = 2
        buckets[values >= 11] = 3
        return buckets

    def _bucket_test_count(self, values):
        values = values.squeeze().long()
        buckets = torch.zeros_like(values)
        buckets[values == 1] = 1
        buckets[values >= 2] = 2
        return buckets

    def _case_context(self, week_id):
        week_id = min(int(week_id), self.cases_data.shape[0] - 1)
        cases = int(self.cases_data[week_id, 0].item())
        cases_4_week_avg = int(self.cases_data[week_id, 1].item())
        context = construct_user_prompt(
            self.include_week_count,
            self.epiweek_start,
            week_id,
            cases,
            cases_4_week_avg,
        )
        return context.replace("\nDoes this person choose to isolate at home?", "")

    def _context_sentence(
        self,
        step,
        last_isolation,
        isolation_streak_bucket,
        total_isolation_bucket,
        is_quarantined,
        awaiting_test_result,
        positive_test_result,
        last_test_action,
        last_test_positive,
        test_count_bucket,
        positive_test_count_bucket,
        quarantine_streak_bucket,
    ):
        details = ["Known personal information available to the person:"]

        if int(is_quarantined):
            quarantine_text = (
                f"- quarantine status: currently in a {self.quarantine_days}-day "
                "quarantine period"
            )
            if int(quarantine_streak_bucket):
                quarantine_text += (
                    " for "
                    + self.streak_text_by_bucket[int(quarantine_streak_bucket)]
                )
            details.append(quarantine_text)
        elif int(positive_test_result):
            details.append("- quarantine status: not currently in quarantine")
            details.append("- current test status: positive COVID test result recorded")
        elif int(awaiting_test_result):
            details.append(
                "- quarantine status: not currently in quarantine"
            )
            details.append(
                "- current test status: waiting for a COVID test result; "
                f"results take {self.test_result_delay_days} days"
            )
        else:
            details.append("- quarantine status: not currently in quarantine")
            details.append("- current test status: no pending test result")

        if int(step) > 0 and (
            int(last_isolation)
            or int(isolation_streak_bucket)
            or int(total_isolation_bucket)
        ):
            total_text = self.total_days_text_by_bucket[int(total_isolation_bucket)]
            if int(last_isolation):
                streak_text = self.streak_text_by_bucket[int(isolation_streak_bucket)]
                details.append(
                    "- previous isolation behavior: isolated at home yesterday; "
                    f"current isolation streak is {streak_text}; total previous "
                    f"isolation days are {total_text}"
                )
            else:
                details.append(
                    "- previous isolation behavior: did not isolate at home "
                    f"yesterday; total previous isolation days are {total_text}"
                )
        else:
            details.append("- previous isolation behavior: no previous isolation recorded")

        if int(last_test_action):
            details.append("- testing history: took a COVID test yesterday")
        if int(last_test_positive):
            details.append("- testing history: most recent COVID test result was positive")
        if int(test_count_bucket) or int(positive_test_count_bucket):
            testing_history = "- testing totals: "
            totals = []
            if int(test_count_bucket):
                totals.append(
                    self.test_count_text_by_bucket[int(test_count_bucket)]
                    + " COVID tests taken"
                )
            if int(positive_test_count_bucket):
                totals.append(
                    self.test_count_text_by_bucket[int(positive_test_count_bucket)]
                    + " positive COVID test results"
                )
            details.append(testing_history + "; ".join(totals))
        elif not int(last_test_action):
            details.append(
                "- testing history: no previous COVID tests recorded"
            )

        return "\n".join(details)

    def _state_contexts(self, state):
        def read(name):
            return get_by_path(state, re.split("/", self.input_variables[name])).to(
                self.device
            )

        age = read("age").squeeze().long()
        last_isolation = read("last_isolation_decision").squeeze().long()
        isolation_streak = self._bucket_streak(read("isolation_streak_days"))
        total_isolation = self._bucket_total_days(read("num_isolation_days"))
        is_quarantined = read("is_quarantined").squeeze().long()
        awaiting_test_result = read("awaiting_test_result").squeeze().long()
        positive_test_result = read("positive_test_result").squeeze().long()
        last_test_action = read("last_test_action").squeeze().long()
        last_test_positive = read("last_test_positive").squeeze().long()
        test_count = self._bucket_test_count(read("num_tests_taken"))
        positive_test_count = self._bucket_test_count(read("num_positive_tests"))
        quarantine_streak = self._bucket_streak(read("quarantine_streak_days"))

        age_texts = [
            self.age_text_by_value.get(int(value), str(int(value)))
            for value in age.detach().cpu().tolist()
        ]
        if self.llm_grouping_mode == "age_week":
            context = "This person has no additional personal simulation history."
            return age_texts, [context] * len(age_texts)

        values = zip(
            last_isolation.detach().cpu().tolist(),
            isolation_streak.detach().cpu().tolist(),
            total_isolation.detach().cpu().tolist(),
            is_quarantined.detach().cpu().tolist(),
            awaiting_test_result.detach().cpu().tolist(),
            positive_test_result.detach().cpu().tolist(),
            last_test_action.detach().cpu().tolist(),
            last_test_positive.detach().cpu().tolist(),
            test_count.detach().cpu().tolist(),
            positive_test_count.detach().cpu().tolist(),
            quarantine_streak.detach().cpu().tolist(),
        )
        cache = {}
        contexts = []
        for row in values:
            if row not in cache:
                cache[row] = self._context_sentence(state["current_step"], *row)
            contexts.append(cache[row])
        return age_texts, contexts

    def _heuristic_isolation(self, state):
        age = get_by_path(state, re.split("/", self.input_variables["age"]))
        age = age.squeeze().long().to(self.device)
        if "initial_isolation_prob" not in self.fixed_args:
            return (torch.rand(self.num_agents, 1, device=self.device) >= 0.5).float()

        probs_by_age = self.fixed_args["initial_isolation_prob"].to(self.device).view(-1)
        age = torch.clamp(age, min=0, max=probs_by_age.shape[0] - 1)
        probs = probs_by_age[age].view(self.num_agents, 1)
        return torch.bernoulli(probs).float()

    def _write_trace(self, state, week_id):
        if not self.trace_path or self.behavior is None:
            return

        prompts = getattr(self.behavior, "last_prompt_list", [])
        group_keys = getattr(self.behavior, "last_group_keys", [])
        group_indices = getattr(self.behavior, "last_group_indices", [])
        raw_outputs = getattr(self.behavior, "last_raw_outputs", [])
        parsed_outputs = getattr(self.behavior, "last_group_outputs", [])

        trace_dir = os.path.dirname(self.trace_path)
        if trace_dir:
            os.makedirs(trace_dir, exist_ok=True)
        with open(self.trace_path, "a", encoding="utf-8") as f:
            for idx, prompt in enumerate(prompts):
                raw = raw_outputs[idx] if idx < len(raw_outputs) else None
                response = raw.get("text", raw) if isinstance(raw, dict) else raw
                parsed = parsed_outputs[idx] if idx < len(parsed_outputs) else None
                row = {
                    "step": int(state["current_step"]),
                    "week": int(week_id),
                    "group_key": str(group_keys[idx]) if idx < len(group_keys) else "",
                    "agent_count": (
                        len(group_indices[idx]) if idx < len(group_indices) else 0
                    ),
                    "prompt": prompt,
                    "response": "" if response is None else str(response),
                    "parsed_decision": None if parsed is None else float(parsed),
                }
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

    def forward(self, state, observation):
        if self.mode == "heuristic":
            will_isolate = self._heuristic_isolation(state)
        else:
            if self.behavior is None or not hasattr(self.behavior, "sample"):
                raise RuntimeError(
                    "COVID EXECUTION_MODE=llm requires a Behavior attached to "
                    "make_isolation_decision."
                )
            week_id = int(state["current_step"]) // 7
            age_texts, contexts = self._state_contexts(state)
            memory_dir = self.config["simulation_metadata"].get(
                "memory_dir",
                os.path.join(
                    self.config["simulation_metadata"]["population_dir"],
                    "simulation_memory_output",
                ),
            )
            kwargs = {
                "device": self.device,
                "current_memory_dir": os.path.join(
                    memory_dir, f"step_{state['current_step']}"
                ),
                "llm_history_k": self.llm_history_k,
                "age": age_texts,
                "location": self.neighborhood.text,
                "agent_state_context": contexts,
                "case_context": self._case_context(week_id),
                "week": week_id,
            }
            isolation_prob = self.behavior.sample(kwargs).to(self.device).float()
            isolation_prob = torch.clamp(isolation_prob, min=0.0, max=1.0)
            will_isolate = torch.bernoulli(isolation_prob)
            self._write_trace(state, week_id)

        if will_isolate.ndim == 1:
            will_isolate = will_isolate.unsqueeze(1)

        if "is_quarantined" in self.input_variables:
            is_quarantined = get_by_path(
                state, re.split("/", self.input_variables["is_quarantined"])
            ).to(self.device)
            will_isolate = torch.maximum(will_isolate, is_quarantined.float())

        output_key = self.output_variables[0] if self.output_variables else "isolation_decision"
        return {output_key: will_isolate}
