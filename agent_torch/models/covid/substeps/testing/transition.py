"""
ToDo: Add is_quarantine_eligible logic - for agents that test positive in result
"""

import torch
import re

from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_by_path, discrete_sample


class UpdateTestStatus(SubstepTransition):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.arguments = arguments

        self.device = torch.device(self.config["simulation_metadata"]["device"])
        self.SUSCEPTIBLE_VAR = self.config["simulation_metadata"]["SUSCEPTIBLE_VAR"]
        self.RECOVERED_VAR = self.config["simulation_metadata"]["RECOVERED_VAR"]
        self.EXPOSED_VAR = self.config["simulation_metadata"]["EXPOSED_VAR"]

        self.num_agents = self.config["simulation_metadata"]["num_agents"]
        self.test_ineligible_days = self.config["simulation_metadata"][
            "test_ineligible_days"
        ]
        self.test_result_delay_days = self.config["simulation_metadata"][
            "test_result_delay_days"
        ]

        self.AWAITING_RESULT_VAR = 1
        self.GOT_RESULT_VAR = -1
        self.INFINITY_TIME = self.config["simulation_metadata"]["INFINITY_TIME"]

    def get_test_result(
        self,
        t,
        agents_awaiting_results,
        agent_result_date,
        current_stages,
        test_re_eligble_date,
        true_positive_prob,
        false_positive_prob,
    ):
        """Agents receive test result"""
        agents_result_expected_today = agent_result_date == t

        # 1: reset agents_awaiting_test_result
        agents_awaiting_results = agents_awaiting_results.bool() & torch.logical_not(
            agents_result_expected_today
        )

        # 2: get true_positive and false_positive results - check candidates + sample based on TPR and FPR
        exposed_infected_agents = (
            (current_stages > self.SUSCEPTIBLE_VAR)
            & (current_stages < self.RECOVERED_VAR)
        )

        true_positive_result_candidates = (
            exposed_infected_agents.bool() & agents_result_expected_today
        )
        false_positive_result_candidates = (
            torch.logical_not(exposed_infected_agents.bool())
            & agents_result_expected_today
        )

        true_positive_mask = discrete_sample(
            sample_prob=true_positive_prob, size=(self.num_agents,), device=self.device
        ).unsqueeze(1).bool()
        true_positive_results = true_positive_result_candidates & true_positive_mask

        false_positive_mask = discrete_sample(
            sample_prob=false_positive_prob, size=(self.num_agents,), device=self.device
        ).unsqueeze(1).bool()
        false_positive_results = false_positive_result_candidates & false_positive_mask

        positive_results = true_positive_results | false_positive_results

        # 3: agents are in-eligible to test again for the next few days
        test_re_eligble_date[agents_result_expected_today.bool()] = (
            t + self.test_ineligible_days
        )  # not a differentiable op
        agent_result_date[agents_result_expected_today.bool()] = self.INFINITY_TIME

        return (
            positive_results,
            agents_awaiting_results,
            test_re_eligble_date,
            agent_result_date,
        )

    def get_tested(
        self, t, agents_awaiting_results, agents_result_date, test_enrolled_agents
    ):
        """Eligible Agent get themselves tested and receive test result date"""
        agents_awaiting_results = (
            agents_awaiting_results.bool() | test_enrolled_agents.bool()
        )
        agents_result_date[test_enrolled_agents.bool()] = (
            t + self.test_result_delay_days
        )  # not a differentiable op

        return agents_awaiting_results, agents_result_date

    def forward(self, state, action=None):
        t = state["current_step"]
        input_variables = self.input_variables

        current_stages = get_by_path(
            state, re.split("/", input_variables["disease_stage"])
        )
        agents_result_date = get_by_path(
            state, re.split("/", input_variables["test_result_date"])
        )
        agents_awaiting_results = get_by_path(
            state, re.split("/", input_variables["awaiting_test_result"])
        )
        #         is_quarantine_eligible = get_by_path(state, re.split('/', input_variables['is_quarantine_eligible']))
        test_re_eligble_date = get_by_path(
            state, re.split("/", input_variables["test_re_eligble_date"])
        )
        true_positive_prob = get_by_path(
            state, re.split("/", self.input_variables["test_true_positive_prob"])
        )
        false_positive_prob = get_by_path(
            state, re.split("/", self.input_variables["test_false_positive_prob"])
        )
        last_test_action = get_by_path(
            state, re.split("/", self.input_variables["last_test_action"])
        )
        num_tests_taken = get_by_path(
            state, re.split("/", self.input_variables["num_tests_taken"])
        )
        last_test_positive = get_by_path(
            state, re.split("/", self.input_variables["last_test_positive"])
        )
        num_positive_tests = get_by_path(
            state, re.split("/", self.input_variables["num_positive_tests"])
        )

        # step 1: agents receive test result and may test positive
        (
            positive_results,
            agents_awaiting_results,
            test_re_eligble_date,
            agent_result_date,
        ) = self.get_test_result(
            t,
            agents_awaiting_results,
            agents_result_date,
            current_stages,
            test_re_eligble_date,
            true_positive_prob,
            false_positive_prob,
        )

        # step 2: agents take test and join result queue
        test_willing_agents = action["citizens"]["test_acceptance_action"]

        test_ineligible_cooldown = t < test_re_eligble_date
        test_ineligible = test_ineligible_cooldown | agents_awaiting_results.bool()
        test_enrolled_agents = test_willing_agents.bool() & torch.logical_not(
            test_ineligible
        )

        agents_awaiting_results, agents_result_date = self.get_tested(
            t, agents_awaiting_results, agents_result_date, test_enrolled_agents
        )

        last_test_action = test_enrolled_agents.float()
        last_test_positive = positive_results.float()
        num_tests_taken = num_tests_taken + test_enrolled_agents.float()
        num_positive_tests = num_positive_tests + positive_results.float()

        return {
            self.output_variables[0]: agents_awaiting_results,
            self.output_variables[1]: agents_result_date,
            self.output_variables[2]: test_re_eligble_date,
            self.output_variables[3]: positive_results.float(),
            self.output_variables[4]: last_test_action,
            self.output_variables[5]: last_test_positive,
            self.output_variables[6]: num_tests_taken,
            self.output_variables[7]: num_positive_tests,
        }
