import torch
import torch.nn as nn
import numpy as np
import re
from agent_torch.core.substep import SubstepAction
from agent_torch.core.helpers import (
    discrete_sample,
    get_by_path,
    logical_and,
    logical_not,
    logical_or,
)


class StartCompliance(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.output_variables = output_variables
        self.num_agents = self.config["simulation_metadata"]["num_agents"]
        self.device = self.config["simulation_metadata"]["device"]

        self.EXPOSED_VAR = self.config["simulation_metadata"]["EXPOSED_VAR"]
        self.INFECTED_VAR = self.config["simulation_metadata"]["INFECTED_VAR"]

    def forward(self, state, observation):
        quarantine_start_prob = observation["quarantine_start_prob"]
        is_quarantined = observation["is_quarantined"]
        positive_test_result = observation["positive_test_result"]

        quarantine_start_decision = discrete_sample(
            quarantine_start_prob, size=self.num_agents, device=self.device
        ).unsqueeze(1).bool()
        quarantine_start_decision = (
            quarantine_start_decision
            & torch.logical_not(is_quarantined.bool())
            & positive_test_result.bool()
        )

        return {self.output_variables[0]: quarantine_start_decision}


class BreakCompliance(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.input_variables = input_variables
        self.output_variables = output_variables
        self.num_agents = self.config["simulation_metadata"]["num_agents"]
        self.device = torch.device(self.config["simulation_metadata"]["device"])

    def forward(self, state, observation):
        quarantine_break_prob = observation["quarantine_break_prob"]
        is_quarantined = observation["is_quarantined"]

        quarantine_break_decision = discrete_sample(
            quarantine_break_prob, size=self.num_agents, device=self.device
        ).unsqueeze(1).bool()
        quarantine_break_decision = is_quarantined.bool() & quarantine_break_decision

        return {self.output_variables[0]: quarantine_break_decision}
