import re

import torch

from agent_torch.core.helpers import get_by_path
from agent_torch.core.substep import SubstepTransition


class UpdateIsolationMemory(SubstepTransition):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)
        self.device = torch.device(self.config["simulation_metadata"]["device"])

    def forward(self, state, action):
        isolation_decision = action["citizens"]["isolation_decision"].to(self.device)
        isolation_decision = (isolation_decision.float() >= 0.5).float()

        current_streak = get_by_path(
            state, re.split("/", self.input_variables["isolation_streak_days"])
        ).to(self.device)
        current_total = get_by_path(
            state, re.split("/", self.input_variables["num_isolation_days"])
        ).to(self.device)

        new_last_decision = isolation_decision
        new_streak = torch.where(
            isolation_decision.bool(),
            current_streak + 1.0,
            torch.zeros_like(current_streak),
        )
        new_total = current_total + isolation_decision.to(current_total.dtype)

        return {
            self.output_variables[0]: new_last_decision,
            self.output_variables[1]: new_streak,
            self.output_variables[2]: new_total,
        }
