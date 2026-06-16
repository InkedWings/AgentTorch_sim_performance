import re
import torch

from agent_torch.core.substep import SubstepTransition
from agent_torch.core.helpers import get_by_path


class UpdateQuarantineStatus(SubstepTransition):
    """Logic: exposed or infected agents can start quarantine"""

    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.device = self.config["simulation_metadata"]["device"]
        self.num_agents = self.config["simulation_metadata"]["num_agents"]
        self.quarantine_days = self.config["simulation_metadata"]["quarantine_days"]
        self.num_steps = self.config["simulation_metadata"]["num_steps_per_episode"]

        self.SUSCEPTIBLE_VAR = self.config["simulation_metadata"]["SUSCEPTIBLE_VAR"]
        self.EXPOSED_VAR = self.config["simulation_metadata"]["EXPOSED_VAR"]
        self.INFECTED_VAR = self.config["simulation_metadata"]["INFECTED_VAR"]
        self.RECOVERED_VAR = self.config["simulation_metadata"]["RECOVERED_VAR"]

        self.INFINITY_TIME = self.config["simulation_metadata"]["INFINITY_TIME"]

        self.END_QUARANTINE_VAR = -1
        self.START_QUARANTINE_VAR = 1
        self.BREAK_QUARANTINE_VAR = -1

    def update_quarantine_status(
        self,
        t,
        is_quarantined,
        quarantine_start_date,
        agent_quarantine_start_action,
        agent_quarantine_break_action,
    ):
        currently_quarantined = is_quarantined.bool()
        start_action = agent_quarantine_start_action.bool()
        break_action = agent_quarantine_break_action.bool()

        quarantine_ends = currently_quarantined & (
            t >= quarantine_start_date + self.quarantine_days
        )
        after_end = currently_quarantined & torch.logical_not(quarantine_ends)

        starts_today = start_action & torch.logical_not(after_end)
        after_start = after_end | starts_today

        breaks_today = break_action & after_start
        new_is_quarantined = after_start & torch.logical_not(breaks_today)

        reset_start_date = quarantine_ends | breaks_today
        new_quarantine_start_date = torch.where(
            reset_start_date,
            torch.full_like(quarantine_start_date, self.INFINITY_TIME),
            quarantine_start_date,
        )
        new_quarantine_start_date = torch.where(
            starts_today,
            torch.full_like(quarantine_start_date, int(t)),
            new_quarantine_start_date,
        )

        return new_is_quarantined, new_quarantine_start_date

    def forward(self, state, action):
        input_variables = self.input_variables
        t = state["current_step"]

        is_quarantined = get_by_path(
            state, re.split("/", input_variables["is_quarantined"])
        )
        quarantine_start_date = get_by_path(
            state, re.split("/", input_variables["quarantine_start_date"])
        )
        positive_test_result = get_by_path(
            state, re.split("/", input_variables["positive_test_result"])
        )
        quarantine_streak_days = get_by_path(
            state, re.split("/", input_variables["quarantine_streak_days"])
        )
        num_quarantine_days = get_by_path(
            state, re.split("/", input_variables["num_quarantine_days"])
        )

        agent_quarantine_start_action = action["citizens"]["start_compliance_action"]
        agent_quarantine_break_action = action["citizens"]["break_compliance_action"]

        new_is_quarantined, new_quarantine_start_date = self.update_quarantine_status(
            t,
            is_quarantined,
            quarantine_start_date,
            agent_quarantine_start_action,
            agent_quarantine_break_action,
        )

        new_is_quarantined_float = new_is_quarantined.float()
        new_quarantine_streak_days = torch.where(
            new_is_quarantined.bool(),
            quarantine_streak_days + 1.0,
            torch.zeros_like(quarantine_streak_days),
        )
        new_num_quarantine_days = num_quarantine_days + new_is_quarantined_float
        consumed_positive_test_result = torch.zeros_like(positive_test_result)

        return {
            self.output_variables[0]: new_is_quarantined,
            self.output_variables[1]: new_quarantine_start_date,
            self.output_variables[2]: consumed_positive_test_result,
            self.output_variables[3]: new_quarantine_streak_days,
            self.output_variables[4]: new_num_quarantine_days,
        }
