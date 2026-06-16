import torch
import re
from agent_torch.core.substep import SubstepAction
from agent_torch.core.helpers import discrete_sample, get_by_path


class AcceptTest(SubstepAction):
    def __init__(self, config, input_variables, output_variables, arguments):
        super().__init__(config, input_variables, output_variables, arguments)

        self.input_variables = input_variables
        self.output_variables = output_variables
        self.arguments = arguments

        self.num_agents = self.config["simulation_metadata"]["num_agents"]
        self.SUSCEPTIBLE_VAR = self.config["simulation_metadata"]["SUSCEPTIBLE_VAR"]
        self.RECOVERED_VAR = self.config["simulation_metadata"]["RECOVERED_VAR"]
        self.device = torch.device(self.config["simulation_metadata"]["device"])

    def forward(self, state, observation):
        agent_is_quarantined = get_by_path(
            state, re.split("/", self.input_variables["is_quarantined"])
        )
        agent_disease_stage = get_by_path(
            state, re.split("/", self.input_variables["disease_stage"])
        )
        test_compliance_prob = get_by_path(
            state, re.split("/", self.input_variables["test_compliance_prob"])
        )

        exposed_infected = (
            (agent_disease_stage > self.SUSCEPTIBLE_VAR)
            & (agent_disease_stage < self.RECOVERED_VAR)
        )
        agent_is_eligible = exposed_infected & torch.logical_not(
            agent_is_quarantined.bool()
        )

        agent_test_compliance = discrete_sample(
            sample_prob=test_compliance_prob,
            size=(self.num_agents,),
            device=self.device,
        ).unsqueeze(1).bool()
        agent_test_action = agent_is_eligible.bool() & agent_test_compliance

        return {self.output_variables[0]: agent_test_action}
