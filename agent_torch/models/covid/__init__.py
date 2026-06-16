from agent_torch.core import Runner, Registry


def get_registry():
    reg = Registry()

    from .substeps.new_transmission.action import MakeIsolationDecision
    from .substeps.new_transmission.memory import UpdateIsolationMemory
    from .substeps.new_transmission.transition import NewTransmission
    from .substeps.seirm_progression.transition import SEIRMProgression
    from .substeps.testing.action import AcceptTest
    from .substeps.testing.observation import GetFromState as GetTestingState
    from .substeps.testing.transition import UpdateTestStatus
    from .substeps.quarantine.action import BreakCompliance, StartCompliance
    from .substeps.quarantine.observation import GetFromState as GetQuarantineState
    from .substeps.quarantine.transition import UpdateQuarantineStatus

    reg.register(MakeIsolationDecision, "make_isolation_decision", key="policy")
    reg.register(NewTransmission, "new_transmission", key="transition")
    reg.register(UpdateIsolationMemory, "update_isolation_memory", key="transition")
    reg.register(SEIRMProgression, "seirm_progression", key="transition")
    reg.register(GetTestingState, "get_testing_state", key="observation")
    reg.register(AcceptTest, "accept_test", key="policy")
    reg.register(UpdateTestStatus, "update_test_status", key="transition")
    reg.register(GetQuarantineState, "get_quarantine_state", key="observation")
    reg.register(StartCompliance, "start_compliance", key="policy")
    reg.register(BreakCompliance, "break_compliance", key="policy")
    reg.register(UpdateQuarantineStatus, "update_quarantine_status", key="transition")

    from .substeps.utils import (
        network_from_file,
        read_from_file,
        get_lam_gamma_integrals,
        get_mean_agent_interactions,
        get_infected_time,
        get_next_stage_time,
        load_population_attribute,
        initialize_id,
    )

    reg.register(network_from_file, "network_from_file", key="network")
    reg.register(read_from_file, "read_from_file", key="initialization")
    reg.register(
        get_lam_gamma_integrals, "get_lam_gamma_integrals", key="initialization"
    )
    reg.register(
        get_mean_agent_interactions, "get_mean_agent_interactions", key="initialization"
    )
    reg.register(get_infected_time, "get_infected_time", key="initialization")
    reg.register(get_next_stage_time, "get_next_stage_time", key="initialization")
    reg.register(
        load_population_attribute, "load_population_attribute", key="initialization"
    )
    reg.register(initialize_id, "initialize_id", key="initialization")

    return reg


registry = get_registry()


def get_runner(config, registry):
    CovidRunner = Runner(config, registry)

    return CovidRunner
