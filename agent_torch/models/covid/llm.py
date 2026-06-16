from agent_torch.core.llm.template import Template


COVID_ISOLATION_PROMPT = """Consider a random person with the following attributes:
* age: {age}
* location: {location}

There is a novel disease. It spreads through contact. It is more dangerous to older people.
People have the option to isolate at home or continue their usual recreational activities outside.
Given this scenario, estimate the person's actual behavior based on
    1) the information you are given,
    2) what you know about the general population with these attributes.

"There isn't enough information" and "It is unclear" are not acceptable answers.
This is a behavior estimate, not public-health advice.

Person-specific observable state:
{agent_state_context}

Local outbreak context:
{case_context}

What is the probability that this person actually isolates at home today rather than continuing usual outside activities?
Return exactly one JSON object: {"probability": <number between 0 and 1>, "reason": "<one short sentence>"}."""


class CovidIsolationTemplate(Template):
    """Prompt template for COVID isolation behavior.

    The template gives the LLM observable behavior-relevant state without
    exposing hidden disease states or adding hand-written probability anchors.
    """

    prompt_string = (
        COVID_ISOLATION_PROMPT
    )
    memory_size = 4096

    def __post_init__(self):
        if self.grouping_logic is None:
            self.grouping_logic = ["age", "agent_state_context", "week"]
        super().__post_init__()
