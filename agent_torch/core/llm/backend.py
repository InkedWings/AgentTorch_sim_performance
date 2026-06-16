"""
LLM Backend Abstraction for AgentTorch
======================================

Provides abstract base class and implementations for LLM integration.

Supported backends:
- MockLLM: For testing without API calls (see mock_llm.py)
- DspyLLM: DSPy-based LLM integration (requires dspy package)

Usage:
    from agent_torch.core.llm.mock_llm import MockLLM
    llm = MockLLM(low=0.1, high=0.9)
    
    # Or with DSPy:
    from agent_torch.core.llm.backend import DspyLLM
    llm = DspyLLM(openai_api_key="...", qa=MyQA, cot=MyCOT)
"""
import os
import sys
import io
import concurrent.futures
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    """
    Abstract base class for LLM backends.
    
    All LLM backends must implement the `prompt()` method which takes
    a list of prompts and returns a list of outputs.
    """
    
    def __init__(self):
        pass

    def initialize_llm(self):
        """Initialize the LLM. Override in subclasses if needed."""
        raise NotImplementedError

    @abstractmethod
    def prompt(self, prompt_list):
        """
        Send prompts to the LLM and get responses.
        
        Args:
            prompt_list: List of prompts. Each can be:
                - str: Simple prompt string
                - dict: {"agent_query": str, "chat_history": list}
                
        Returns:
            List of outputs, one per input prompt.
            Each output should be a dict with at least {"text": str}
        """
        pass

    def inspect_history(self, last_k, file_dir):
        """Inspect LLM call history. Override in subclasses if supported."""
        raise NotImplementedError


class DspyLLM(LLMBackend):
    """
    DSPy-based LLM backend.
    
    Uses DSPy's chain-of-thought reasoning for structured prompting.
    
    Args:
        openai_api_key: OpenAI API key
        qa: Question-answering signature class
        cot: Chain-of-thought module class
        model: Model name (default: "gpt-4o-mini")
    """
    
    def __init__(self, openai_api_key, qa, cot, model="gpt-4o-mini"):
        super().__init__()
        self.qa = qa
        self.cot = cot
        self.backend = "dspy"
        self.openai_api_key = openai_api_key
        self.model = model

    def initialize_llm(self):
        import dspy
        self.llm = dspy.OpenAI(
            model=self.model, api_key=self.openai_api_key, temperature=0.0
        )
        dspy.settings.configure(lm=self.llm)
        self.predictor = self.cot(self.qa)
        return self.predictor

    def prompt(self, prompt_list):
        agent_outputs = self.call_dspy_agent(prompt_list)
        return agent_outputs

    def call_dspy_agent(self, prompt_inputs):
        agent_outputs = []
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                agent_outputs = list(
                    executor.map(self.dspy_query_and_get_answer, prompt_inputs)
                )
        except Exception as e:
            print(e)
        return agent_outputs

    def dspy_query_and_get_answer(self, prompt_input):
        if type(prompt_input) is str:
            agent_output = self.query_agent(prompt_input, [])
        else:
            agent_output = self.query_agent(
                prompt_input["agent_query"], prompt_input["chat_history"]
            )
        return {"text": agent_output}

    def query_agent(self, query, history):
        pred = self.predictor(question=query, history=history)
        return pred.answer

    def inspect_history(self, last_k, file_dir):
        buffer = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = buffer
        self.llm.inspect_history(last_k)
        printed_data = buffer.getvalue()
        if file_dir is not None:
            save_path = os.path.join(file_dir, "inspect_history.md")
            with open(save_path, "w") as f:
                f.write(printed_data)
        sys.stdout = original_stdout


class LangchainLLM(LLMBackend):
    """
    LangChain/OpenAI chat backend.

    This keeps the older public API used by the docs and COVID model imports:
    `LangchainLLM(openai_api_key=..., agent_profile=..., model=...)`.
    """

    def __init__(
        self,
        openai_api_key,
        agent_profile,
        model="gpt-4o-mini",
        temperature=0.0,
        base_url=None,
        max_tokens=None,
        timeout=None,
        max_concurrency=8,
    ):
        super().__init__()
        self.backend = "langchain"
        self.openai_api_key = openai_api_key
        self.agent_profile = agent_profile
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_concurrency = max(1, int(max_concurrency))
        self.llm = None

    def initialize_llm(self):
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": self.model,
            "api_key": self.openai_api_key,
            "temperature": self.temperature,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout

        self.llm = ChatOpenAI(**kwargs)
        return self.llm

    def prompt(self, prompt_list):
        if self.llm is None:
            self.initialize_llm()

        if len(prompt_list) <= 1 or self.max_concurrency == 1:
            return [self._prompt_one(prompt_input) for prompt_input in prompt_list]

        workers = min(self.max_concurrency, len(prompt_list))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self._prompt_one, prompt_list))

    def _prompt_one(self, prompt_input):
        if isinstance(prompt_input, dict):
            query = prompt_input.get("agent_query", "")
            history = prompt_input.get("chat_history", [])
            system_prompt = prompt_input.get("system_prompt", self.agent_profile)
        else:
            query = prompt_input
            history = []
            system_prompt = self.agent_profile

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self._normalize_history(history))
        messages.append({"role": "user", "content": query})

        response = self.llm.invoke(messages)
        return {"text": getattr(response, "content", str(response))}

    def _normalize_history(self, history):
        messages = []
        for item in history or []:
            if hasattr(item, "content"):
                messages.append(item)
                continue

            if isinstance(item, dict) and "role" in item and "content" in item:
                messages.append(item)
                continue

            if isinstance(item, dict) and "query" in item and "output" in item:
                query = item["query"]
                output = item["output"]
                if isinstance(query, dict):
                    query = query.get("agent_query", str(query))
                if isinstance(output, dict):
                    output = output.get("text", str(output))
                messages.append({"role": "user", "content": str(query)})
                messages.append({"role": "assistant", "content": str(output)})
        return messages

    def inspect_history(self, last_k, file_dir):
        raise NotImplementedError(
            "inspect_history is not implemented for LangchainLLM."
        )
