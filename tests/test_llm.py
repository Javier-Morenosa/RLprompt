"""Tests for the LLM module."""

from prompt_rl.core.prompt import Prompt
from prompt_rl.llm import MockLLM, LLMResponse


def test_mock_llm_complete() -> None:
    llm = MockLLM(default_response="OK")
    resp = llm.complete("Hello")
    assert isinstance(resp, LLMResponse)
    assert resp.text == "OK"
    assert llm.call_count == 1


def test_mock_llm_with_prompt_object() -> None:
    llm = MockLLM()
    p = Prompt("Test prompt")
    resp = llm.complete(p)
    assert "[Refined]" in resp.text or len(resp.text) > 0
