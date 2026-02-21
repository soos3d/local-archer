"""Unit tests for archer.llm.ollama_llm."""

import pytest
from unittest.mock import MagicMock, patch

from archer.core.config import LLMConfig, PersonalityConfig


def _make_ollama_llm(system_prompt="You are a helpful AI assistant.", model="gemma3"):
    """Create an OllamaLLM with LangChain deps mocked."""
    from archer.llm.ollama_llm import OllamaLLM

    config = LLMConfig(provider="ollama", model=model, base_url="http://localhost:11434")
    personality = PersonalityConfig(system_prompt=system_prompt)

    with (
        patch("archer.llm.ollama_llm.LangchainOllama") as MockLangchain,
        patch("archer.llm.ollama_llm.RunnableWithMessageHistory") as MockHistory,
    ):
        MockLangchain.return_value = MagicMock()
        mock_chain = MagicMock()
        MockHistory.return_value = mock_chain
        llm = OllamaLLM(config, personality)
        llm._chain_with_history = mock_chain

    return llm


class TestOllamaLLMGenerate:
    def test_generate_returns_stripped_response(self):
        llm = _make_ollama_llm()
        llm._chain_with_history.invoke.return_value = "  hello world  "
        result = llm.generate("hello", "test_session")
        assert result == "hello world"

    def test_generate_uses_correct_session_id(self):
        llm = _make_ollama_llm()
        llm._chain_with_history.invoke.return_value = "response"
        llm.generate("hello", "my_session")
        call_kwargs = llm._chain_with_history.invoke.call_args[1]
        assert call_kwargs["config"]["configurable"]["session_id"] == "my_session"

    def test_generate_passes_input_text(self):
        llm = _make_ollama_llm()
        llm._chain_with_history.invoke.return_value = "response"
        llm.generate("what is the weather?", "session")
        call_args = llm._chain_with_history.invoke.call_args[0][0]
        assert call_args["input"] == "what is the weather?"


class TestOllamaLLMSessionHistory:
    def test_get_session_history_creates_new(self):
        llm = _make_ollama_llm()
        history = llm._get_session_history("new_session")
        assert "new_session" in llm._sessions
        assert llm._sessions["new_session"] is history

    def test_get_session_history_reuses_existing(self):
        llm = _make_ollama_llm()
        history1 = llm._get_session_history("session_a")
        history2 = llm._get_session_history("session_a")
        assert history1 is history2

    def test_get_session_history_different_sessions_are_distinct(self):
        llm = _make_ollama_llm()
        history_a = llm._get_session_history("session_a")
        history_b = llm._get_session_history("session_b")
        assert history_a is not history_b


class TestOllamaLLMClearHistory:
    def test_clear_existing_session(self):
        llm = _make_ollama_llm()
        llm._get_session_history("session_x")
        # Replace with a mock so we can verify clear() was called
        mock_history = MagicMock()
        llm._sessions["session_x"] = mock_history
        llm.clear_history("session_x")
        mock_history.clear.assert_called_once()

    def test_clear_nonexistent_session_no_error(self):
        llm = _make_ollama_llm()
        # Should not raise
        llm.clear_history("nonexistent_session")


class TestOllamaLLMSetSystemPrompt:
    def test_set_system_prompt_updates_personality(self):
        llm = _make_ollama_llm()
        llm.set_system_prompt("New system prompt.")
        assert llm.personality.system_prompt == "New system prompt."

    def test_set_system_prompt_recreates_chain(self):
        llm = _make_ollama_llm()
        with patch("archer.llm.ollama_llm.RunnableWithMessageHistory") as MockHistory:
            new_chain = MagicMock()
            MockHistory.return_value = new_chain
            llm.set_system_prompt("Updated prompt.")
            assert llm._chain_with_history is new_chain


class TestOllamaLLMInit:
    def test_init_creates_langchain_with_correct_config(self):
        from archer.llm.ollama_llm import OllamaLLM

        config = LLMConfig(provider="ollama", model="llama3", base_url="http://custom:11434")
        personality = PersonalityConfig()

        with (
            patch("archer.llm.ollama_llm.LangchainOllama") as MockLangchain,
            patch("archer.llm.ollama_llm.RunnableWithMessageHistory"),
        ):
            MockLangchain.return_value = MagicMock()
            OllamaLLM(config, personality)
            MockLangchain.assert_called_once_with(
                model="llama3",
                base_url="http://custom:11434",
            )
