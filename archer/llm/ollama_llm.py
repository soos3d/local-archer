"""Ollama Language Model implementation."""

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import OllamaLLM as LangchainOllama

from archer.core.config import LLMConfig, PersonalityConfig
from archer.llm.base import BaseLLM


class OllamaLLM(BaseLLM):
    """Ollama language model provider using LangChain."""

    def __init__(self, config: LLMConfig, personality: PersonalityConfig):
        """
        Initialize Ollama LLM.

        Args:
            config: LLM configuration.
            personality: Personality configuration with system prompt.
        """
        self.config = config
        self.personality = personality
        self._sessions: dict[str, InMemoryChatMessageHistory] = {}

        # Initialize LLM
        self._llm = LangchainOllama(
            model=config.model,
            base_url=config.base_url,
        )

        # Create prompt template
        self._prompt_template = ChatPromptTemplate.from_messages([
            ("system", personality.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        # Create chain
        self._chain = self._prompt_template | self._llm

        # Create chain with history
        self._chain_with_history = RunnableWithMessageHistory(
            self._chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Get or create chat history for a session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = InMemoryChatMessageHistory()
        return self._sessions[session_id]

    def generate(self, text: str, session_id: str = "default") -> str:
        """
        Generate a response to the input text.

        Args:
            text: User input text.
            session_id: Session identifier for conversation history.

        Returns:
            Generated response text.
        """
        response = self._chain_with_history.invoke(
            {"input": text},
            config={"configurable": {"session_id": session_id}},
        )
        return response.strip()

    def clear_history(self, session_id: str = "default") -> None:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier to clear.
        """
        if session_id in self._sessions:
            self._sessions[session_id].clear()

    def set_system_prompt(self, prompt: str) -> None:
        """
        Set or update the system prompt.

        Args:
            prompt: New system prompt.
        """
        self.personality.system_prompt = prompt

        # Recreate prompt template and chain
        self._prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        self._chain = self._prompt_template | self._llm
        self._chain_with_history = RunnableWithMessageHistory(
            self._chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
