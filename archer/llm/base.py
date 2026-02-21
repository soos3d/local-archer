"""Abstract base class for Language Model providers."""

from abc import ABC, abstractmethod
from collections.abc import Iterator


class BaseLLM(ABC):
    """Abstract interface for language model providers."""

    @abstractmethod
    def generate(self, text: str, session_id: str = "default") -> str:
        """
        Generate a response to the input text.

        Args:
            text: User input text.
            session_id: Session identifier for conversation history.

        Returns:
            Generated response text.
        """
        pass

    @abstractmethod
    def clear_history(self, session_id: str = "default") -> None:
        """
        Clear conversation history for a session.

        Args:
            session_id: Session identifier to clear.
        """
        pass

    @abstractmethod
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set or update the system prompt.

        Args:
            prompt: New system prompt.
        """
        pass

    def stream(self, text: str, session_id: str = "default") -> Iterator[str]:
        """
        Stream response as complete sentences.

        Default implementation falls back to generate() and yields the full
        response as a single chunk. Providers may override for true streaming.

        Args:
            text: User input text.
            session_id: Session identifier for conversation history.

        Yields:
            Complete sentences from the response.
        """
        yield self.generate(text, session_id)
