from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque

from app.domain.chat import ChatMessage


@dataclass(frozen=True)
class ConversationMemoryConfig:
    max_messages_per_session: int = 12


class ConversationMemoryService:
    """Stores bounded chat history separately from document-grounded context."""

    allowed_roles = {"user", "assistant", "system"}

    def __init__(self, config: ConversationMemoryConfig | None = None) -> None:
        self.config = config or ConversationMemoryConfig()
        if self.config.max_messages_per_session <= 0:
            raise ValueError("max_messages_per_session must be greater than zero.")

        self._messages: dict[str, Deque[ChatMessage]] = defaultdict(
            lambda: deque(maxlen=self.config.max_messages_per_session)
        )

    def append_message(self, session_id: str, role: str, content: str) -> ChatMessage:
        clean_session_id = self._validate_session_id(session_id)
        clean_role = self._validate_role(role)
        clean_content = content.strip()
        if not clean_content:
            raise ValueError("content cannot be empty.")

        message = ChatMessage(role=clean_role, content=clean_content)
        self._messages[clean_session_id].append(message)
        return message

    def get_history(self, session_id: str) -> list[ChatMessage]:
        clean_session_id = self._validate_session_id(session_id)
        return list(self._messages[clean_session_id])

    def clear_session(self, session_id: str) -> None:
        clean_session_id = self._validate_session_id(session_id)
        self._messages.pop(clean_session_id, None)

    def format_history_for_prompt(self, session_id: str) -> str:
        history = self.get_history(session_id)
        return "\n".join(
            f"{message.role.title()}: {message.content}"
            for message in history
        )

    def _validate_session_id(self, session_id: str) -> str:
        clean_session_id = session_id.strip()
        if not clean_session_id:
            raise ValueError("session_id cannot be empty.")
        return clean_session_id

    def _validate_role(self, role: str) -> str:
        clean_role = role.strip().lower()
        if clean_role not in self.allowed_roles:
            raise ValueError(f"Unsupported chat role: {role}")
        return clean_role
