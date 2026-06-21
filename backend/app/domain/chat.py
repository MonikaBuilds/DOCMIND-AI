from dataclasses import dataclass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class CitedAnswer:
    answer: str
    citations: list[str]


@dataclass(frozen=True)
class GroundedPrompt:
    system_prompt: str
    user_prompt: str
