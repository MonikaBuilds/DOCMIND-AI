import pytest

from app.services.memory_service import ConversationMemoryConfig, ConversationMemoryService


def test_memory_appends_and_returns_history_in_order():
    memory = ConversationMemoryService()

    memory.append_message("session-1", "user", "What is RAG?")
    memory.append_message("session-1", "assistant", "RAG uses retrieval.")

    history = memory.get_history("session-1")

    assert [message.role for message in history] == ["user", "assistant"]
    assert [message.content for message in history] == ["What is RAG?", "RAG uses retrieval."]


def test_memory_trims_old_messages():
    memory = ConversationMemoryService(ConversationMemoryConfig(max_messages_per_session=2))

    memory.append_message("session-1", "user", "first")
    memory.append_message("session-1", "assistant", "second")
    memory.append_message("session-1", "user", "third")

    history = memory.get_history("session-1")

    assert [message.content for message in history] == ["second", "third"]


def test_memory_keeps_sessions_separate():
    memory = ConversationMemoryService()

    memory.append_message("a", "user", "message a")
    memory.append_message("b", "user", "message b")

    assert memory.get_history("a")[0].content == "message a"
    assert memory.get_history("b")[0].content == "message b"


def test_memory_formats_history_for_prompt():
    memory = ConversationMemoryService()
    memory.append_message("session-1", "user", "Question")
    memory.append_message("session-1", "assistant", "Answer")

    formatted = memory.format_history_for_prompt("session-1")

    assert formatted == "User: Question\nAssistant: Answer"


def test_memory_clear_session_removes_messages():
    memory = ConversationMemoryService()
    memory.append_message("session-1", "user", "Question")

    memory.clear_session("session-1")

    assert memory.get_history("session-1") == []


def test_memory_rejects_invalid_role_and_empty_content():
    memory = ConversationMemoryService()

    with pytest.raises(ValueError):
        memory.append_message("session-1", "critic", "Nope")

    with pytest.raises(ValueError):
        memory.append_message("session-1", "user", "   ")


def test_memory_rejects_invalid_config():
    with pytest.raises(ValueError):
        ConversationMemoryService(ConversationMemoryConfig(max_messages_per_session=0))
