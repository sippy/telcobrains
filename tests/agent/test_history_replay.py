from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import LLMResponse
from nanobot.session.manager import Session
from nanobot.utils.helpers import find_legal_message_start


def _make_loop(tmp_path: Path, context_window_tokens: int = 200_000) -> AgentLoop:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation.max_tokens = 4096
    return AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        context_window_tokens=context_window_tokens,
    )


def _populated_session(turns: int) -> Session:
    session = Session(key="test:populated")
    for index in range(turns):
        session.add_message("user", f"msg-{index}")
        session.add_message("assistant", f"reply-{index}")
    return session


def _tool_round(call_id: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": "x", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "name": "x", "content": "ok"},
    ]


def _checkpointed_assistant_tail() -> Session:
    """A session whose committed checkpoint left an assistant-leading tail."""
    session = Session(key="test:mid-turn")
    session.messages = [
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "big first instruction"},
        *_tool_round("c1"),
        {"role": "assistant", "content": "final answer"},
        {"role": "user", "content": "next question"},
        {"role": "assistant", "content": "next answer"},
    ]
    session.commit_summary_checkpoint("summary", insert_at=3)
    return session


def test_checkpointed_assistant_tail_replays_verbatim() -> None:
    """A mid-turn cut's assistant-leading tail stays whole, including bounded reads."""
    session = _checkpointed_assistant_tail()

    history = session.get_history(include_runtime_context=False)
    assert [message.get("content") for message in history] == [
        None, "ok", "final answer", "next question", "next answer",
    ]
    assert history[0]["tool_calls"][0]["id"] == "c1"
    assert find_legal_message_start(history) == 0
    # A bounded estimate that still fits the whole tail must not drop the
    # retained leading assistant tool round to realign on the next user turn.
    bounded = session.get_text_history(max_tokens=200, include_runtime_context=False)
    assert [message.get("content") for message in bounded] == [
        None, "ok", "final answer", "next question", "next answer",
    ]


def test_verbatim_assistant_tail_bounded_estimate_keeps_retained_lead() -> None:
    session = _checkpointed_assistant_tail()

    # The budget trims the leading tool round, but the retained "final answer"
    # (whose question was archived) must survive instead of being dropped to
    # realign the head on the next user turn.
    bounded = session.get_text_history(max_tokens=60, include_runtime_context=False)
    assert [message.get("content") for message in bounded] == [
        "final answer", "next question", "next answer",
    ]
    assert find_legal_message_start(bounded) == 0


def test_legacy_head_still_realigns_to_first_user_turn() -> None:
    """A head without a checkpoint keeps the existing mid-turn realignment."""
    session = Session(key="test:legacy")
    session.messages = [
        {"role": "assistant", "content": "orphaned reply"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
    ]

    assert [m.get("content") for m in session.get_history(include_runtime_context=False)] == [
        "first question", "first answer",
    ]
    assert [m.get("content") for m in session.get_text_history(max_tokens=60, include_runtime_context=False)] == [
        "first question", "first answer",
    ]


def test_message_index_maps_verbatim_assistant_tail_positions() -> None:
    session = _checkpointed_assistant_tail()

    # Replayable positions after the checkpoint: 0=assistant(c1), 1=result,
    # 2=final answer, 3=next question, 4=next answer.
    assert session.message_index_at_history_position(0) == 4
    assert session.message_index_at_history_position(1) == 5
    assert session.message_index_at_history_position(4) == 8
    assert session.message_index_at_history_position(5) is None


def test_default_history_has_no_message_count_limit() -> None:
    session = _populated_session(1_001)

    history = session.get_history()

    assert len(history) == 2_002
    assert history[0]["content"] == "msg-0"
    assert history[-1]["content"] == "reply-1000"


def test_explicit_message_limit_still_starts_at_user_turn() -> None:
    history = _populated_session(30).get_history(max_messages=25)

    assert len(history) <= 25
    assert history[0]["role"] == "user"


@pytest.mark.asyncio
async def test_process_message_hands_complete_replay_to_runner(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, context_window_tokens=32_768)
    loop.provider.chat_stream_with_retry = AsyncMock(
        return_value=LLMResponse(content="ok", tool_calls=[], usage=None)
    )
    loop.tools.get_definitions = MagicMock(return_value=[])

    session = await loop.sessions.state.import_messages("cli:test", _populated_session(30).messages)
    expected_history = session.get_history()
    loop.provider.aclose = AsyncMock()
    result = await loop.process_direct("hello", session_key="cli:test")

    assert result is not None
    sent = loop.provider.chat_stream_with_retry.await_args.kwargs["messages"]
    assert sent[1:-1] == expected_history
    assert sent[-1]["content"].startswith("hello\n\n")
    await loop.aclose()


@pytest.mark.asyncio
async def test_runner_checkpoint_keeps_current_user_as_replay_boundary(tmp_path: Path) -> None:
    loop = _make_loop(tmp_path, context_window_tokens=8_000)
    loop.provider.estimate_prompt_tokens.side_effect = (
        lambda messages, tools, model: (100 * len(messages), "test")
    )
    loop.provider.chat_stream_with_retry = AsyncMock(
        return_value=LLMResponse(content="ok", tool_calls=[], usage=None)
    )
    loop.tools.get_definitions = MagicMock(return_value=[])

    session = loop.sessions.get_or_create("cli:test")
    session.add_message("user", "old")
    session.add_message("assistant", "old answer")
    session.add_message("user", "long older turn")
    for index in range(70):
        session.messages.extend(_tool_round(f"older-{index}"))
    session.add_message("assistant", "older final")

    result = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="test",
            content="new question",
        )
    )

    assert result is not None
    sent_messages = loop.provider.chat_stream_with_retry.await_args.kwargs["messages"]
    sent_text = "\n".join(str(message.get("content")) for message in sent_messages)
    assert "new question" in sent_text
    assert [message["role"] for message in sent_messages] == ["system", "user"]
    assert sent_messages[1]["content"].startswith("new question\n\n")
    assert "File access:" in sent_messages[1]["content"]
    assert any(message.get("content") == "long older turn" for message in session.messages)
