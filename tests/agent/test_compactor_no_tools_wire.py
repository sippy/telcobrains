"""Wire-level regression: configured compactor requests must stay tool-free.

Round 1 asserted the archiver boundary with a mocked provider.  These tests
drive the tool-free summarization request through the real OpenAI-compatible
provider and capture the outgoing JSON body on the wire (Chat Completions
and Responses APIs), in both request shapes: full transcript replay and
native-state replay.  The request must send no ``tools`` and no
``tool_choice`` even when the model preset's ``extra_body`` configures
managed tools and a ``tool_choice`` override that the SDK would otherwise
merge on top of the request.  Ordinary calls (tools or empty tools without
the no-tools contract) keep their previous wire shape, including configured
managed tools.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import AsyncOpenAI

from nanobot.agent.memory import MemoryArchiver, MemoryStore
from nanobot.providers.base import GenerationSettings, LLMResponse, ProviderConversationState
from nanobot.providers.model_api import ModelAPICapabilities
from nanobot.providers.openai_compat_provider import OpenAICompatProvider
from nanobot.utils.llm_runtime import LLMRuntime

_API_BASE = "http://localhost:9/v1"

TASK_TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the workspace.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
    },
}]

# A "managed tool" configured through the preset's extra_body. Deliberately
# not a web_search tool so the Responses API is not force-enabled by it.
MANAGED_TOOLS = [{
    "type": "function",
    "function": {
        "name": "managed_lookup",
        "description": "Configured managed tool.",
        "parameters": {"type": "object", "properties": {}},
    },
}]

HISTORY = [
    {"role": "user", "content": "old question"},
    {"role": "assistant", "content": "old answer"},
]


def _answer(request: httpx.Request) -> httpx.Response:
    """Serve a minimal Chat Completions or Responses API reply."""
    if request.url.path.endswith("/responses"):
        response = {
            "id": "resp_fixture", "object": "response", "status": "completed",
            "output": [{
                "id": "msg_fixture", "type": "message", "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }],
        }
        events = [
            {"type": "response.output_text.delta", "delta": "ok"},
            {"type": "response.completed", "response": response},
        ]
    else:
        response = {"id": "chat_fixture", "choices": [{
            "index": 0, "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }]}
        events = [{"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": "stop"}]}]
    if not json.loads(request.content).get("stream"):
        return httpx.Response(200, json=response)
    if request.url.path.endswith("/responses"):
        content = "".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
        )
    else:
        # Chat Completions SSE is unnamed: plain data lines plus [DONE].
        content = (
            "".join(f"data: {json.dumps(event)}\n\n" for event in events)
            + "data: [DONE]\n\n"
        )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=content,
    )


def _make_provider(
    *, responses_api: bool = False, extra_body: dict[str, Any] | None = None,
) -> tuple[OpenAICompatProvider, list[httpx.Request]]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _answer(request)

    provider = OpenAICompatProvider(
        api_key="fixture",
        api_base=_API_BASE,
        default_model="compactor-model",
        extra_body=extra_body,
        model_api=(
            ModelAPICapabilities(supported_apis=("responses",), preferred_api="responses")
            if responses_api else None
        ),
    )
    client = AsyncOpenAI(
        api_key="fixture",
        base_url=_API_BASE,
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    provider._client = client  # pyright: ignore[reportPrivateUsage]
    return provider, requests


def _runtime(provider: OpenAICompatProvider) -> LLMRuntime:
    return LLMRuntime(
        provider=provider,
        model="compactor-model",
        generation=GenerationSettings(max_tokens=512),
        context_window_tokens=20_000,
    )


def _archiver(tmp_path) -> MemoryArchiver:
    return MemoryArchiver(
        store=MemoryStore(tmp_path),
        build_messages=lambda **_: [],
        get_tool_definitions=lambda: TASK_TOOLS,
    )


async def _summarize(
    provider: OpenAICompatProvider,
    tmp_path,
    request_tools: list[dict[str, Any]],
    no_tool_contract: bool = True,
) -> str | None:
    """Run a configured-compactor summarization request through the real provider."""
    summary = await _archiver(tmp_path).archive(
        list(HISTORY),
        runtime=_runtime(provider),
        session_key="cli:wire",
        history=list(HISTORY),
        request_tools=request_tools,
        input_token_budget=20_000,
        persist=False,
        no_tool_contract=no_tool_contract,
    )
    assert summary == "ok"
    return summary


async def test_compactor_request_is_tool_free_on_chat_wire(tmp_path) -> None:
    """A tool-free compactor request stays tool-free despite configured extra_body tools."""
    provider, requests = _make_provider(extra_body={
        "tools": MANAGED_TOOLS,
        "tool_choice": "auto",
    })

    await _summarize(provider, tmp_path, request_tools=[])

    request, = requests
    assert request.url.path.endswith("/chat/completions")
    body = json.loads(request.content)
    assert body["model"] == "compactor-model"
    # The request offers no tools at all: no task tools, no configured
    # managed tools, and no tool_choice (strict endpoints reject it without
    # tools). The summarizer therefore cannot start calling tools.
    assert "tools" not in body
    assert "tool_choice" not in body


async def test_compactor_request_is_tool_free_on_responses_wire(tmp_path) -> None:
    """Same guarantee on the Responses API surface."""
    provider, requests = _make_provider(
        responses_api=True,
        extra_body={"tools": MANAGED_TOOLS, "tool_choice": "auto"},
    )

    await _summarize(provider, tmp_path, request_tools=[])

    request, = requests
    assert request.url.path.endswith("/responses")
    body = json.loads(request.content)
    assert body["model"] == "compactor-model"
    assert "tools" not in body
    assert "tool_choice" not in body


def _responses_state_with_tool_history() -> ProviderConversationState:
    """Native state replaying a historical transcript that used a tool.

    The state persists conversation items only — never tool schemas — so a
    tool-free request can replay the prior transcript as-is.
    """
    return ProviderConversationState(
        kind="openai_responses",
        provider=f"openai_compat:custom:{_API_BASE}",
        model="compactor-model",
        version=1,
        payload={
            "context_tokens": 100,
            "items": [
                {
                    "id": "msg1", "type": "message", "role": "user",
                    "status": "completed",
                    "content": [{"type": "input_text", "text": "read the file"}],
                },
                {
                    "id": "fc1", "type": "function_call", "call_id": "call_1",
                    "name": "read_file", "arguments": "{\"path\": \"x\"}",
                    "status": "completed",
                },
                {
                    "id": "fco1", "type": "function_call_output", "call_id": "call_1",
                    "output": "file contents", "status": "completed",
                },
                {
                    "id": "msg2", "type": "message", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "done", "annotations": []}],
                },
            ],
        },
    )


async def test_native_state_compactor_request_is_tool_free_on_responses_wire(
    tmp_path,
) -> None:
    """A tool-free compactor request over native-state replay stays tool-free
    on the wire despite configured extra_body tools, and the replayed
    transcript — including its historical tool call — is preserved."""
    provider, requests = _make_provider(
        responses_api=True,
        extra_body={"tools": MANAGED_TOOLS, "tool_choice": "auto"},
    )

    summary = await _archiver(tmp_path).archive(
        list(HISTORY),
        runtime=_runtime(provider),
        session_key="cli:wire",
        history=list(HISTORY),
        request_tools=[],
        input_token_budget=20_000,
        persist=False,
        provider_state=_responses_state_with_tool_history(),
        no_tool_contract=True,
    )

    assert summary == "ok"
    request, = requests
    assert request.url.path.endswith("/responses")
    body = json.loads(request.content)
    # The request stays tool-free: no task tools, no configured managed
    # tools, and no tool_choice.
    assert "tools" not in body
    assert "tool_choice" not in body
    # Native-state replay is intact: the historical transcript, including
    # the tool call and its output, is replayed as input items, and the
    # archive prompt follows as the final user item.
    items = body["input"]
    assert [item.get("type") for item in items[:4]] == [
        "message", "function_call", "function_call_output", "message",
    ]
    assert items[-1]["role"] == "user"


async def test_legacy_empty_registry_compaction_keeps_wire_shape(tmp_path) -> None:
    """Legacy no-override compaction of a session whose tool registry is
    empty keeps the pre-compactor-policy wire shape: no no-tools contract is
    injected, so configured extra_body tools ride along exactly as they do
    on any ordinary tool-less call."""
    provider, requests = _make_provider(extra_body={
        "tools": MANAGED_TOOLS,
        "tool_choice": "auto",
    })

    await _summarize(
        provider, tmp_path, request_tools=[], no_tool_contract=False,
    )

    request, = requests
    assert request.url.path.endswith("/chat/completions")
    body = json.loads(request.content)
    assert body["tools"] == MANAGED_TOOLS
    assert body["tool_choice"] == "auto"


async def test_ordinary_empty_tools_call_keeps_configured_tools(tmp_path) -> None:
    """Default unchanged: an ordinary tool-less call (no no-tools contract)
    still receives the configured managed tools with the provider's default
    choice behavior."""
    provider, requests = _make_provider(extra_body={
        "tools": MANAGED_TOOLS,
        "tool_choice": "auto",
    })

    reply = await provider.chat(
        messages=[{"role": "user", "content": "hello"}],
        model="compactor-model",
        tools=[],
    )

    assert reply.content == "ok"
    body = json.loads(requests[0].content)
    assert body["tools"] == MANAGED_TOOLS
    # The configured extra_body tool_choice rides along on the wire, as before
    # (valid here because managed tools are offered).
    assert body["tool_choice"] == "auto"


async def test_ordinary_tools_call_keeps_auto_choice_and_managed_tools(tmp_path) -> None:
    """Default unchanged: a normal call with tools keeps managed tools merged
    and the usual auto choice."""
    provider, requests = _make_provider(extra_body={
        "tools": MANAGED_TOOLS,
        "tool_choice": "auto",
    })

    reply = await provider.chat(
        messages=[{"role": "user", "content": "hello"}],
        model="compactor-model",
        tools=TASK_TOOLS,
    )

    assert reply.content == "ok"
    body = json.loads(requests[0].content)
    assert body["tools"] == [*TASK_TOOLS, *MANAGED_TOOLS]
    assert body["tool_choice"] == "auto"


async def test_explicit_none_with_tools_forbids_calls_on_chat_wire(tmp_path) -> None:
    """A caller that offers tools but forbids calls keeps the schemas and
    forces tool_choice='none' so a configured extra_body override cannot
    re-enable calls."""
    provider, requests = _make_provider(extra_body={
        "tools": MANAGED_TOOLS,
        "tool_choice": "auto",
    })

    reply = await provider.chat(
        messages=[{"role": "user", "content": "hello"}],
        model="compactor-model",
        tools=TASK_TOOLS,
        tool_choice="none",
    )

    assert reply.content == "ok"
    body = json.loads(requests[0].content)
    assert body["tools"] == [*TASK_TOOLS, *MANAGED_TOOLS]
    assert body["tool_choice"] == "none"


async def test_explicit_none_with_tools_forbids_calls_on_responses_wire(tmp_path) -> None:
    provider, requests = _make_provider(
        responses_api=True,
        extra_body={"tools": MANAGED_TOOLS, "tool_choice": "auto"},
    )

    reply = await provider.chat(
        messages=[{"role": "user", "content": "hello"}],
        model="compactor-model",
        tools=TASK_TOOLS,
        tool_choice="none",
    )

    assert reply.content == "ok"
    body = json.loads(requests[0].content)
    # Responses API converts the caller's tool schemas to the flat function
    # format; extra_body managed tools keep their configured (nested) shape —
    # pre-existing merge behavior, asserted here to stay stable.
    names = [
        tool.get("name") or (tool.get("function") or {}).get("name")
        for tool in body["tools"]
    ]
    assert names == ["read_file", "managed_lookup"]
    assert body["tool_choice"] == "none"


def _fake_provider() -> MagicMock:
    provider = MagicMock(name="provider")
    provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(
        content="Checkpoint.", tool_calls=[],
    ))
    return provider


def _fake_state() -> SimpleNamespace:
    state = SimpleNamespace(payload={"context_tokens": 100}, pending_messages=[])
    state.with_pending_messages = lambda messages: SimpleNamespace(
        payload=dict(state.payload), pending_messages=list(messages),
    )
    return state


@pytest.mark.parametrize(
    (
        "request_tools", "provider_state", "no_tool_contract",
        "expected_tool_choice", "expected_tools",
    ),
    [
        # Configured compactor, full transcript replay: the tool-free request
        # carries the explicit no-tools contract.
        ([], None, True, "none", []),
        # Configured compactor, native-state replay: the same contract
        # applies; provider state persists items, not tool schemas.
        ([], "state", True, "none", []),
        # Legacy no-override: the session's tools are offered, no contract.
        (TASK_TOOLS, None, False, None, TASK_TOOLS),
        # Legacy no-override with an empty session tool registry: the
        # request shape is unchanged (no contract injected).
        ([], None, False, None, []),
        # Legacy native-state replay keeps its existing request shape.
        (TASK_TOOLS, "state", False, None, []),
    ],
)
async def test_archiver_no_tools_contract_at_provider_boundary(
    tmp_path,
    request_tools,
    provider_state,
    no_tool_contract,
    expected_tool_choice,
    expected_tools,
) -> None:
    """The configured compactor says 'no tools' explicitly
    (tool_choice='none'), which is the contract adapters enforce; legacy
    requests — including an empty session tool registry — are unchanged."""
    state = _fake_state() if provider_state is not None else None
    provider = _fake_provider()
    if state is not None:
        provider.can_resume_conversation_state.return_value = True
    runtime = LLMRuntime(
        provider=provider, model="compactor-model",
        generation=GenerationSettings(max_tokens=512),
        context_window_tokens=20_000,
    )

    await _archiver(tmp_path).archive(
        list(HISTORY),
        runtime=runtime,
        session_key="cli:contract",
        history=list(HISTORY),
        request_tools=request_tools,
        input_token_budget=20_000,
        persist=False,
        provider_state=state,
        no_tool_contract=no_tool_contract,
    )

    provider.chat_stream_with_retry.assert_awaited_once()
    kwargs = provider.chat_stream_with_retry.await_args.kwargs
    assert kwargs["tools"] == expected_tools
    assert kwargs["tool_choice"] == expected_tool_choice
