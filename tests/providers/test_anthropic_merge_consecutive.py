from nanobot.providers.anthropic_provider import AnthropicProvider
from nanobot.utils.helpers import estimate_prompt_tokens, estimate_prompt_tokens_chain


class TestMergeConsecutive:

    def test_basic_alternation(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "bye"},
        ]
        result = AnthropicProvider._merge_consecutive(msgs)
        assert len(result) == 3
        assert [m["role"] for m in result] == ["user", "assistant", "user"]

    def test_consecutive_same_role_merged(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "reply"},
        ]
        result = AnthropicProvider._merge_consecutive(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_trailing_assistant_stripped(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = AnthropicProvider._merge_consecutive(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hello"

    def test_multiple_trailing_assistant_stripped(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "b"},
            {"role": "assistant", "content": "c"},
        ]
        result = AnthropicProvider._merge_consecutive(msgs)
        assert len(result) == 3
        assert result[-1]["role"] == "user"
        assert result[-1]["content"] == "ok"

    def test_empty_messages(self):
        assert AnthropicProvider._merge_consecutive([]) == []

    def test_single_user_message(self):
        msgs = [{"role": "user", "content": "hi"}]
        result = AnthropicProvider._merge_consecutive(msgs)
        assert len(result) == 1

    def test_single_assistant_rerouted_to_user(self):
        """When stripping leaves nothing, the last assistant is rerouted to
        ``user`` so we don't produce an empty messages array."""
        msgs = [{"role": "assistant", "content": "hi"}]
        result = AnthropicProvider._merge_consecutive(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "hi"

    def test_all_assistants_collapse_then_rerouted(self):
        """Consecutive trailing assistants merge into one, which is then
        rerouted as a user turn carrying the merged content."""
        msgs = [
            {"role": "assistant", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
        result = AnthropicProvider._merge_consecutive(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
        ]

    def test_assistant_with_tool_use_not_rerouted(self):
        """A trailing assistant carrying ``tool_use`` blocks cannot become a
        user turn (Anthropic rejects ``tool_use`` inside user messages), so
        the method returns an empty list rather than forging a bad request."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "let me search"},
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {}},
                ],
            }
        ]
        result = AnthropicProvider._merge_consecutive(msgs)
        assert result == []

    def test_leading_assistant_gets_synthetic_user(self):
        """If the first turn is a bare assistant (e.g. history truncation
        dropped the original user request), prepend a synthetic opener so
        the conversation still starts with ``user``."""
        msgs = [
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "reply"},
        ]
        result = AnthropicProvider._merge_consecutive(msgs)
        assert [m["role"] for m in result] == ["user", "assistant", "user"]
        assert result[0]["content"] == "(conversation continued)"
        assert result[1]["content"] == "hi"
        assert result[2]["content"] == "ok"

    def test_leading_tool_use_gets_opener_without_splitting_pair(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                ],
            },
        ]
        result = AnthropicProvider._merge_consecutive(msgs)
        assert result[0] == {"role": "user", "content": "(conversation continued)"}
        assert result[1:] == msgs


def test_request_budget_counts_the_anthropic_replay_opener(monkeypatch):
    monkeypatch.setattr("nanobot.utils.helpers._get_token_encoding", lambda: None)
    provider = object.__new__(AnthropicProvider)
    raw = [
        {"role": "system", "content": "Previous checkpoint."},
        {"role": "assistant", "tool_calls": [{
            "id": "call-1", "type": "function", "function": {"name": "search", "arguments": "{}"},
        }]},
        {"role": "tool", "tool_call_id": "call-1", "content": "Found it."},
    ]
    _, wire_messages = provider._convert_messages(raw)
    opener = wire_messages[0]
    assert opener["role"] == "user"
    cost, _ = estimate_prompt_tokens_chain(provider, None, raw)
    assert cost == estimate_prompt_tokens([raw[0], opener, *raw[1:]])
    assert cost > estimate_prompt_tokens(raw)
