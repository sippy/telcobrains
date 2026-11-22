# TelcoBrains nanobot customizations

The `telcobrain-production` branch is based on upstream nanobot `v0.4.2`. This file records the branch changes and the separate host configuration used to run them; `docs/` is the canonical documentation for each feature.

## Native subagents can use a different model

The `subagent` tool accepts an optional `model_preset` on `create` (for example, `local-qwen-sglang`); the child uses that preset while the parent keeps its selected model, and omitting it preserves inheritance. Unknown presets return the list of available presets; inline and background runs both honor the preset. See [Subagents](docs/concepts.md#subagents).

## Follow up on completed, cancelled, or iteration-limited subagents

The `subagent` tool accepts `action: "follow_up"` with a completed, cancelled, or iteration-limit-stopped task's `task_id` and a new `message`. It starts a new linked subtask that replays the saved history and tool results without rerunning old tool calls; each new run starts with a fresh full iteration budget counted from its own start, and it works after a restart while the history is saved. Interrupted and errored tasks cannot be followed up. Use `send` for queued or running tasks; wait for cancellation to finish before following up. Cancelled tasks retain only saved history. The original named preset is reused unless overridden, and only the owning persistent conversation can follow up. Follow-up does not resume the old execution or restore a process or its in-memory state. See [Subagents](docs/concepts.md#subagents).

If an inline run in a persistent conversation hits the limit, its result includes the task ID to use for `follow_up`. If the original run has no saved model preset, pass `model_preset` explicitly.

## Import OpenClaw conversations

Run the importer from this checkout with the host's nanobot Python environment; preview with `--dry-run` first, then rerun without it. Refresh the WebUI and look for the topic marked **(OpenClaw)**. It copies visible text and timestamps, not attachments, tool activity, or runtime state; reruns skip duplicates. See [Import an OpenClaw conversation](docs/cli-reference.md#import-an-openclaw-conversation) for alternate source databases and destination workspaces.

## Consolidate with a dedicated compactor model

`agents.defaults.compactor.modelOverride` selects a dedicated model for context compaction (including subagent contexts), manual `/compact`, SDK session compaction, and idle archival. It changes summarization, not the model doing the task. Set it to a named `modelPresets` entry, with the endpoint configured on that preset's provider; the implicit `default` is rejected. The preset's `maxTokens` and `contextWindowTokens` bound the summarization call and cap the checkpoint. Current exception: `/new` background archival still uses the session model.

Foreground context-pressure compaction, idle archival, manual `/compact`, and SDK session compaction summarize an older prefix and keep a recent tail verbatim when both the compactor and replay budgets allow it. If no legal cut fits, they use the existing full-prefix path. Idle archival does not compact the same retained tail repeatedly without new activity.

`modelOverride` is also settable from the WebUI under Settings → System → Sessions: a preset picker whose options are the named `modelPresets` entries; leaving it unset keeps the session model. Existing validation (unknown and implicit `default` presets rejected) and the no-tools compactor routing are unchanged.

`agents.defaults.compactor.nextTurnInputReserveTokens` (default `0`, also in WebUI Settings → System → Sessions) reserves that many foreground input tokens for the next turn's fresh input when idle, manual `/compact`, or SDK session compaction sizes the retained verbatim tail, including the projected post-compaction probe. It never reduces compactor capacity or applies to in-turn compaction, which already sizes against the real pending input. It is opt-in: a larger reserve keeps less recent history verbatim; when no cut still fits, the existing full-prefix compaction is used.

Limitations: if the input does not fit the compactor's context window or the call fails, nanobot writes a raw (unsummarized) checkpoint instead, with no retry. If the compactor is a different provider than the session model, provider-native compaction summarizes the materialized transcript instead of resuming the foreign provider's private state. This is model routing and replay sizing only. See [Compactor](docs/configuration.md#compactor).

## Local deployment settings (outside Git)

- Pending rollout: set `agents.defaults.maxToolIterations` to `1000` in the host's `~/.nanobot/config.json`. This sets the ordinary budget for the main agent and each subagent run, including follow-ups. An iteration is a model round and may include multiple tool calls. The upstream default remains `200`. Verify the effective gateway value with `my(action="check", key="max_iterations")` after applying the change.
- The LAN WebUI listens at `http://192.168.23.109:8765`, with token authentication enabled. The password remains in the host's private nanobot config, not in this repository. The endpoint is HTTP and should only be used on a trusted LAN.
- The `local-qwen-sglang` preset points to provider `local-a40` at `http://127.0.0.1:8081/v1`. The served model is `local3.8` (Qwen3.8-27B AWQ INT4), with a 220,000-token context, a 30,720-token output limit, and thinking enabled through Qwen's chat template. The SGLang endpoint stays on loopback; WebUI clients talk to nanobot, not directly to SGLang.
- The host launcher defaults `NANOBOT_STREAM_IDLE_TIMEOUT_S` and `NANOBOT_OPENAI_COMPAT_TIMEOUT_S` to 900 seconds. These are gateway-wide settings, not Qwen-only. Restart the installed gateway from the home directory, not this source checkout: starting it from the checkout without built WebUI assets previously caused the UI to return HTTP 404.
