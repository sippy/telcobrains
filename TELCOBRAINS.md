# TelcoBrains nanobot customizations

The `telcobrain-production` branch is based on upstream nanobot `v0.4.2` (`0dc8679ed`). This file records the branch changes and the separate host configuration used to run them; `docs/` is the canonical documentation for each feature.

## Recommended patch stack

The reviewed tip `f806542b4` has 18 commits above the vendor tag. They can be repacked into eight commits, in the order below, without changing the tracked tree. This is a proposed grouping; `telcobrain-production` has not been rewritten. The pre-update tree was reproduced in a temporary Git index and compared byte-for-byte at the Git tree level.

| Proposed commit | Reviewed commits | Rationale |
| --- | --- | --- |
| Native subagent model presets and follow-up | `9bce21e1b`, `a84facdd9` | Keep preset selection, saved-history follow-up, and iteration-limit behavior together. |
| OpenClaw conversation importer | `94ce87461` | Standalone migration utility. |
| Session compaction routing, retention, and dry run | `8a77f4876`, `960f9546a`, `c1d60704d`, `24651d2b6`, `60a167f2c` | Keep cut selection, tool-safe replay, staged summarization, dry-run parity, and reporting together. |
| Portable SUPERTURN workspace skill | `609a08eb3` | Independently installable workspace skill. |
| WebUI live-tail follow | `8a151bddd` | Independent UI behavior fix; keep separately revertible. |
| TelcoBrains WebUI branding | `43ac12996`, `330f98416` | Keep the icon set and its favicon contrast correction together. |
| Opt-in Claude Code CLI provider | `f2508b78d`, `e118335ca`, `6b42a03f4`, `8d84b8c12`, `f806542b4` | Keep the connector, preset setup, error display, and native-session recovery together; the fixes depend on the connector. |
| TelcoBrains branch and deployment notes | `ae2ada413` and later edits to this file | Keep this cross-feature document last, including the edits originally made alongside the skill. |

## Native subagents can use a different model

The `subagent` tool accepts an optional `model_preset` on `create` (for example, `local-qwen-sglang`); the child uses that preset while the parent keeps its selected model, and omitting it preserves inheritance. Unknown presets return the list of available presets; inline and background runs both honor the preset. See [Subagents](docs/concepts.md#subagents).

## Follow up on completed, cancelled, or iteration-limited subagents

The `subagent` tool accepts `action: "follow_up"` with a completed, cancelled, or iteration-limit-stopped task's `task_id` and a new `message`. It starts a new linked subtask that replays the saved history and tool results without rerunning old tool calls; each new run starts with a fresh full iteration budget counted from its own start, and it works after a restart while the history is saved. Interrupted and errored tasks cannot be followed up. Use `send` for queued or running tasks; wait for cancellation to finish before following up. Cancelled tasks retain only saved history. The original named preset is reused unless overridden, and only the owning persistent conversation can follow up. Follow-up does not resume the old execution or restore a process or its in-memory state. See [Subagents](docs/concepts.md#subagents).

If an inline run in a persistent conversation hits the limit, its result includes the task ID to use for `follow_up`. If the original run has no saved model preset, pass `model_preset` explicitly.

## Import OpenClaw conversations

Run the importer from this checkout with the host's nanobot Python environment; preview with `--dry-run` first, then rerun without it. Refresh the WebUI and look for the topic marked **(OpenClaw)**. It copies visible text and timestamps, not attachments, tool activity, or runtime state; reruns skip duplicates. See [Import an OpenClaw conversation](docs/cli-reference.md#import-an-openclaw-conversation) for alternate source databases and destination workspaces.

## Consolidate with a dedicated compactor model

`agents.defaults.compactor.modelOverride` selects a dedicated model for context compaction (including subagent contexts), manual `/compact`, SDK session compaction, and idle archival. It changes summarization, not the model doing the task. Set it to a named `modelPresets` entry, with the endpoint configured on that preset's provider; the implicit `default` is rejected. The preset's `maxTokens` and `contextWindowTokens` bound the summarization call and cap the checkpoint. Current exception: `/new` background archival still uses the session model.

Foreground context-pressure compaction, idle archival, manual `/compact`, and SDK session compaction can summarize an older prefix and keep a recent tail verbatim. Idle, manual, and SDK compaction default to a 90,000-token recent-history floor (`minVerbatimTailTokens`); with a positive floor they skip when no legal cut passes both budgets. Setting the floor to `0` enables the legacy full-prefix fallback. Foreground pressure instead prioritizes fitting the current request. Idle archival does not compact the same retained tail repeatedly without new activity.

Session compaction can cut mid-turn without splitting tool calls from their results, summarize an oversized prefix in bounded stages, and probe up to three qualifying cuts if replay exceeds its budget. `/compact --dry-run` runs the same compaction without applying it; it can still call the model. See [Compactor](docs/configuration.md#compactor) and [Memory](docs/memory.md).

`modelOverride` is also settable from the WebUI under Settings → System → Sessions: a preset picker whose options are the named `modelPresets` entries; leaving it unset keeps the session model. Existing validation (unknown and implicit `default` presets rejected) and the no-tools compactor routing are unchanged.

`agents.defaults.compactor.nextTurnInputReserveTokens` (default `0`, also in WebUI Settings → System → Sessions) reserves that many foreground input tokens for the next turn's fresh input when idle, manual `/compact`, or SDK session compaction sizes the retained verbatim tail, including the projected post-compaction probe. It never reduces compactor capacity or applies to in-turn compaction, which already sizes against the real pending input. It is opt-in: a larger reserve can keep less recent history verbatim, subject to the floor and the skip behavior above.

Limitations: if the input does not fit the compactor's context window or the call fails, nanobot writes a raw (unsummarized) checkpoint instead, with no retry. If the compactor is a different provider than the session model, provider-native compaction summarizes the materialized transcript instead of resuming the foreign provider's private state. This is model routing and replay sizing only. See [Compactor](docs/configuration.md#compactor).

## SUPERTURN workspace skill

The portable [SUPERTURN skill](skills/telcobrains/superturn/SKILL.md) is adapted from the OpenClaw copy for bounded assignments with Git-backed evidence and controller review. Copy the entire `skills/telcobrains/superturn/` directory to `<workspace>/skills/superturn/` so nanobot discovers it alongside other workspace skills and can still resolve its references and template. The repository copy is not a built-in skill. Installing it does not start workers or create scheduled jobs.

## WebUI live-tail follow and branding

Returning explicitly to the latest message re-engages live follow during a turn; scrolling away still disengages it. The branch also ships TelcoBrains welcome marks and app icons, including a higher-contrast favicon.

## Claude Code CLI models

The opt-in [Claude Code CLI provider](docs/claude-cli.md) uses the host Claude Code login, not an API key or URL. It resumes native sessions bound to the account, organization, and workspace without replaying the transcript. Tools are off by default; the only supported opt-in is workspace-scoped `read_file` through Nanobot's turn dispatch. The WebUI can show validated quota reset details. After a structured usage or rate-limit rejection, a new message can resume only when the native history proves the refused input was already recorded. Billing failures and the WebUI Retry button are not covered by that recovery guarantee.

## Local deployment settings (outside Git)

- The host sets `agents.defaults.maxToolIterations` to `1000` in `~/.nanobot/config.json`. This sets the ordinary budget for the main agent and each subagent run, including follow-ups. An iteration is a model round and may include multiple tool calls. The upstream default remains `200`.
- The host has opt-in `claude-haiku`, `claude-sonnet-5-5`, and `claude-fable-5-2` presets. The shared Claude CLI completion timeout is 180 seconds, tools remain off, and `local-qwen-sglang` stays the default.
- The LAN WebUI listens at `http://192.168.23.109:8765`, with token authentication enabled. The password remains in the host's private nanobot config, not in this repository. The endpoint is HTTP and should only be used on a trusted LAN.
- The `local-qwen-sglang` preset points to provider `local-a40` at `http://127.0.0.1:8081/v1`. The served model is `local3.8` (Qwen3.8-27B AWQ INT4), with a 220,000-token context, a 30,720-token output limit, and thinking enabled through Qwen's chat template. The SGLang endpoint stays on loopback; WebUI clients talk to nanobot, not directly to SGLang.
- The host launcher defaults `NANOBOT_STREAM_IDLE_TIMEOUT_S` and `NANOBOT_OPENAI_COMPAT_TIMEOUT_S` to 900 seconds. These are gateway-wide settings, not Qwen-only. Restart the installed gateway from the home directory, not this source checkout: starting it from the checkout without built WebUI assets previously caused the UI to return HTTP 404.
