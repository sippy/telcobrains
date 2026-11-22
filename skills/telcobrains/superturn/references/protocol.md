# SUPERTURN portable protocol — v0.2

## 1. Scope and activation

A super-turn is a bounded assignment ending with durable evidence and a review gate. The worker stops before another assignment; only its bound controller may continue it. Completion of the goal parks the lane rather than inventing work.

Reading, linking, reviewing or installing this protocol does not launch anything. Start only on a user's run request or an existing explicit run authorization. Parallel lanes, additional agents, spending and outbound actions remain subject to the caller's authority; the protocol grants none of them itself.

Normative words MUST/SHOULD describe this version's contract, not permissions to override system instructions, workspace rules or runtime policy.

## 2. Bind the instance before executing

Create a project-owned run manifest from ../assets/RUN.example.yaml. Resolve all required fields against the current environment, not against the protocol's location.

Keep these independent:

- **Workspace:** repository/root, checkout/worktree, machine and evidence store.
- **Controller:** exact session/process identity and namespace that owns the run.
- **Worker:** runtime family and returned worker/session handle for each lane.
- **Completion route:** worker-to-controller event/return mechanism.
- **Report route:** controller-to-requester conversation/channel/account/thread.

No default to `main`, “last channel,” a historical session key, the skill's own directory, or a remembered remote host. Read live context and existing state before mutating jobs or configuration. A non-applicable external route is explicitly null, not an invitation to guess a recipient.

Use `(run_id, lane_id, turn_number, generation)` as the assignment identity. Labels are for humans, not routing. A superseding assignment increments the generation. Boundaries and controller events carry this identity.

### Placement handshake

Local-only is the default resource policy. Before any spawn, scheduling, continuation or migration, verify that the worker's execution environment is the current local workspace environment, or record the user's explicit instruction authorizing that non-local task/resource. Remote host access, configured cloud profiles, a repository location or a general request for subagents is not such an instruction. Unknown locality blocks dispatch. Keep grants scoped to the specified task/resources and apply them to descendant workers and their tests, containers and benchmarks; never escalate indirectly through a local runner.

Bind toolkit/runtime, execution host and model/inference route independently. Honor any explicit dispatch-toolkit requirement. Runtime family alone does not establish resource locality. Do not substitute cloud workers/inference when the authorized local route is unavailable. Include exact allowed resources, toolkit and model routes in every worker brief, continuation and controller payload; reconcile receipts against them. Stop affected owned activity and preserve/checkpoint before correcting an unauthorized placement. Relay failures never authorize replacement workers.

Before substantive work, the worker acknowledges assignment identity, runtime handle if observable, actual absolute working/evidence roots, host identity, scope, resource grants, repository/base/branch/worktree, report path, time-control mode with next review or deadline, and no-self-continuation. The controller compares this with the manifest and the spawn receipt/live runtime metadata; a worker's self-description alone is insufficient evidence of channel placement. Some runtimes cannot observe a chat channel at all.

When a paused startup handshake is unsupported, use an initial acknowledgement- only task that ends, then an explicit follow-up to start work after validation. Do not idle-poll for a GO message. If placement cannot be validated, keep the lane unbound and report the missing evidence. Rebinding a worker is an explicit registry transition, never a silent fallback to another runtime.

## 3. Roles, authority and state

The **owner** sets goals/authority; the **controller** scopes, schedules, verifies and continues; the **worker** executes only its assignment. A human may fill any role. A second reviewer agent is optional, not inherent to this protocol.

Persist a small current registry and append-only event log under the run root. Only the controller writes controller state; workers write their lane evidence. Use one active controller writer or an implementation-level lock/CAS revision. Each lane records assignment identity, exact handle, status, resource ownership, deadline, last accepted report, review identity and pending dispatch receipt.

```text
UNBOUND → READY → STARTING → RUNNING → WRAPPING → REPORTED → REVIEWING
                        ↘ FAILED/BLOCKED             ↓
                                           READY | PARKED | BLOCKED | RETIRED
```

READY means authorized for a concrete next assignment, not permission for a worker to restart itself. PARKED requires a controller decision to resume within existing authority; RETIRED is omitted from recurring work. A run-level HOLD prevents new dispatch and governs draining even if a timer fires. Unknown owned resource state blocks conflicting dispatch regardless of lifecycle status.

Before dispatch, record its idempotency identity and intended worker. Record the returned receipt afterward. On timeout/crash in between, reconcile the runtime before retrying. Never equate “no visible result” with “no worker exists.”

## 4. Time and stopping

Select a time-control policy per lane; honor explicit user choices:

- **Progress review (default):** check each active lane every 30 minutes. The controller inspects new commits, report/checkpoint and actual activity, then records CONTINUE, WRAP, PIVOT or BLOCK with a reason and next due time. A review does not automatically stop/restart the lane or create a new turn. A lack of final report at a progress check is not an overdue wrap. Make progress commits to share evidence; do not interrupt productive work for a status ceremony.
- **Deadline:** specify either a duration from actual assignment dispatch or an absolute UTC end. Give a suitable wrap lead and cleanup/review grace. Begin safe wrap before the end and stop new work at the deadline. Review early completions immediately; the successor gets its own new deadline.

Progress-review mode still obeys the run's total time, cost, turn and scope limits. A wrap notice states a concrete safe-stop target; a material pivot wraps and commits the current turn before a new branch/worktree/brief is issued.

Use supported automation wakeups, not worker sleeps/polling. Each event carries run/lane/turn/generation and a check sequence or intended due time. Deduplicate checks; ignore superseded events instead of wrapping a successor turn. A late review inspects current progress once and schedules the next check from the actual review time; do not burst through missed slots or shift a deadline.

Fixed-grid scheduling is opt-in only. When explicitly using a grid, record its absolute boundaries and exemption rules in the run configuration. Adopting this protocol does not authorize migrating live jobs or worktrees.

The controller checks expected next progress review or deadline wrap/review slots in authorized normal turns. Restricted scheduled turns operate within actual API scope and report limitations, not rebuild invisible jobs. Keep scheduler acceptance, inference execution, delivery and worker consumption as distinct observations.

Before relying on any model-backed controller automation, follow [the scheduling preflight and recovery procedure](scheduling.md). Explicitly verify its authorized inference route through the scheduler’s supported binding and readback, then verify harmless real execution before issuing scheduled worker GO. Neither parent-session settings nor messaging-account selection proves job inference authorization. A manual overdue wrap does not repair future notices; repair and verify the existing jobs, respecting holds and absolute due times.

Prompt timing is cooperative. A hard runtime timeout can cut off reporting or leave remote processes alive. Configure a separate cleanup/watchdog mechanism where a hard limit is required, and record what it actually guarantees.

## 5. Workspace, evidence and resource isolation

### Required project disk allocation

Before substantive GO, record a user-approved project disk limit in exact bytes, accounting scope and measured baseline, plus bounded temporary headroom. Missing allocation blocks GO. The budget covers all project lanes and persists across turns/restarts. Follow [the disk-budget gate](controller-hygiene.md#project-disk-budget-hard-cleanup-gate). Slight approved overrun is allowed only between cleanups; each cleanup checkpoint and final closeout must bring measured usage at or below the base allocation. Actively archive/compress cold older evidence losslessly and inspect the largest artifacts for retention value, redundancy and reproducibility. Verify retrieval and net savings; exhaust applicable safe reclamation options or document concrete blockers before requesting more space. Repeat safe cleanup passes where useful. If compliance cannot be achieved or measurement is incomplete, persist BLOCKED_DISK_BUDGET, stop new substantive work/successor creation, and wait for the user to raise the allocation or direct further cleanup. Retention and ownership safeguards still apply. No automatic budget increase or off-accounting relocation. This policy installs no filesystem quota and does not itself allocate storage.


Use one controller-maintained shared local Git repository for the run. Create a fresh lane/turn branch and linked worktree from the latest integrated mainline at every turn start. Read [the Git lifecycle](git-turns.md) for exact commands, commit/tag conventions, sharing and safe worktree retirement.

Commit all evidence/logs/artifacts and the worker report from that worktree. Use lane/turn-specific evidence paths to avoid cross-lane overwrites. Prefer Git commit/tag references over repetitive per-file checksum manifests for tracked bytes. Historical evidence is retrieved by permanent turn tag and relative path; new mainline content does not change the tagged snapshot. Git refs are mutable: prohibit tag replacement, published-history rewriting and deleting retained refs. A shared writable repo is not a security boundary or a backup.

Every new turn sees all integrated prior turns/lanes through its mainline base. Active worktrees remain pinned: use read-only `git show` on another lane's progress commit or a controller-relayed reference; do not merge live changes into a working lane without coordinating. Never read another lane's uncommitted files as authoritative evidence or remove artifacts before committing them.

For unavoidable external large payloads, commit a durable retrieval reference, size and content digest; verify payload retention before cleanup. Symlinks, LFS pointers and submodule IDs do not by themselves store payload/source bytes. Normal logs and evidence belong in Git, not disposable external paths.

Resolve mutation roots absolutely including symlinks. Worker writes stay in its assigned current worktree; the controller owns mainline, reviews, tags, worktree creation/removal and state. Finalize the report in the submitted commit; cite the final commit in the completion message so there is no self-hash cycle.

Scope resources by stable identity (host/device/service/port/branch), not display indices. Record exclusivity and owned process/job handles. Safe stop means the worker's owned activity is finished/cancelled and checked, not that every shared machine is idle. Do not kill other teams' processes. Persistent services require an explicit handoff owner and stop policy; otherwise clean them up.

If a host is unreachable or cleanup cannot be observed, mark resource state UNKNOWN and block conflicting work. Worker failure does not release resource ownership automatically. Respect existing exclusions/holds across restarts.

## 6. Report and review gate

Worker report (aim for one page, with detail in linked evidence):

1. Assignment identity; actual roots/host; start/end; completion/partial/blocker.
2. Objective and work performed; deviations from scope.
3. Claim/evidence/verdict table: proven, disproven or not proven, with scope.
4. Repository/base/branch and committed artifact paths; reproducible key checks. Return the final report-containing commit ID in the completion message.
5. Owned-resource cleanup: checked, handed off or UNKNOWN, with evidence.
6. Blockers, integrity incidents, owner/controller actions and proposed next task.

End the worker run with a short pointer to that report. A saved partial report is preferable to a promise. Do not claim completion without checking files.

Controller review MUST check assignment identity, report existence, evidence availability/integrity and resource state. Independently replay decisive cheap checks and examine raw evidence supporting important claims; disclose checks not performed. Do not rerun every expensive experiment without cause.

Record acceptance/partial/rejection, scope of verified claims, remaining gaps and one of CONTINUE/PARK/BLOCK/RETIRE. Finalize the reviewed branch through the merge/tag/retire lifecycle before a successor turn. A next assignment gives exact input paths, fresh branch/worktree/base, acceptance criteria, resource grants, restrictions, budget and time-control policy with next review or deadline. Continue the existing bound worker with its runtime's follow-up operation when supported. Fresh workers require a recorded replacement and placement handshake. Recovery from interruption may retain the same turn/branch before submission; published turn tags are never reopened or moved. A new assignment must explicitly rebind the continuing worker from the retired worktree to its new worktree.

Deduplicate review and continuation by assignment identity. A completion event and timer review racing must not produce two follow-ups. Controller review is not a request for fresh human permission every turn: continue within the owner's existing scope and budget. Escalate actual scope/authority gaps only.

## 7. Runtime adapter contract

An adapter supplies discover/bind, start, validate-placement, inspect, steer, follow-up, stop, completion, schedule/reconcile and owner-report operations. It documents unsupported operations rather than pretending every backend has them.

Use live tool schemas for operations and canonical identities; labels, model names and handles from another runtime are not interchangeable. If worker creation cannot set a working directory, use and verify explicit absolute paths, or require managed placement when actual isolation is needed. If a scheduled turn cannot address the owning controller's worker, use a supported wake to that controller or mark the operation unavailable. Do not substitute a new worker or infer a route from the reporting destination.

Owner reporting has one designated publisher, normally the controller. Workers report to it, not directly to chat recipients. Honor the current turn's live delivery contract: do not combine an explicit completed send with an automatic final reply. Persist a report identity and delivery outcome; reconcile uncertain sends rather than blindly retrying. Exactly-once transport is not guaranteed by this document. Verify real viewer delivery in a pilot.

Child completion batches may hold notifications until sibling lanes settle. Record that runtime behavior, avoid mixing very different horizons in a batch, and do not diagnose a delivery failure from a held notification alone. Use event continuations instead of polling loops. A worker's runtime-specific model/token budget belongs in its adapter/brief, rather than imposing one model family's limits on all workers.

## 8. Minimal invocation and worker brief

Invocation intent:

> Start a SUPERTURN run for this project using the current workspace and conversation as controller/report context. Goal: <goal>. Scope: <allowed work>. Use <one worker / explicitly authorized lanes>, <cadence>, and stop after <maximum turns, elapsed time, cost or goal>. Use progress reviews every 30 minutes unless a deadline is specified. Bind and validate placement before substantive work. Do not inherit settings from another run.

Worker brief, populated from the validated manifest:

> Assignment <run/lane/turn/generation>; controller <exact identity>; runtime <family>; work root <absolute>; evidence root <absolute>; host <identity>. First acknowledge placement only. After controller GO: achieve <objective>; acceptance <checks>; inputs <exact paths/revisions>; resources <grants>; prohibited <actions>; model/tool budget <limits>. Git <repo/base/branch/worktree>. Timing <progress-review + next check, or deadline + wrap time>; stop policy <policy>. Commit evidence and report at <repository-relative paths> and check owned-resource cleanup. Report only to the controller through <route>. End after reporting; do not continue yourself or spawn extra agents.

Install only a short workspace entry-point pointer if desired; keep runtime bindings in the project manifest. The portable protocol contains no live channel IDs, worker handles, machine addresses or credentials.
