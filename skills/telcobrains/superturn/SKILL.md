---
name: superturn
description: "Plan or run SUPERTURN checkpoint/review cycles for bounded team work, with explicit workspace, worker, controller and reporting bindings."
license: MIT
---

# SUPERTURN

## 1. Select the requested mode

- **Explain:** describe or evaluate the workflow; do not create workers or jobs.
- **Prepare:** write a project-owned run configuration and concrete launch plan; do not start task execution or enable controller jobs.
- **Run:** execute the user's authorized task through bounded assignments, checkpointing, verified review and explicit continuation.

Treat “use SUPERTURN for this task” as Run when a task and authorization are already present. Treat “explain,” “evaluate,” and “prepare” as their respective modes. Installing or reading this skill never authorizes a launch. Preserve existing scope and authorization across turns; do not request approval at every review gate or infer authority for extra agents, spending or external actions.

Done when the selected mode and intended outcome are clear from the request.

## 2. Bind the run to the requesting context

For Prepare or Run, read [the protocol](references/protocol.md), then copy [the run template](assets/RUN.example.yaml) into the target project's run root. Resolve relative skill resource paths against this skill directory, not the shell's inherited cwd. Store all live configuration and evidence outside the installed skill.

Discover and record these independently from live runtime context:

1. Actual host/environment, absolute project root and checkout/worktree.
2. Exact controller identity and supported scheduling target, if scheduling.
3. Worker runtime family, placement and returned worker/session identity.
4. Worker-to-controller completion route.
5. Controller-to-requester conversation, channel/account and thread where relevant.

Default to local-only subagents. Do not spawn, schedule, resume or migrate a worker on remote SSH hosts, cloud profiles, paired nodes or other non-local resources without an explicit user instruction covering that task and resource. Verify locality against the current local workspace execution environment; reachable/configured resources and general delegation authority are not grants. Record the instruction and allowed resource IDs before any non-local dispatch; if locality or authorization is unknown, block dispatch rather than guess.

Bind the requested toolkit independently from placement and model route. Honor an explicitly requested toolkit; do not substitute another runtime. A toolkit name alone does not establish locality. Do not fall back from a local model route to cloud inference or reuse the parent's account for children without authorization. Propagate these restrictions to worker briefs, continuations, controller jobs, containers/tests/benchmarks and descendants.

Never inherit another run's `main` session, channel, remote host, resource assignments or paths. A report destination does not choose execution placement. Do not invent spawn parameters or cross native-child/session runtime families. If live identity or tool scope cannot be verified, leave the affected operation unbound and identify the missing information; do not create a replacement worker or automation merely because an existing one is invisible.

Before Run GO, bind a user-approved project disk budget in exact bytes, a measured accounting scope/baseline and explicit temporary headroom. Missing allocation blocks GO. Follow [the disk-budget gate](references/controller-hygiene.md#project-disk-budget-hard-cleanup-gate); this is procedural enforcement, not a provisioned filesystem quota.

Choose one worker unless parallel lanes are authorized. Record an objective, acceptance checks, scope/exclusions, resources, cadence and an overall stopping condition. Use progress-review mode by default: review each active lane every 30 minutes, without automatically ending its turn. Alternatively assign a fixed duration or absolute deadline. Do not use fixed-grid turnover unless the user requests it. Read [shared Git turn storage](references/git-turns.md) before preparing a run; the controller maintains one shared local repository with a fresh lane/turn branch and worktree from current mainline at every turn start.

Done when Prepare has produced a reviewable manifest/launch plan, or Run has resolved the required execution bindings without placeholders. The template is contract data, not an executable API payload or an implemented schema validator.

## 3. Verify placement before substantive work

Before issuing worker GO for a scheduled run, complete the [controller automation preflight](references/scheduling.md). Bind every model-backed wrap/review job to an authorized inference route using the scheduler’s supported mechanism, read back the effective binding, and verify a harmless real run. The parent's selected model and the notification's channel account are not automation inference authorization. Keep scheduling UNVERIFIED/BLOCKED until these checks pass; do not describe a created or queued job as a working backstop. For manual runs, record that no automatic backstop exists instead. On nanobot, see the [runtime notes](references/nanobot.md) before dispatch.

Start an acknowledgement-only worker task using the bound runtime. Include run, lane, turn and generation IDs; absolute roots; host; controller/return route; scope/resource grants; local-only policy or explicit non-local user grant; required toolkit/model route; repository, base commit, lane/turn branch and worktree; report path; time-control mode and next review/deadline; and no-self-continuation. Check dispatch placement against the grant before even this initial spawn.

Compare its acknowledgement with actual spawn receipts/runtime metadata. Do not accept a worker's claimed channel as sole proof of placement. When the runtime cannot pause at startup, let this initial task end and use its supported follow-up operation to issue GO after validation. Use explicit absolute paths when cwd cannot be set; do not claim that prompt paths create a sandbox.

Done when receipt and acknowledgement match the run binding. Otherwise stop before substantive execution and correct placement within existing authority. If unauthorized work already started, stop only affected owned activity, preserve its checkpoint and verify cleanup before an authorized relaunch. Do not treat visibility or relay errors as permission for replacement runners or resources.

## 4. Execute bounded assignments

Record current lane state, dispatch receipt, assignment identity and resource ownership in a controller-owned registry under the run root. Keep historical transitions in an append-only log, not mixed into current assignments. Reconcile uncertain dispatch before retrying to avoid duplicate workers.

Give each assignment exact inputs, measurable acceptance checks, resources, restrictions, applicable model/tool budgets and its time-control policy:

- **Progress review (default):** schedule a controller check in 30 minutes; inspect committed progress/activity and choose CONTINUE, WRAP, PIVOT or BLOCK. A check is not a wrap, does not increment the turn, and does not restart the worker. Record a reason and the next check if continuing. Honor overall limits.
- **Deadline:** give a duration from actual dispatch or an absolute end time, plus a suitable wrap lead. At wrap start checkpointing; stop by the deadline.

Use assignment IDs plus due-time/review sequence to reject stale check events. Do not force productive lanes to stop just because a review slot arrived. For a material pivot, wrap and commit the current turn, review/merge/tag it, then create the next turn's branch/worktree with an updated brief.

Use supported scheduler/event continuations rather than sleep/poll loops. Bind scheduled execution to the controller explicitly and verify its real tool/model scope and recorded inference account. Recheck readiness after changing the execution agent, session target, model, account or runtime. If a scheduler cannot address the worker, use a supported wake to its owning controller or report the limitation; do not substitute another runtime. Reject stale wrap/review events by assignment identity and due-time/check sequence. Describe prompt deadlines as cooperative unless an actual enforcement mechanism exists.

Recover missed notices using [the scheduling recovery procedure](references/scheduling.md). Send an overdue wrap for the current assignment when needed, but do not treat that manual steer as repair of future notifications. Repair/read back the same job, preserve cadence and holds, and verify execution before restoring reliance on it. If caller scope prevents repair, record the exact blocker and fallback; do not silently leave a disabled job as a functioning scheduler.

At a controller wrap/pivot notice, early completion or assigned deadline, stop starting work, commit the report/evidence and quiesce owned activity. In deadline mode end by the assigned deadline. In progress-review mode the review time is not a stop signal. Mark unreachable/unchecked resource state UNKNOWN and block conflicting work; do not assume a stopped agent cleaned up remote jobs. Respect holds, parked/retired lanes and overall turn/time/spend limits.

Done when the worker has reported and stopped, or a partial/blocker and owned resource state have been captured for controller recovery.

## 5. Verify evidence, review, and continue explicitly

At every cleanup checkpoint and final closeout, measure the entire project footprint and require usage at or below its base disk budget. Repeat safe cleanup passes while useful, including the required oldest-evidence compression/archival and largest-consumer value review. Exhaust applicable safe reclamation options or document concrete blockers before escalating. If still over budget or measurement is incomplete, persist a disk-budget hold and await the user raising the limit or directing further cleanup. No successor creation or substantive work while held. Temporary headroom is allowed only between cleanups, never as a permanent increase.

Before successor creation/dispatch, apply the [five-ST controller audit and all-resource cleanup gate](references/controller-hygiene.md): ns > 1 && ns % 5 == 1. Reconcile controller state and every allocated task-owned resource, remove obsolete residue, and consolidate HEAD while preserving historical retrieval. Use an authorized scoped janitor lane for large inventories; controller review remains mandatory.

Follow [the Git handoff procedure](references/git-turns.md), including its mandatory closeout checklist recorded in every controller review. Merge/tag is not done: finish post-test ignored-cache recovery, worktree retirement and absence checks BEFORE creating the successor branch/worktree (not merely before dispatch). A real cleanup blocker must be owned and resolved; it does not waive this gate. Each lane writes only its assigned worktree and commits all task evidence, logs, artifacts and report under a lane/turn-scoped evidence directory. Make useful progress commits so other lanes can inspect snapshots before completion. Do not modify another lane's worktree, mainline checkout, Git configuration, historical evidence paths or published refs. New turns start from current integrated mainline; existing worktrees do not automatically receive other lanes' new commits.

Have the worker write `SUPERTURN.md` with assignment identity, base revision, work performed, claim/evidence/verdicts, reproducible checks, cleanup status, blockers and proposed next task. Include it in the final commit, and return the final commit ID in the completion message (not a self-referential report hash). Use commit/tag + repository-relative paths instead of mandatory per-file checksum manifests for tracked bytes. Preserve content hashes only for external payloads that Git does not actually store. Never claim a symlink or LFS pointer alone preserves the target bytes.

The controller reviews the exact submitted commit, merges the turn branch into mainline, publishes a unique annotated turn tag pointing to the submitted commit and records the merge/review commit. Keep tags permanent and do not rewrite published history. Remove the worktree only after clean/ignored-file, owned-job and tag/retrieval checks pass. Preserve partial or rejected evidence too; never silently discard it just because code cannot yet be integrated. Conflicts block finalization until resolved without overwriting prior evidence. Only then start the next turn from the new mainline; rebind a continuing worker to its new path.

Verify files, integrity and decisive raw evidence; replay meaningful cheap checks and state what was not independently checked. Record a durable review with accepted/partial/rejected findings and CONTINUE/PARK/BLOCK/RETIRE. Deduplicate completion/timer reviews and next dispatch by assignment identity. Continue the same bound worker through its supported follow-up operation; validate placement again for an explicit replacement. Workers never start their next turn themselves.

Publish one logical owner update through the current runtime's delivery contract. Do not combine a completed explicit send with an automatic final reply. Track uncertain delivery separately from worker completion, and do not blindly resend. Park when the goal is met; continue autonomously only within existing authority and stopping limits.

Done when findings and next disposition are saved, owned resources are accounted for, the report route has been checked, and no unintended worker/job continues.

## Limits

This is a procedural skill, not a controller service, filesystem sandbox, hard-deadline watchdog or exactly-once transport. Before unattended use, pilot the runtime adapter on a disposable project through two assignments, checking placement, early completion, stale notices, cleanup, continuation and actual requester-visible reporting. Disclose untested adapter behavior.
