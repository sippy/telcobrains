# Shared Git repository and turn lifecycle

Use Git's existing [linked worktrees](https://git-scm.com/docs/git-worktree) and [annotated tags](https://git-scm.com/docs/git-tag), not per-turn clones or a new artifact service. Keep the repo local unless the user explicitly authorizes non-local resources or publication. Do not push anything merely to share lanes.

## Prepare once: controller

Inspect existing repositories, dirty state, identity, ignore/attribute rules and worktrees first. Reuse an appropriate project repo or initialize a dedicated local run repo with an initial commit; never silently capture unrelated files from the entire workspace. A dedicated research repo may vendor authorized source inputs as tracked files; record provenance. No nested untracked repo may stand in for preserved source bytes.

Record the absolute repo, controller checkout, mainline branch, external sibling worktree root and run namespace. Only the controller changes mainline or creates and retires lane worktrees/tags. Keep controller live scheduling scratch outside committed lane evidence; commit assignments/reviews and integration outcomes.

Suggested names, validated with `git check-ref-format` before use:

```text
branch:   st/<run>/<lane>/ST<n>
worktree: <worktree-root>/<run>-<lane>-ST<n>
evidence: evidence/<run>/<lane>/ST<n>/
tag:      st/<run>/<lane>/ST<n>
review:   reviews/<run>/<lane>/ST<n>.md
```

Branch and worktree names must contain lane name and turn number; the run ID prevents collisions. Do not reuse another lane's branch or worktree. Worktree roots must be outside the controller checkout to avoid recursively adding them.

## Start each turn: controller

If the project has adopted a guarded handoff tool, use its reviewed operation and require both a successful exit and a completed retirement/creation receipt before placement acknowledgement. Never bypass a failed guard with manual Git commands. Verify any project-specific tooling before relying on it.

FIRST complete the predecessor closeout checklist, including worktree removal and verified absence from both disk and git worktree list. This is a hard gate BEFORE creating the successor branch/worktree, not merely before worker GO. If predecessor cleanup is blocked, do not create its successor worktree; resolve retention/cache recovery first. Other independently authorized active lanes are not predecessors and must remain untouched.

Resolve mainline to an exact base commit after integrating completed turns. Create a fresh branch and linked worktree; do not reset or repurpose the previous turn's branch. With variables populated from the validated run binding:

```bash
git -C "$st_repo" worktree add -b "$st_branch" "$st_worktree" "$st_base_commit"
git -C "$st_worktree" rev-parse --show-toplevel HEAD
git -C "$st_worktree" branch --show-current
```

Give the worker the exact repo, base commit, branch, worktree, evidence directory, timing policy and required report path. Verify its cwd/branch/receipt before GO. When continuing an existing worker session, explicitly replace its old cwd and input paths; never let it recreate the removed previous worktree by accident.

## Stale-run isolation (correctness gate, not housekeeping)

Retired worktrees, import paths, build outputs and logs can produce false passes or false failures. Cleanup protects reproducibility, not just disk space.

- Before a test/build, resolve the active worktree, branch, base and assignment; bind cwd, source/import paths, mounts, output and counter/log paths explicitly. Never select a sibling turn by glob, newest mtime, or an old absolute path.
- Historical files are inputs only when explicitly selected by commit/tag and recorded as reused provenance. An inherited old PASS log is not a new result.
- Record the tested source revision (and dirty diff if any), loaded module paths, image identity and exact command with each validation result. Check actual imports/mounts resolve to the intended snapshot, not just the command cwd.
- Fresh validation outputs must be current-assignment scoped and not silently reused. Wrong/stale source, output or run identity INVALIDATES the result; correct the binding and rerun before making pass/fail claims.
- On handoff, remove retired worktrees from active search/import/build paths. Recovery-cache archives are not runnable inputs or an alternate source tree.

## During work: lane

Work only in the assigned worktree. Write new logs, raw results, test scripts, source/config changes and artifacts there. Commit useful progress snapshots; relay commit IDs and important paths to the controller for cross-lane sharing. All worktrees share committed objects/refs, but not working directory updates. Inspect integrated or in-progress evidence read-only:

```bash
git -C "$st_worktree" show "$st_other_commit:$st_relative_path"
```

Cite exact commits/tags, not a moving branch name. Other active lanes do not automatically acquire new mainline changes. Read with `git show`; coordinate any code adoption rather than merging/rebasing behind the controller's back. At the next turn, the new worktree starts from current mainline and sees integrated prior turns/lanes. Uncommitted cross-lane evidence is not a stable reference.

Do not amend/rebase/reset published commits, move/delete published tags or touch another active worktree's files. Add corrections as new turn-scoped files/commits. Only the controller-reviewed periodic hygiene gate may remove duplicate, superseded or irrelevant historical evidence paths from current mainline via normal cleanup commits, after verifying retained commit/tag retrieval and recording the old paths in a canonical index. Historical bytes/refs stay intact; workers must not quietly remove logs. See [controller hygiene](controller-hygiene.md).

At wrap, stop owned writers, flush logs, write `SUPERTURN.md`, and commit all task artifacts and evidence, including the report. Check ignore rules explicitly: logs/binaries may otherwise be omitted. Add only known task paths; intentionally include ignored evidence with an exact `git add -f -- <path>` when appropriate. Never blanket-add credentials, caches or unrelated files. Required raw logs and artifacts must be preserved, not dropped merely because they are large/ignored. Compress exact bytes if useful. For payloads truly unsuitable for Git, retain a durable local content-addressed copy and commit its reference/size/digest. Report this exception and prove retention; no sole copy in temporary/remote storage.

Return the final commit ID, branch, report path, resource cleanup state and remaining blockers. A clean working tree alone is not proof ignored/untracked evidence was captured. Inventory expected outputs and compare committed paths.

## Finish each turn: controller

### Mandatory closeout checklist (record in the controller review)

Do not mark closeout complete or create a successor branch/worktree until each item is checked, or explicitly BLOCKED with its exact path, reason and owned recovery step. A successful merge/tag alone is not closeout.

- [ ] Worker and owned writers quiescent; exact submitted branch/commit verified.
- [ ] Report and expected raw evidence reviewed; checks and verdict recorded.
- [ ] Mainline integration (or evidence-only exception) and permanent annotated tag verified; use explicit refs/heads/ and refs/tags/ to avoid ambiguity.
- [ ] Report AND representative raw evidence retrieved from the tag.
- [ ] Tracked, untracked AND ignored inventories checked after parent tests.
- [ ] Disposable caches recovered outside the worktree (preserve relative paths); real evidence committed or durably retained, never classified as cache merely because ignored. No force removal or blanket git clean.
- [ ] Clean worktree removed without --force; filesystem absence AND absence from git worktree list verified. Keep turn branch and tag.
- [ ] Retirement recorded with old path, retained refs, mainline HEAD and any recovered-cache location; current assignment index reconciled.

Parent validation should use PYTHONDONTWRITEBYTECODE=1 / python -B and read-only mounts where possible. Tests can create ignored __pycache__ even after the worker reports a clean tree: recheck AFTER replay. Resolve recoverable cache cleanup in the same review turn; do not leave a known cleanup guard as an unowned TODO or claim the worktree was collected before removal succeeds. If cleanup genuinely blocks, retain the worktree and arrange a concrete completion path before ending. A cleanup blocker blocks successor worktree creation; recording the blocker does not waive this ordering gate.


1. Reconcile worker ownership/processes; no worker or owned job may still write into a worktree being integrated/retired. Read the report from the exact submitted commit and verify expected evidence exists in that tree.
2. Review claims with raw evidence and decisive checks. The code's acceptance and the run's evidence retention are separate decisions. Preserve failed/partial results; do not silently discard them.
3. Confirm the lane branch still resolves to the submitted commit. On the clean controller mainline checkout, merge the branch without squashing so its history remains reachable. Serialize merges and resolve conflicts without overwriting another turn's evidence. If implementation conflicts or is unsafe to integrate, retain the submitted commit with its permanent turn tag and integrate its evidence/report only with a controller review explaining the exception; mark code integration pending/rejected. Never use an `ours` merge to pretend omitted evidence was integrated. Do not remove its worktree until evidence-only integration has been verified.
4. Create a unique annotated turn tag pointing to the submitted commit, then commit the review on mainline recording submitted commit, tag, base, merge commit (or exception), validation and disposition. On interrupted recovery, reuse a tag only if it already points to the exact expected commit. Never force-update a conflicting tag; report the discrepancy. A tag means retained evidence, not that every claim/code change was accepted.
5. Prove retrieval with `git show <tag>:<report-path>` and representative raw evidence paths. Inspect both `git status --porcelain=v1 --untracked-files=all` and `git ls-files --others --ignored --exclude-standard` in the worktree. Preserve missing task outputs first. Leave unresolved/dirty worktrees intact.
6. Remove only the verified quiescent clean worktree using `git worktree remove` without `--force`. If ignored disposable caches prevent removal, retain the worktree until recoverable cleanup is complete; do not override the guard. Keep the published tag and turn branch; never delete reachable evidence refs as part of cleanup. Record retired path and mainline head. The next turn gets a new branch/worktree from that integrated head.

Typical merge/tag/retrieval commands (after the checks above):

```bash
git -C "$st_repo" merge --no-ff "$st_branch" -m "$st_merge_message"
git -C "$st_repo" tag -a "$st_tag" "$st_submitted_commit" -m "$st_tag_message"
git -C "$st_repo" show "$st_tag:$st_report_path"
git -C "$st_repo" worktree remove "$st_worktree"
```

No routine per-file checksums are needed for bytes actually stored in Git; cite commit/tag and path instead. A tag can be moved and shared `.git` storage can be damaged: prohibit destructive ref/history operations, retain refs and use an authorized durable local backup/snapshot when available. This is a preservation workflow, not an access-control boundary. LFS pointers, symlinks and submodule gitlinks require separate payload retention checks.
