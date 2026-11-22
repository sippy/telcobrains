# Controller automation preflight and recovery

Read this before enabling or relying on model-backed progress-review or deadline wrap/review automations, and when recovering a missed controller notification. Inspect current tool schemas/CLI help; use only supported fields and operations. These instructions do not authorize unrelated jobs, new accounts or extra inference spending.

## 1. Bind inference separately from execution and delivery

For every controller job, verify execution (the actual controller identity and session or equivalent target) separately from delivery (the requesting conversation, channel, account and thread as applicable). For model-backed jobs also record the authorized provider/model route and any required inference account reference. Use the runtime's supported binding mechanism: an explicit per-job selection where exposed, or a documented inherited route that can be verified for the job. Don't guess from a successful parent turn or a messaging account.

Inspect the effective binding through supported readback and execution evidence. If it cannot be verified, leave scheduling UNVERIFIED/BLOCKED. Do not add unsupported fields, edit internal state or credentials, or change job kind to bypass authorization. Record account references, not secrets. Jobs without inference still need execution, schedule, tool-scope and delivery verification.

## 2. Verify before relying on the scheduler

In an authorized normal controller turn:

1. Read every progress-review or deadline wrap/review job, including existing disabled ones. Verify identity, execution target, authorized inference binding, where applicable, allowed tools, enabled/disabled status, boundary/assignment guards, and notification route.
2. Match each job to the lane's selected time-control policy. Progress-review jobs inspect every 30 minutes by default and decide CONTINUE/WRAP/PIVOT/BLOCK; they must not automatically stop/restart a worker or increment its turn. Set the next check from actual review time. Deadline jobs wrap at `end - wrap_lead` and reconcile at `end + review_grace`. Record exact expected next slots and assignment/review sequence. Preserve grid anchors only for runs that explicitly selected them; never turn a review time into a stop deadline.
3. Verify a real inference execution using the same job's bound inference route, target and tool policy through a harmless preflight branch. That branch reports readiness only: no worker steering, continuation, resource operations or owner-report duplication. Use supported dry-run/test facilities if available; otherwise use a harmless test job with the same authorized path or a supported preflight mode. Do not temporarily mutate an immutable scheduled-continuation definition or trigger its integrity failure. After a permitted test, restore/read back the original payload and intended next slots before enabling production use. Do not force-run a production wrap/review payload as a test.
4. Inspect the completed run, not merely its creation/queue receipt. Verify that the expected execution ran without admission or inference errors and the bound controller/return path was reached. Record the test run ID and result. Check the actual first due production run for correct assignment, steering and delivery.
5. Track job saved, inference binding verified (if applicable), test executed, notice delivered and worker consumed as separate evidence. Acceptance of a steer does not prove the worker consumed it; use an acknowledgement or supported execution evidence.

Mark scheduling READY only after configuration/readback and real execution verification. If that cannot be done within existing authority, mark it BLOCKED/UNVERIFIED and use an explicit manual-controller plan or park the lane; do not dispatch unattended work that depends on an unverified stop notice.

## 3. Recover a missed or refused execution

1. Read the current assignment, timing mode and holds. A late progress check requires a progress assessment, not an automatic stop. If a deadline or an explicitly issued wrap is due, send the overdue checkpoint/stop notice for that exact assignment/boundary through its bound runtime. Record delivery/consumption evidence honestly. A manual wrap protects this assignment only; it does not fix the scheduler.
2. Inspect the failed job and other affected wrap/review jobs from the authorized controller. Repair the binding through supported operations, preserving IDs unless the scheduler explicitly requires replacement. Reconcile uncertain outcomes before retrying or replacing anything; do not create duplicates, alter delivery to repair inference, or bypass an integrity/admission check.
3. Read back the repaired binding, perform the harmless execution verification, restore intended payloads/timing, and re-enable jobs within the run's existing authorization when a failure disabled them. Respect explicit owner holds or deliberate disabling; do not override them. Reconcile to future valid slots rather than replaying stale wraps against a new turn.
4. Report separately: immediate worker stop status, inference repair status, schedule/enabled status, verified run IDs and any remaining limitations. Do not close recovery as complete while required future notices remain disabled or unverified.

Restricted scheduled recovery turns may lack inference configuration, job-update or cross-job access. Perform only the immediate safe work available, record the remaining repair and route it to the owning authorized controller/operator. Do not recreate invisible jobs or loop-retry an admission refusal. If no supported handoff is available, state the limitation and park dependent work. After a normal controller turn repairs it, clear the blocker only on evidence.
