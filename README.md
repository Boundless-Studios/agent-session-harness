# agent-session-harness

Crash-safe context accounting and fresh-session supervision for Claude Code and Codex.

The harness keeps long agent tasks moving without letting a single conversation grow indefinitely. It measures native usage, warns at 65% of the model window, drains at 70%, verifies a durable handoff, fences the predecessor, and launches a genuinely fresh successor. It never uses Claude `--continue`/`--resume` or Codex `resume --last` for an automatic rotation.

## Guarantees

- Claude assistant rows are deduplicated by stable API message ID. Missing IDs are deterministic but explicitly degraded.
- Codex forked rollouts subtract each child’s inherited pre-spawn baseline instead of charging copied history again.
- Live context and cumulative spend are separate values.
- Lifecycle hooks retain only identifiers, timestamps, normalized tool names, and working directories. Conversation bodies are not stored.
- Unknown context, stale/corrupt lifecycle state, an unverified required checkpoint, or unknown fencing state disables automatic termination.
- Required checkpoint stores are written and read back by exact capsule fingerprint before ownership is fenced.
- Rotation stops the predecessor before launching the successor, and coordinator lease epochs reject stale owners.
- Supervisor phases are persisted before external effects, completed effects are journaled, and process launches are keyed by chain and generation.

## Install

Version `v0.1.0` pins [`agent-coordinator` v0.2.0](https://github.com/Boundless-Studios/agent-coordinator/releases/tag/v0.2.0) by immutable Git tag.

```bash
python -m pip install \
  'agent-session-harness @ git+https://github.com/Boundless-Studios/agent-session-harness.git@v0.1.0'

agent-session-harness doctor --json
```

For development:

```bash
uv sync --extra dev
uv run pytest -q
```

## Inspect usage

Claude:

```bash
agent-session-harness inspect \
  --runtime claude \
  --path ~/.claude/projects/<project>/<conversation>.jsonl \
  --window-tokens 1000000 \
  --json
```

Codex, including a fork lineage:

```bash
agent-session-harness inspect \
  --runtime codex \
  --lineage ~/.codex/sessions/<root>.jsonl \
  --lineage ~/.codex/sessions/<child>.jsonl \
  --json
```

The Codex result reports both the naïve sum of cumulative snapshots and the corrected incremental total. A missing child baseline is `degraded`; it is never guessed.

## Configuration and observe-only mode

Configuration precedence is explicit path, project `.agent-session-harness.toml`, then the platform user config directory.

```toml
observe_only = false

[governor]
warn_percent = 65.0
rotate_percent = 70.0
stale_event_timeout_seconds = 30.0
```

Managed rotation is enabled only when the host declares every required capability known. Otherwise the loaded configuration is forced to observe-only. Preflight a host integration without starting a model:

```bash
agent-session-harness supervise \
  --runtime codex \
  --cwd "$PWD" \
  --chain-id <stable-chain-id> \
  --task-type linear \
  --task-id BOU-2195 \
  --task-fingerprint <stable-task-fingerprint> \
  --state .agent-session-harness/supervisor.json \
  --required-capabilities-known \
  --check --json
```

Run a managed fresh-only chain by supplying executable JSON adapters. Every argv value is a JSON string array; no value is evaluated by a shell:

```bash
agent-session-harness supervise \
  --runtime codex \
  --cwd "$PWD" \
  --chain-id <stable-chain-id> \
  --task-type linear \
  --task-id BOU-2195 \
  --task-fingerprint <stable-task-fingerprint> \
  --state .agent-session-harness/supervisor.json \
  --required-capabilities-known \
  --executable "$(command -v codex)" \
  --usage-adapter '["/absolute/path/to/usage-adapter"]' \
  --capsule-adapter '["/absolute/path/to/capsule-adapter"]' \
  --safety-adapter '["/absolute/path/to/project-safety-adapter"]' \
  --required-adapter 'beads=["/absolute/path/to/beads-adapter"]' \
  --mirror-adapter 'linear=["/absolute/path/to/linear-adapter"]' \
  --adapter-env linear=LINEAR_API_KEY \
  --json
```

The supervisor runs until interrupted. `--max-ticks` exists for bounded automation and tests. A terminal exit stops the managed child, fences its claim, and persists `blocked` rather than leaving an unowned agent running. `blocked` is terminal for that run specification: after resolving retained ownership, start a new chain and state path instead of reusing or resuming it.

The long-lived integration surface is `agent_session_harness.supervisor.Supervisor`. A host supplies four small protocols: native usage reader, checkpoint manager, fenced coordinator, and process driver. This keeps Linear, beads, PR dashboards, worktree launchers, and project safety policy outside the reusable package. The deterministic E2E test uses a real child process and proves root → checkpoint → fence → stop → fresh successor → acknowledgement with no overlap.

## Durable handoff capsule

`HandoffCapsule` contains only bounded operational state:

- task identifiers, objective, exact next action, and completed/remaining criteria;
- repository, branch, HEAD, dirty paths, and file/symbol anchors;
- test results, decisions, blockers, and allowlisted process summaries;
- predecessor conversation, target generation, creation time, and SHA-256 fingerprint.

Unknown fields are rejected. Required adapters receive canonical JSON on stdin and are invoked as an argv array, never through a shell:

```json
{
  "schema_version": 1,
  "operation": "write",
  "idempotency_key": "chain-1:1",
  "capsule": {"schema_version": 1, "fingerprint": "..."}
}
```

They return one bounded object:

```json
{
  "ok": true,
  "fingerprint": "...",
  "retryable": false,
  "error": null
}
```

Required adapters must pass `write` and independent `read` fingerprint checks. Mirror failures go to a locked `0600` outbox and can be retried later:

```bash
agent-session-harness outbox replay \
  --path .agent-session-harness/mirrors.jsonl \
  --adapter 'linear=["/absolute/path/to/linear-adapter"]' \
  --adapter-env linear=LINEAR_API_KEY \
  --json
```

Adapter subprocesses inherit only a controlled, non-secret base: user/path/locale values, XDG and beads paths, TLS certificate paths, `NO_PROXY`, and `SOPS_AGE_KEY_FILE`. Credential variables such as `LINEAR_API_KEY` and proxy credentials are absent unless named explicitly with repeatable `--adapter-env NAME=KEY`. Unrelated ambient secrets and prefix-matching variables are never copied.

### Usage and capsule adapter contracts

The usage adapter receives one bounded JSON request on stdin:

```json
{
  "schema_version": 1,
  "operation": "sample",
  "process": {
    "pid": 1234,
    "process_group_id": 1235,
    "registry_key": "chain-1:0",
    "identity": "...",
    "command_digest": "...",
    "launch_nonce": "..."
  }
}
```

It returns `conversation_id`, `context_percent`, and `confidence` (`confident`, `degraded`, or `unknown`), plus optional `context_tokens`, `window_tokens`, and `cumulative_tokens`:

```json
{
  "conversation_id": "native-conversation-id",
  "context_percent": 68.2,
  "confidence": "confident",
  "context_tokens": 136400,
  "window_tokens": 200000,
  "cumulative_tokens": 481000
}
```

The capsule adapter receives:

```json
{
  "schema_version": 1,
  "operation": "checkpoint",
  "checkpoint": {
    "chain_id": "chain-1",
    "predecessor_conversation_id": "native-conversation-id",
    "target_generation": 1,
    "idempotency_key": "chain-1:1"
  }
}
```

It returns either the complete `HandoffCapsule` or `{"capsule": <complete HandoffCapsule>}`. The harness validates the chain, predecessor, generation, bounded schema, credential exclusion, and canonical fingerprint before any durable adapter runs.

An optional project safety adapter receives `schema_version`, `operation: "probe"`, `cwd`, `chain_id`, and `generation`. It returns a bounded status plus named critical sections:

```json
{
  "schema_version": 1,
  "status": "busy",
  "active_critical_sections": ["git-index-lock", "deployment"],
  "warnings": []
}
```

Allowed statuses are `quiescent`, `busy`, and `unknown`. Project and native lifecycle state are merged before every governor tick; either source being `unknown` disables automatic rotation, while either source being `busy` keeps the predecessor running. Adapter failure is converted to `unknown` without persisting its diagnostic text.

All adapter commands are direct argv execution with a five-second default timeout. Usage/capsule stdout is capped at 1 MiB; checkpoint stdout and all stderr are capped at 64 KiB. Overflow, malformed JSON, timeout, or non-zero exit fails closed; mirror checkpoint failures alone are queued for retry.

## Native hooks

Install owned hook fragments additively; unrelated hooks and their order are preserved.

```bash
agent-session-harness hooks install \
  --runtime claude --path ~/.claude/settings.json --json

agent-session-harness hooks install \
  --runtime codex --path .codex/hooks.json --json

agent-session-harness hooks check \
  --runtime codex --path .codex/hooks.json --json
```

`--dry-run` and `hooks uninstall` are supported. Invalid JSON is never mutated. Hook execution requires `AGENT_SESSION_HARNESS_MANAGED=1`, writes only to the local lifecycle ledger, performs no network work, and applies a 1 MiB input bound.

At a Stop event, normal sessions are allowed to stop. A draining session receives one continuation request listing the configured durable checkpoints. A recursion marker prevents repeated blocking. Once the capsule fingerprint is verified, Stop is allowed immediately.

## Successor acknowledgement

The supervisor gives a successor only the capsule path, fingerprint, target generation, and a short instruction. The successor verifies the capsule locally and acknowledges it:

```bash
agent-session-harness acknowledge \
  --state .agent-session-harness/supervisor.json \
  --generation 1 \
  --fingerprint <sha256> \
  --conversation-id <native-conversation-id> \
  --json
```

The persistent launch guardian exports its owner PID to the managed runtime, and the CLI uses it automatically. `--owner-pid` is available for a host that performs the acknowledgement directly instead of from the managed runtime environment.

Normal dispatch stays disabled until the expected generation and fingerprint match. Automatic rotation is equivalent to a safe `/clear`: it creates a new native conversation and deliberately has no resume path. Manual resume remains the responsibility of the calling launcher.

## Status consumers

`report --json` emits the stable downstream status contract used by terminal and dashboard integrations: runtime, governor state, context percentage/confidence, quiescence, active turn/tool/subagent/critical counts, chain/conversation/generation IDs, last checkpoint fingerprint, and outbox depth.

```bash
agent-session-harness report \
  --state .agent-session-harness/supervisor.json \
  --ledger .agent-session-harness/supervisor.json.lifecycle \
  --outbox .agent-session-harness/mirrors.jsonl \
  --json
```

Consumers should project this record; they should not infer lifecycle ownership from CPU usage or launch their own model-backed summarizer.

## Integration boundaries

- [`agent-coordinator`](https://github.com/Boundless-Studios/agent-coordinator) owns atomic claims and lease-epoch fencing.
- `worktree-deck` may route fresh actions through a managed command while preserving its explicit manual-resume action.
- `agentic-pr-dash` may ingest lifecycle events and canonical status reports, but PR concepts never enter this package.
- Project adapters own beads/Linear updates, repository safety checks, and any project-specific hook policy.

## Privacy and recovery

Usage parsers whitelist only accounting metadata. Capsules reject unknown fields and credential-shaped assignments, adapter diagnostics are bounded and redacted, ledgers/outboxes/state files use `0600`, and corrupt records fail closed. Harness-controlled state, lock, registry, outbox, claim, and hook-manifest operations reject symlink targets and symlinked parent directories.

On restart, the supervisor resumes from the last non-terminal durable phase. Checkpoints, fencing, stopping, same-owner claims, and launch keys are idempotent. If a crash occurs after a child starts but before completion is recorded, the process registry finds the existing chain/generation instead of launching a duplicate. The persistent guardian also watches the supervisor heartbeat and terminates the runtime process group even when no cleanup handler runs.

The coordinator store treats lease expiry—not supervisor-PID death—as the reclaim boundary because a guardian-owned runtime can briefly outlive its supervisor. Claim and persisted heartbeat timestamps are identical. The watchdog deadline is three seconds earlier than the claim lease, exceeding its bounded poll, TERM, KILL, and scheduling-slack budget so process-group shutdown precedes reclamation. Accordingly, `heartbeat_interval_seconds` must be lower than `lease_seconds - 3`; the defaults are 20 and 60 seconds.

MIT licensed. See [LICENSE](LICENSE).
