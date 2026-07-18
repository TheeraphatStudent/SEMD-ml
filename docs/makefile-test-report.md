# SEMD Makefile Verification Report

**Status: PARTIAL / HANDED OFF.** Group A (read-only) and part of Group B (service lifecycle) are
tested with real defects found and fixed. Group C (train/predict/evaluate), Group D (migration),
the rest of Group B (`start`/`stop`/`restart` full lifecycle, final `worker` retest), and Phase 6
regression were **not executed** in this session — see §14/§15/§16 for exact continuation steps.
This report follows the required structure; sections for untested work are marked explicitly
rather than filled with invented results.

## 1. Executive Summary

| | Count |
|---|---|
| Total public targets | 22 (`help`, `venv`, `run`, `verify-imports`, `mlflow-permissions`, `start`, `stop`, `status`, `restart`, `logs`, `cli`, `ml-help`, `train`, `train-obo`, `predict`, `predict-test`, `evaluate`, `feature-engineering`, `worker`, `queue-status`, `data-migrate`, `data-migrate-feature`) |
| Tested this session | 9 (`help`, `verify-imports`, `ml-help`, `cli`, `queue-status`, `status`, `venv`, `logs`, `run`) fully; `worker` partially |
| Passed (after fixes) | 9 |
| Partial / needs final confirmation | 1 (`worker`) |
| Not yet executed | 12 (`mlflow-permissions`, `start`, `stop`, `restart`, `train`, `train-obo`, `predict`, `predict-test`, `evaluate`, `feature-engineering`, `data-migrate`, `data-migrate-feature`) |
| Defects found | 7 (D1–D7) |
| Defects fixed | 7 |
| Defects retested clean | 5 of 7 (D1, D3, D4, D5 confirmed; D2 fix applied but not yet exercised against a live model; D6+D7 fix applied, partial retest showed correct behavior for job processing but the final clean-shutdown timing confirmation was interrupted before completion) |
| **Final Section 2 readiness** | **No — see §15.** Group C, Group D, and the full lifecycle/regression pass (Phase 6) have not run yet. |

## 2. Environment

| Item | Value |
|---|---|
| OS | Ubuntu 24.04.4 LTS (Noble Numbat) |
| Repository path | `/home/semd/Desktop/Project/SEMD/semd-ml` |
| Branch | `feat/mlops` (baseline commit `9f2af32`) |
| `python3 --version` | Python 3.12.3 |
| `uv --version` | uv 0.11.26 (Homebrew build) |
| `docker --version` | not installed (`command not found`) — **Podman-only host** |
| `podman --version` | podman version 4.9.3 |
| `podman-compose --version` | podman-compose version 1.0.6 |
| `podman compose` (plugin) | delegates to `podman-compose` 1.0.6 ("external compose provider") |
| `make --version` | GNU Make 4.3 |
| Active Python environment | `.venv/bin/python` -> `/usr/bin/python` (3.12); all CLI targets go through `uv run`, no manual activation needed |
| MLflow client (`uv run`, host) | 3.14.0 — matches server |
| MLflow client (inside `semd-ml-service` container) | 3.10.1 — pre-existing skew, non-blocking (§14) |
| MLflow server (`ghcr.io/mlflow/mlflow:v3.14.0`) | 3.14.0, healthy |
| Container runtime | Podman only, no Docker on this host |

### Service topology at session start (already running, from a prior session)

| Container | Image | Status | Port |
|---|---|---|---|
| `semd-database` | `postgres:latest` | Up ~1h | host `5433` -> container `5432` |
| `semd-redis` | `database_redis:latest` | Up ~1h, healthy | `6379` |
| `docker_backend_1` | `docker_backend:latest` | Up ~33m | `8000` |
| `semd-mlflow` | `ghcr.io/mlflow/mlflow:v3.14.0` | Up ~21m, healthy | `5000` |
| `semd-ml-service` | `docker_ml-service:latest` | Up ~21m | none published (worker, no HTTP) |

A separate, non-containerized process listens on `127.0.0.1:5432` (native/system PostgreSQL, unrelated to `semd-database`'s `5433` mapping) — see §14, this affects `make start`'s port-based health check semantics (not a defect fixed in this session, a documented host-level ambiguity).

`semd-ml-service`'s running image (built `2026-07-13T21:37:39Z`) was verified byte-for-byte identical to the working tree's `src/workers/queue_worker.py` **before** this session's Group B fixes were applied (`podman exec semd-ml-service cat /app/src/workers/queue_worker.py` diffed clean). **After** this session's `queue_worker.py`/`infra/redis_client.py` edits (D6/D7, see §7), the container's baked-in copy is now **stale** relative to source — `docker-compose.yml` does not bind-mount `src/`, so a container rebuild (`podman compose -f docker/docker-compose.yml build ml-service`) is required before the container's own worker reflects the shutdown-hang fix. Flagged in §14/§16, not done in this session.

I stopped `semd-ml-service` mid-session to isolate a `make worker` foreground test from the container's own combined worker (both would otherwise consume the same Redis queues, which would have produced ambiguous/duplicate-worker evidence). It was **restarted** before handoff (`podman start semd-ml-service`, confirmed `Up`) — baseline container topology is restored, but the container is running the pre-fix code (see previous paragraph).

### Champion baseline (MLflow registry) — verified unchanged at handoff time

- Registered model: `semd-malicious-url-detector`
- `champion` alias -> version **4**, run `9fef933e698b49a4b81d77dd4b4670d7`, experiment `semd-url-classification-v2` (id `29`)
- `previous-champion` alias -> version 1 (pre-fix, unrecoverable — untouched)
- `candidate` alias -> version 4

No target executed in this session wrote a new model version or moved any alias. This must be re-verified after Group C runs (`gate-check`/`promote` were never invoked).

## 3. Baseline Infrastructure State

Baseline `git status --short` at session start (before this session's fixes):

```
 M .env.example
 M docker/docker-compose.yml
 M docs/section-10-infrastructure-validation.md
 M docs/troubleshooting.md
 M mlflow_data/mlflow.db
 M src/core/config.py
 M src/ml/training_service.py
 M src/tracking/__init__.py
 M src/tracking/mlflow_tracker.py
 M src/workers/queue_worker.py
 M tests/integration/test_mlflow_artifact_persistence.py
?? artifacts/23/
?? artifacts/29/
?? artifacts/30/
?? artifacts/36/
?? tests/unit/test_queue_worker.py
```

This uncommitted state is **prior, unrelated session work** documented in `docs/section-10-infrastructure-validation.md` ("Session 2") — it fixed the MLflow artifact-persistence bug, the backend Redis-config bug, and silent queue-job loss, producing the champion v4 described above. This report audits the **Makefile surface only**; it does not re-verify that prior work (Phase 6 regression, when run, will exercise the tests that cover it).

## 4. Target Inventory

| Target | Description (per `make help`) | Risk Class | Dependencies |
|---|---|---|---|
| `help` | List all targets | A (read-only) | none |
| `venv` | Create `.venv`, install `requirements.txt` via `uv` | B (lifecycle) | `uv` |
| `run` | `uv run python $ARGS` from `src/` | B | `uv`, arbitrary script |
| `verify-imports` | `uv run python verify_imports.py` from `src/` | A | `uv`, all app modules |
| `mlflow-permissions` | mkdir + chown/chmod `mlflow_data`/`models`/`reports` | B (uses `sudo`) | filesystem, `sudo` |
| `start` | Start Postgres/Redis (backend compose), backend API, MLflow | B | Podman, backend submodule checkout |
| `stop` | Stop all of the above | B | Podman, backend submodule checkout |
| `status` | Report container/port status | A | Podman |
| `restart` | `stop` + sleep 3 + `start` | B | same as start/stop |
| `logs` | Follow logs for `mlflow`\|`backend`\|`database` | B (foreground) | Podman compose |
| `cli` | Passthrough `uv run main.py $ARGS` | A | `uv`, CLI parser |
| `ml-help` | `main.py [$ARGS] --help` | A | `uv`, CLI parser |
| `train` | `main.py train` | C (training) | MLflow, dataset files |
| `train-obo` | `main.py train-obo` (legacy one-by-one) | C | MLflow, `dataset/store` |
| `predict` | `main.py predict` | C | MLflow registry, champion/candidate model |
| `predict-test` | `main.py predict-test` (batch) | C | MLflow registry |
| `evaluate` | `main.py evaluate` | C | MLflow, dataset files |
| `feature-engineering` | `main.py feature-engineering` | C | feature reference CSVs |
| `worker` | `main.py worker --mode $MODE` (foreground) | B | Redis |
| `queue-status` | `main.py queue-status` | A | Redis |
| `data-migrate` | Extract dataset archives -> raw CSVs | D (migration) | `dataset/store/*` archives |
| `data-migrate-feature` | Migrate feature reference CSVs | D | `dataset/feature/store/*`, `dataset_feature.yaml` |

Note: the underlying CLI (`src/cli/main.py`) also exposes `data validate`, `register`, `promote`, `rollback`, `gate-check`, `feedback`, `review`, `monitor`, `retrain` — these have **no dedicated Makefile target** (reachable only via `make cli ARGS='...'`). Not a defect: `make help`'s own text only promises the targets it lists, and `cli`'s passthrough legitimately covers the rest. Flagged for awareness, not fixed.

## 5. Verification Matrix

| Target | Command | Exit Code | Result | Evidence | Side Effects |
|---|---|---|---|---|---|
| `help` | `make help` | 0 | PASS | §6.1 | none |
| `verify-imports` | `make verify-imports` | 2 -> **0 after fix** | FAIL -> PASS | §6.2, §7 D1 | none |
| `ml-help` (no ARGS) | `make ml-help` | 0 | PASS | §6.3 | none |
| `ml-help` (ARGS=predict) | `make ml-help ARGS=predict` | 0 | PASS | §6.3 | none (also surfaced D2) |
| `cli` (no ARGS) | `make cli` | 2 (usage guard, expected) | PASS | §6.4 | none |
| `cli` (queue-status) | `make cli ARGS="queue-status"` | 0 | PASS | §6.4 | none |
| `queue-status` | `make queue-status` | 0 | PASS | §6.5 | none |
| `status` | `make status` | 0 -> **0, corrected output after fix** | PASS (partial->full) | §6.6, §7 D3 | none |
| `run` | `make run ARGS='verify_imports.py'` | 2 -> **0 after fix** | FAIL -> PASS | §6.7, §7 D4 | none |
| `venv` | `make venv` | 0 | PASS | §6.8 | reinstalled `.venv` packages (idempotent) |
| `logs` (mlflow) | `timeout 8 make logs LOG_SERVICE=mlflow` | 124 (expected — bounded externally) | PASS | §6.9 | none (read-only follow) |
| `logs` (backend) | `timeout 5 make logs LOG_SERVICE=backend` | 124 (expected) | PASS | §6.9 | none |
| `logs` (database) | `timeout 5 make logs LOG_SERVICE=database` | 2 -> **124 after fix** | FAIL -> PASS | §6.9, §7 D5 | none |
| `logs` (invalid) | `make logs LOG_SERVICE=bogus` | 2 (usage guard, expected) | PASS | §6.9 | none |
| `worker` (prediction mode) | `uv run python main.py worker --mode prediction` (equivalent to `make worker MODE=prediction`) | n/a — foreground, SIGTERM-terminated | **PARTIAL** | §6.10, §7 D6/D7 | 1 valid + 1 invalid prediction job processed correctly; found and fixed a shutdown-hang bug; final clean-shutdown timing retest not completed before handoff |
| `mlflow-permissions` | — | — | **NOT RUN** | — | — |
| `start` | — | — | **NOT RUN** | — | — |
| `stop` | — | — | **NOT RUN** | — | — |
| `restart` | — | — | **NOT RUN** | — | — |
| `train` | — | — | **NOT RUN** | — | — |
| `train-obo` | — | — | **NOT RUN** | — | — |
| `predict` | — | — | **NOT RUN** (fix D2 applied, unexercised) | — | — |
| `predict-test` | — | — | **NOT RUN** | — | — |
| `evaluate` | — | — | **NOT RUN** | — | — |
| `feature-engineering` | — | — | **NOT RUN** | — | — |
| `data-migrate` | — | — | **NOT RUN** | — | — |
| `data-migrate-feature` | — | — | **NOT RUN** | — | — |

## 6. Detailed Target Results

### 6.1 `help`
`make help` — exit 0. Lists all 22 documented Make targets (see §4 note on the 9 additional CLI subcommands with no dedicated target — not a defect). Descriptions matched real behavior for every target actually exercised this session.

### 6.2 `verify-imports` — FAIL then PASS (D1)
Before fix: `uv run python verify_imports.py` was run from the **repo root**, but the script lives at `src/verify_imports.py`. Failure: `can't open file '.../semd-ml/verify_imports.py': [Errno 2] No such file or directory`, `make` exit 2.
After fix (D1, §7): exit 0, `10/10` checks passed (core/features/data/ml/infra/tracking/worker/CLI imports, backward compatibility, feature extraction — 73 features extracted on the smoke URL).

### 6.3 `ml-help`
`make ml-help` (no ARGS): exit 0, output matches `src/cli/main.py`'s argparse tree exactly (21 subcommands including the 9 not exposed as Make targets — see §4).
`make ml-help ARGS=predict`: exit 0, revealed the `predict` subcommand's parser only defines a **positional** `url` (`nargs="?"`), not a `--url` flag — this is what led to finding D2.

### 6.4 `cli`
No `ARGS`: correctly prints usage and exits 1 (make reports this as its own exit 2 — see the GNU Make note below).
`ARGS="queue-status"`: exit 0, correct passthrough, real Redis output.

**GNU Make exit-code ceiling (applies to every target, not a defect):** empirically verified (`printf 'x:\n\texit 3\n' | make -f - x` -> `make` itself always exits `2` on any failing recipe line, regardless of the child's real exit code — this is GNU Make's own documented behavior, "exit status 2 means make itself encountered an error"). Every target in this Makefile correctly produces exit 0 on success and a **nonzero** exit on failure (visible via `make`'s own "Error N" message, which reports the true child code even though `make`'s own process exit is always 2). Read every "exit 2" in this report's matrix in that light — it means "the underlying command failed and the child's real code is shown in the Error line," not "the Makefile mis-propagated a code."

### 6.5 `queue-status`
`make queue-status`: exit 0. Reported all 3 queues (`ml_training_queue`, `ml_prediction_queue`, `ml_result_queue`), each showing depth without ever printing `REDIS_PASSWORD`.

### 6.6 `status` — PASS (partial) then PASS (full) (D3)
Before fix: exit 0, but the "Backend Database:" section was **silently blank** for Postgres — `--filter "name=postgres"` never matches the real container name `semd-database` (confirmed against `semd-backend/database/docker-compose.database.yaml`'s `container_name: semd-database`). Redis/backend/mlflow filters happened to match by accidental substring luck.
After fix (D3, §7): `semd-database: Up About an hour` now shown correctly.
**Separately (not fixed, documented limitation):** the Port Status line `[OK] PostgreSQL: localhost:5432` is checking a **non-containerized system Postgres process** (native, `127.0.0.1:5432`) unrelated to the project's own `semd-database` container (mapped `5433`->`5432`). This is a host-level ambiguity inherent to port-based health checks, not something fixable purely inside this Makefile — flagged in §14.

### 6.7 `run` — FAIL then PASS (D4)
Same root cause as D1, in the sibling `run` target. `make run ARGS='verify_imports.py'` (the Makefile's **own documented usage example** in `make help`) failed identically: `can't open file '.../semd-ml/verify_imports.py'`. Fixed identically (add `cd $(SRC_DIR) &&`), retested — exit 0, same 10/10 pass output as §6.2.

### 6.8 `venv`
`make venv`: exit 0. Confirmed idempotent (ran against an already-existing `.venv`, reinstalled cleanly, listed ~90 packages including `scikit-learn 1.9.0`). Uses `requirements.txt` — confirmed this **matches** `docker/Dockerfile`'s own `pip install -r requirements.txt` (both diverge from the separately-present `pyproject.toml`/`uv.lock`, which neither `venv` nor the container build actually use — see §14, not a Makefile defect since venv and the container build are already consistent with each other). Does not require manual activation for any other target (all others use `uv run`, independent of whether `make venv` ran).

### 6.9 `logs` — FAIL (database only) then PASS (D5)
`LOG_SERVICE=mlflow`: real, live MLflow server logs streamed correctly (bounded externally via `timeout 8`), including live evidence of prior-session MLflow/champion activity (registry lookups for `semd-malicious-url-detector`/`champion`, run `9fef933e...` — corroborates §2's champion baseline independently).
`LOG_SERVICE=backend`: real backend container logs streamed (bounded via `timeout 5`).
`LOG_SERVICE=database` — before fix: `Error: executing /usr/bin/podman-compose logs -f: exit status 255` — `no compose.yaml ... file found`. Root cause: this case in the `logs` target's `case` statement omits `-f docker-compose.database.yaml` (present in every other case, and in the `start`/`stop` targets' equivalent code), and `podman-compose` doesn't autodetect `docker-compose.database.yaml` as a default filename. After fix (D5, §7): streams real database container logs, bounded exit 124 as expected.
Invalid `LOG_SERVICE=bogus`: correctly prints `Available logs: mlflow, backend, database` and exits nonzero.

### 6.10 `worker` — PARTIAL, two real defects found and fixed (D6, D7)
Tested via the exact command the Makefile target wraps (`cd src && uv run python main.py worker --mode prediction`), after temporarily stopping the already-running `semd-ml-service` container's own combined worker (`podman stop semd-ml-service`) so job-processing evidence could be attributed unambiguously to this test instance rather than a duplicate consumer. Restarted the container at the end of the session (see §2).

**Job processing — PASS.** Pushed two real jobs directly onto `ml_prediction_queue` (same Redis instance the container uses) via `redis_client.push_to_queue`:
- Valid job (`http://makefile-worker-smoke-test.bad-example.net/verify`) -> processed successfully, champion model artifact downloaded and used, result pushed to `ml_result_queue`.
- Invalid job (`model_id="nonexistent-alias-makefile-test"`) -> produced a structured failure (`ModelRegistryError`, matching the queue-worker robustness fix already in the working tree from the prior session), and — critically — **the worker continued and was ready for the next job afterward**, satisfying the "worker continues after a failure" requirement.

**Clean shutdown — FAIL, then two real fixes applied (D6, D7).**
1. First hang (D6): `_signal_handler` logged *before* setting `self.running = False`. A logging call from inside a signal handler intermittently raised `RuntimeError: reentrant call inside <_io.BufferedWriter name='<stderr>'>` (the main thread was mid-write to the same stream when the signal landed) — when that happened, `self.running = False` was never reached, so shutdown was never flagged. Fixed by reordering the two lines (flag first, log second) so the flag is set unconditionally regardless of whether the log call itself faults.
2. Second, deeper hang (D7), found by retesting after the D6 fix: even with the flag set correctly and the signal handled exactly once, the worker still did not exit for a long and unpredictable time (observed: still alive >70s after a single `SIGTERM`, before eventually raising `redis.exceptions.TimeoutError: Timeout reading from socket` and exiting cleanly). Root cause: `src/infra/redis_client.py`'s `redis.Redis(...)` constructor set no `socket_timeout`/`socket_connect_timeout`, so once a signal interrupts an in-flight blocking `BRPOP` read, redis-py has no client-side bound on how long it will wait to recover — it doesn't return control to the `while self.running:` loop until *something* eventually breaks the read. This is the exact same class of bug the prior session's `docs/section-10-infrastructure-validation.md` already fixed for `semd-backend`'s two Redis client classes ("previously unbounded, risking an indefinite hang") — that fix's scope was `semd-backend` only and never touched `semd-ml`'s own `infra/redis_client.py`. Fixed by adding `socket_connect_timeout=5, socket_timeout=10` (10s comfortably exceeds the longest `BRPOP` timeout `QueueWorker` uses, which is 5s, so it never fires during normal polling — only as a bound on a stuck/interrupted read).

**Status at handoff:** both fixes are applied to `src/workers/queue_worker.py` and `src/infra/redis_client.py`. A retest after the D7 fix showed the worker correctly processing jobs; the final timed re-verification of "SIGTERM -> exit within a few seconds" was in progress (using `run_in_background` process tracking to avoid the interactive-shell PID-tracking flakiness that affected earlier attempts — see the "process handling notes" below) when this session was asked to hand off. **This must be the first thing re-verified in a continuation session** — see §16 for the exact command.

No duplicate worker processes were left running at handoff — confirmed via `ps -ef` on the host (only the restarted `semd-ml-service` container's own worker, plus the unrelated backend `workers.prediction_worker` and MLflow server processes, are present).

**Process-handling notes for whoever continues this:** this sandboxed shell exhibits `set -e`-like behavior across compound `;`-separated command chains in a single `Bash` tool call — any command that legitimately exits nonzero (e.g. `pgrep`/`grep` finding no matches, which is a normal "nothing found" signal, not an error) can silently abort the rest of the chain and suppress output, even when followed by `|| true`. Prefer the tool's `run_in_background: true` execution for anything long-running, and re-fetch PIDs with a fresh, isolated `ps -ef | grep` in its own call rather than chaining kill/verify logic into one multi-statement script.

## 7. Defects Found

| ID | Target | Root Cause | Impact | Files Changed | Retest |
|---|---|---|---|---|---|
| D1 | `verify-imports` | Recipe never `cd`s into `$(SRC_DIR)`; script is at `src/verify_imports.py` | Target always fails (`ENOENT`) | `makefile` | PASS — 10/10 checks |
| D2 | `predict` | Recipe passes `--url "$(URL)"`, but `predict`'s argparse subparser (`src/cli/main.py`) only defines a positional `url` (`nargs="?"`), no `--url` flag — every `make predict URL=...` invocation would fail argparse | Documented usage in `make help` cannot work at all | `makefile` (changed `--url "$(URL)"` -> `"$(URL)"` positional) | Fix applied, **not yet exercised** — first item for continuation (§16) |
| D3 | `status` | `--filter "name=postgres"` never matches the real container name `semd-database` (confirmed against `semd-backend/database/docker-compose.database.yaml`) | Postgres status line silently blank, even when the container is healthy | `makefile` | PASS — line now shows `semd-database: Up ...` |
| D4 | `run` | Same as D1, sibling target — no `cd $(SRC_DIR)` | `make help`'s own documented usage example fails | `makefile` | PASS — 10/10 checks |
| D5 | `logs` (`LOG_SERVICE=database`) | `database` case in the `logs` target's `case` statement omits `-f docker-compose.database.yaml` (present in every sibling case and in `start`/`stop`) | `podman-compose` can't find a default-named compose file, exits 255 | `makefile` | PASS — streams real logs, bounded exit as expected |
| D6 | `worker` | `_signal_handler` logs before setting `self.running = False`; if the logging call itself raises (observed: reentrant stderr write), the shutdown flag is never set | Worker can ignore SIGINT/SIGTERM entirely | `src/workers/queue_worker.py` | Fix applied; confirmed the flag is now unconditionally set, but see D7 — this alone was insufficient to produce a fast clean shutdown |
| D7 | `worker` | `infra/redis_client.py`'s `redis.Redis(...)` sets no `socket_timeout`/`socket_connect_timeout`; a signal interrupting an in-flight blocking `BRPOP` read leaves redis-py with no bound on recovery time (observed >70s before it self-resolved) | Worker shutdown after a signal is unpredictably slow, effectively "doesn't shut down cleanly" per the target's own pass criterion | `src/infra/redis_client.py` (added `socket_connect_timeout=5, socket_timeout=10`) | Fix applied; job-processing retest passed; **final SIGTERM-timing retest not completed before handoff** — first item for continuation (§16) |

No defects were found that required touching `src/cli/main.py` itself, `docker/docker-compose.yml`, `.env`/`.env.example`, or any file outside `makefile` + the two `queue_worker.py`/`redis_client.py` edits above.

## 8. Training and MLflow Evidence

**Not applicable — Group C (`train`, `train-obo`, `predict`, `predict-test`, `evaluate`, `feature-engineering`) was not executed this session.** No new experiment, run, or registered model version was created. The champion (v4, run `9fef933e698b49a4b81d77dd4b4670d7`, experiment `semd-url-classification-v2` id `29`) is unchanged from the session-start baseline (§2). §16 has the exact plan for this (isolated experiment `semd-makefile-verification`, fixture `src/dataset/raw/t093_smoke_fixture.csv`).

## 9. Queue and Worker Evidence

Two real jobs processed by a live `worker --mode prediction` instance against the real Redis queues, using the real (non-mocked) prediction pipeline and the real champion model:

| Job ID | Type | Input | Outcome |
|---|---|---|---|
| `f8292562-a657-4753-8f9e-72ec93104eb8` | prediction | `http://makefile-worker-smoke-test.bad-example.net/verify` | `status: success`, champion artifact downloaded, result pushed to `ml_result_queue` |
| `225d69f5-8bca-4fdb-a68e-749b8d90e57a` | prediction | `http://makefile-worker-invalid-test.bad-example.net/x`, `model_id="nonexistent-alias-makefile-test"` | Structured failure: `error_type=ModelRegistryError`, `error_message="Unable to load model from MLflow registry: Alias 'nonexistent-alias-makefile-test' is not assigned for model 'semd-malicious-url-detector'"`, pushed to `ml_result_queue` — job not lost, worker continued |

`queue-status` confirmed all 3 queues at depth 0 both before and after (jobs drained, not stuck). No duplicate workers left running (§6.10). Full worker-mode (`combined`/`training`) and the final clean-shutdown timing retest are **not done** — §16.

## 10. Migration Evidence

**Not applicable — Group D (`data-migrate`, `data-migrate-feature`) was not executed this session.** Isolation plan (verified safe by reading `src/cli/commands/migrate.py`, not yet executed): both commands accept `--store-path`/`--raw-path` overrides and `extract_csvs_from_archive(..., overwrite=False, ...)`/dedup logic is idempotent by construction — pointing them at temp directories seeded from `src/dataset/store/*` / `src/dataset/feature/store/*.csv` is safe and was not expected to require any additional Makefile changes. See §16.

## 11. Regression Results

**Not run this session.** Phase 6's three regression commands (`pytest tests/ -q`, `scripts/verify_container_paths.py`, backend `unittest`) and the final lifecycle sequence (`stop`/`start`/`status`/`queue-status`/`predict-test`/`restart`/`status`) were not executed. §16 has the exact commands.

## 12. Files Changed

### `semd-ml` (this module — the only module touched)

| File | Change | Defect(s) |
|---|---|---|
| `makefile` | `run`/`verify-imports`: added `cd $(SRC_DIR) &&`. `predict`: changed `--url "$(URL)"` to positional `"$(URL)"`. `status`: changed Postgres container filter from `name=postgres` to `name=semd-database`. `logs` (`database` case): added `-f docker-compose.database.yaml` | D1, D2, D3, D4, D5 |
| `src/workers/queue_worker.py` | `_signal_handler`: set `self.running = False` before logging, not after | D6 |
| `src/infra/redis_client.py` | `RedisClient.__init__`: added `socket_connect_timeout=5, socket_timeout=10` to the `redis.Redis(...)` constructor | D7 |

No other repository or module was modified. `docs/makefile-test-report.md` (this file) is new.

## 13. Git Status

- Baseline commit `9f2af32` — **untouched**, no commits created or amended this session.
- No new commits created (per standing instruction: only commit when explicitly asked).
- Uncommitted changes at handoff = session-start baseline (§3) **plus**:
  - `makefile` (D1/D2/D3/D4/D5 fixes)
  - `src/workers/queue_worker.py` (D6 fix, on top of the prior session's own uncommitted changes to this same file)
  - `src/infra/redis_client.py` (D7 fix — this file had no prior uncommitted changes)
  - `docs/makefile-test-report.md` (new, this report)
- Runtime/generated noise deliberately not evaluated for commit-worthiness (`mlflow_data/mlflow.db`, `artifacts/23,29,30,36/`) — pre-existing from the prior session, unrelated to this report's scope.

## 14. Remaining Issues

**Blocking (must resolve before Section 2 can be marked ready):**
- Groups C and D entirely untested (12 targets).
- `worker`'s clean-shutdown fix (D7) needs a final timing confirmation retest.
- `start`/`stop`/`restart` never exercised — `mlflow-permissions` never exercised.
- Phase 6 regression suite never run.
- `semd-ml-service` container is running **pre-fix** code for D6/D7 (source fixed, image not rebuilt — `docker-compose.yml` doesn't bind-mount `src/`). Needs `podman compose -f docker/docker-compose.yml build ml-service` + recreate before the container's own worker reflects the fix.

**Non-blocking (documented, not fixed — out of this session's scope):**
- `status`'s Port Status line for PostgreSQL (`localhost:5432`) is satisfied by an unrelated native/system Postgres process, not the project's own `semd-database` container (mapped to `5433`). Inherent to port-based health checks; would need a container-aware check (e.g. `podman inspect` port mapping lookup) to fully close, which is more than the "smallest maintainable correction" this session's fixes aimed for.
- `mlflow` client version skew: 3.14.0 (host `.venv`, matches server) vs 3.10.1 (baked into `semd-ml-service`'s image, from a looser `mlflow>=3.10.0` pin in `requirements.txt` resolved at a different build time). Same category of pre-existing skew as the already-flagged `scikit-learn` 1.8.0-vs-1.9.0 issue in `docs/section-10-infrastructure-validation.md`.
- `pyproject.toml`/`uv.lock` exist alongside `requirements.txt` but neither `make venv` nor `docker/Dockerfile` use them — both actually use `requirements.txt`, so `venv`'s own pass criterion ("uses the same dependency source as the container build") is satisfied, but the two dependency-declaration files can silently drift from each other over time. Worth reconciling in a future session, not a Makefile defect.
- The CLI has 9 subcommands (`data validate`, `register`, `promote`, `rollback`, `gate-check`, `feedback`, `review`, `monitor`, `retrain`) with no dedicated Makefile target — reachable only via `make cli ARGS=...`. Not a defect (documented behavior of the `cli` passthrough), flagged for awareness since Group C testing (§16) will need `gate-check`/`register`/`promote` reachable this way if a full training-to-promotion smoke cycle is exercised.

**Production-quality follow-ups (pre-existing, out of scope for this report, listed in `docs/section-10-infrastructure-validation.md` already):** champion-comparison promotion-gate logic, duplicate Redis client files in `semd-backend`, Postgres connection URL logged in plaintext in `semd-backend`, scikit-learn version skew, fixture-based validation not representing production model quality.

## 15. Ready for Next Section

**No — Groups C and D (12 targets: `mlflow-permissions`, `start`, `stop`, `restart`, `train`, `train-obo`, `predict`, `predict-test`, `evaluate`, `feature-engineering`, `data-migrate`, `data-migrate-feature`) are untested, the D7 worker-shutdown fix needs a final timing confirmation, and Phase 6 regression has not run. Continue from §16.**

## 16. Continuation Plan (exact next steps)

1. **Confirm D7 worker-shutdown fix.** In one isolated shell session (avoid chaining `pgrep`/`kill`/verification into a single multi-statement call — see the process-handling note in §6.10):
   ```
   cd src && uv run python main.py worker --mode prediction   # run_in_background: true
   # wait ~4s, find pid via a fresh `pgrep -f "main.py worker --mode prediction"`
   # send SIGTERM, poll `kill -0 $PID` once per second
   # expect exit within ~11s (10s socket_timeout + ~1s except-block sleep), not the >70s seen pre-fix
   ```
   Confirm no duplicate worker is left running afterward (`ps -ef | grep "main.py worker"`).

2. **Rebuild `semd-ml-service`** so the container reflects D6/D7 (and confirm `make restart` / Phase 6's lifecycle sequence exercises the rebuilt image, not the stale one):
   ```
   podman compose -f docker/docker-compose.yml build ml-service
   ```

3. **Retest D2 (`predict`)** now that a positional-URL fix is in place, against the live champion:
   ```
   make predict URL='https://example.com'
   make predict URL='http://secure-login99.bad-example.net/verify?token=99' MODEL_ID=champion
   ```
   Confirm classification + confidence + `model_version=4` in the output, and a nonzero exit on a deliberately invalid case (e.g. no URL/URLS/URL_FILE, or a bad `MODEL_ID`).

4. **Group B remainder:** `mlflow-permissions` (rerun-safe, check it doesn't `chown -R` anything outside `models`/`reports`/`mlflow_data`), then `stop` -> `start` -> `status` as a real lifecycle cycle (record container state before/after each), with the pre-existing native-Postgres-on-5432 ambiguity from §14 kept in mind when interpreting `start`'s "already running" branch.

5. **Group C, isolated:**
   ```
   MLFLOW_EXPERIMENT_NAME=semd-makefile-verification make train \
     DATASET_FILES=dataset/raw/t093_smoke_fixture.csv ALGORITHMS=random_forest RUN_NAME=makefile-verify-$(date +%s)
   make evaluate DATASET_FILES=dataset/raw/t093_smoke_fixture.csv ALGORITHMS=random_forest
   make feature-engineering URL='https://example.com'
   make predict-test URL='https://example.com' MODEL_ID=champion
   ```
   Do **not** let any of this call `promote` against the real registry unless deliberately testing `gate-check`/`register`/`promote` via `make cli ARGS=...` in the same isolated experiment — never call `promote` against `semd-url-classification-v2`'s real champion.

6. **Group D, isolated (do not touch `src/dataset/raw` or `src/dataset/feature/raw` directly):**
   ```
   mkdir -p /tmp/makefile-verify-store && cp src/dataset/store/malicious_moutasm_tamimi.zip /tmp/makefile-verify-store/
   make data-migrate STORE_PATH=/tmp/makefile-verify-store RAW_PATH=/tmp/makefile-verify-raw
   # rerun the same command to confirm idempotency (overwrite=False)
   mkdir -p /tmp/makefile-verify-feature-store && cp src/dataset/feature/store/*.csv /tmp/makefile-verify-feature-store/
   make data-migrate-feature STORE_PATH=/tmp/makefile-verify-feature-store RAW_PATH=/tmp/makefile-verify-feature-raw CONFIG=src/dataset/feature/dataset_feature.yaml
   ```

7. **Phase 6 regression:**
   ```
   MLFLOW_TRACKING_URI=http://localhost:5000 uv run python -m pytest tests/ -q
   uv run python scripts/verify_container_paths.py
   uv run python -m unittest tests.unit.test_ml_prediction_service tests.unit.test_settings_redis -v   # in semd-backend
   make stop && make start && make status && make queue-status && make predict-test URL='https://example.com' MODEL_ID=champion && make restart && make status
   ```
   Verify at the end: MLflow/Redis/ML-service/backend all healthy, champion still version 4, no Redis auth errors, no dropped jobs, no unintended alias changes.

8. Update this report's §5/§6/§8/§9/§10/§11/§15 with the results, then re-derive §1's executive-summary counts and give a final `Yes` / `No — <blocker>` verdict.
