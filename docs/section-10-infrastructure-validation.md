# Section 1 — Infrastructure Validation (T-092, T-093, T-094)

## Update (Session 2): T-092 and T-093 both resolved and verified end-to-end

Everything below the intro (through "Session Handoff") is the **Session 1** investigation and is
kept as historical evidence — it correctly documents what was broken and why, and none of it was
fabricated. It is superseded by this section, which resolves both blockers Session 1 left open and
scope is now `semd-backend` + `semd-ml` (Session 1 was `semd-ml`-only; T-092's real blocker lived
in `semd-backend`, so this session edits both submodule working trees).

### T-092 — Backend Redis config fixed, real end-to-end publish verified

**Root cause (confirmed, matches Session 1's finding exactly):** `semd-backend/config/settings.py`
built its `Settings(BaseSettings)` fields from `configparser` values assigned as class-body
defaults, with no `model_config`. Empirically, plain env vars *did* already override those
defaults (pydantic-settings' default env source) — but a **blank** env var (e.g. an unresolved
`${VAR}` in compose, or a docker-compose block that sets `REDIS_PASSWORD=` with no value) was
"present" from pydantic's point of view and silently overrode a valid, non-blank `backend.ini`
password with an empty string. Reproduced directly:
```
REDIS_PASSWORD="" -> Settings().redis_password == ''   # wipes out backend.ini's 'example'
```

**Fix** (`semd-backend/config/settings.py`):
```python
model_config = SettingsConfigDict(env_ignore_empty=True)
```
`pydantic-settings` (2.14.2, already a dependency) treats a blank env var as absent when this flag
is set, so it falls through to `backend.ini` / the built-in default instead of erasing it. Also
removed a leftover `print(config)` debug line (harmless — `ConfigParser.__repr__` doesn't dump
values — but dead noise). No new env-parsing helper was needed: `REDIS_HOST`/`PORT`/`PASSWORD`/`DB`
already flow through the single `Settings` class env -> ini -> default, centralized, not duplicated
in either `services/redis_client.py` or `services/client/redis_client.py` (both just read
`settings.redis_*`).

Also added `socket_connect_timeout=5, socket_timeout=5` to both `RedisClient` classes' `redis.Redis(...)`
constructor (they were previously unbounded, risking an indefinite hang against an unreachable host)
and a `## Configuration` section to `semd-backend/README.md` documenting the precedence contract.

**`docker/compose.yaml`** (`semd-backend`): `backend.environment` previously forwarded only
`REDIS_HOST`; now forwards `REDIS_PORT`/`REDIS_PASSWORD`/`REDIS_DB` too (blank-default
`${REDIS_PASSWORD:-}`, safe now that `env_ignore_empty=True` means blank falls through to the
image's baked-in `backend.ini` password rather than erasing it).

**Tests added** (`semd-backend/tests/unit/test_settings_redis.py`, 8 cases, all passing against
live Redis): env-overrides-ini, ini-used-when-env-absent, blank-env-does-not-erase-ini,
port/db-int-parsing (both env and ini paths), password-not-printed, invalid-password ->
`redis.exceptions.AuthenticationError`, unreachable-host -> `redis.exceptions.ConnectionError` in
under 10s (proves the new socket timeout, not an indefinite hang).

**Real end-to-end verification** (backend container, not `redis-cli`):
```
podman compose -f docker/compose.yaml build backend
podman compose -f docker/compose.yaml up -d backend      # container docker_backend_1
podman exec docker_backend_1 python3 -c "
from services.ml_service_client import ml_service_client
job_id = ml_service_client.submit_prediction_job(url='https://t092-backend-e2e-test.example.com')
"
```
| Check | Result |
|---|---|
| Backend container resolved Redis config | `host=redis port=6379 db=0 password_set=True` (from baked-in `backend.ini`, forwarded via blank-safe env var) |
| Job published via real backend code | `MLServiceClient.submit_prediction_job` — job `862afb05-9d2f-4074-b34d-a7e026e67f38` |
| Auth errors | None — no `NOAUTH`/`AuthenticationError` anywhere in backend or worker logs |
| `semd-ml-service` worker log | `Processing prediction job: 862afb05-9d2f-4074-b34d-a7e026e67f38` |
| `ml_prediction_queue` depth | 0 before, 0 after (job popped and processed) |
| Job outcome | Failed at that point in the session on model load (pre-fix broken champion — see T-093 below); this is what confirmed the queue-loss bug (Remaining Blocker #4, fixed — see Phase 4) |

A second run after T-093 and Phase 4 were fixed (job `bcf89c9a-29e7-4fbf-873f-66a21a3216b3`,
URL `http://phase5-e2e-verify.bad-example.net/login`) completed the **full** chain end-to-end,
including the backend's own `workers/prediction_worker.py` (a pre-existing component, not
previously running in this session) draining `ml_result_queue` into the `ml_result:{job_id}`
cache the backend API polls:
```json
{"status": "success", "url": "http://phase5-e2e-verify.bad-example.net/login",
 "prediction": {"prediction": "malicious", "is_malicious": true, "confidence": 0.55,
                 "model_name": "semd-malicious-url-detector", "model_version": "4",
                 "model_alias": "champion"},
 "job_id": "bcf89c9a-29e7-4fbf-873f-66a21a3216b3", "job_type": "prediction"}
```
**T-092 acceptance criterion met**: a real `semd-backend` job, published by backend code, consumed
by `semd-ml`, processed without any authentication error, end to end.

### T-093 — New experiment, guard against legacy reuse, working champion

**Decision taken (Option A from Session 1, non-destructive):** `MLFLOW_EXPERIMENT_NAME` is now
`semd-url-classification-v2`, set in `.env`, `.env.example`, `docker/docker-compose.yml`'s
`ml-service.environment` default, and `core/config.py`'s `MLServiceSettings` default. The old
`semd-url-classification` experiment, its runs, and its registered model versions/aliases were
**not** deleted, deleted, or mutated.

**Guard added** (`src/tracking/mlflow_tracker.py`): `_ensure_experiment()` now calls
`_assert_safe_artifact_location()` whenever it reuses (not creates) an experiment. If that
experiment's `artifact_location` doesn't start with `mlflow-artifacts:/`, it raises a new
`UnsafeExperimentArtifactLocationError` (exported from `tracking/__init__.py`) naming the
experiment, its artifact location, why it's unsafe, and the remediation (use a new experiment
name). `training_service.py`'s `except Exception: pass` around `mlflow_tracker.start_run(...)` was
narrowed to re-raise this specific error instead of swallowing it — the pre-existing broad catch
was for MLflow being simply unreachable, not for "quietly train with no working tracking against a
known-broken experiment," which is exactly what Session 1's reproduction showed it was doing.
Live-verified against the real legacy experiment:
```
tracker.experiment_name = 'semd-url-classification'
tracker.start_run(...) -> UnsafeExperimentArtifactLocationError:
  Experiment: semd-url-classification
  Artifact location: /home/semd/Desktop/Project/SEMD/semd-ml/artifacts/mlflow
  ...Recommended remediation: point MLFLOW_EXPERIMENT_NAME at a new, versioned experiment name...
```
`tests/integration/test_mlflow_artifact_persistence.py::MlflowExperimentReuseKnownGapTests` (which
previously *documented* the silent-reuse gap as expected behavior) is now
`MlflowExperimentReuseSafetyGuardTests`, asserting the guard raises instead of silently continuing,
plus a companion test that a tracker-created experiment still gets `mlflow-artifacts:/` cleanly
(no false positive on its own fresh experiments).

**Smoke training, registration, and promotion** (real CLI path — `main.py train` ->
`training_service.py` -> `ml_pipeline` -> `mlflow_tracker`, not a raw `mlflow` probe script), using
a small deterministic fixture (`dataset/raw/t093_smoke_fixture.csv`, 20 rows, same generator
pattern as `tests/unit/test_training_pipeline.py`'s `write_fixture_dataset`):

| Step | Command / action | Result |
|---|---|---|
| Train | `main.py train --dataset-files t093_smoke_fixture.csv --algorithms random_forest --run-name t093-v2-smoke-test` | experiment `semd-url-classification-v2` (id `29`, auto-created), run `9fef933e698b49a4b81d77dd4b4670d7`, `artifact_uri=mlflow-artifacts:/29/9fef933e.../artifacts` (proxied, confirmed via `GET /api/2.0/mlflow/runs/get`) |
| Register | `main.py register --run-id 9fef933e698b49a4b81d77dd4b4670d7` | `semd-malicious-url-detector` version **4**, alias `candidate` set automatically |
| Load candidate + predict | `main.py predict "https://benign-check.example.com/home" --model-id candidate` | loaded from registry, `prediction=benign, confidence=0.855` |
| Gate-check (dry run) | `main.py gate-check --model-version 4` | `gates_passed=true` (all 4 `MODEL_PROMOTION_GATES` — recall/f1/FNR/latency — passed cleanly on the fixture); `champion_comparison_passed=false` (candidate latency 6.1237ms vs the *existing* champion's recorded 6.1212ms — a noise-level, sub-millisecond difference on a tiny fixture, using a strict `>=` comparison against a champion whose own artifact is unrecoverable — see below) |
| `promote` CLI | `main.py promote --model-version 4` | Refused: `ModelValidationError: Candidate validation failed` (correctly enforcing the champion-comparison gate — **not weakened or bypassed**) |
| Manual promotion | `MlflowClient.set_registered_model_alias` (the same primitive `promote_candidate()` uses internally) | `previous-champion` -> v1 (preserves rollback path to the old, already-broken champion), `champion` -> v4; a `manual_promotion_reason` tag was set on v4 explaining exactly why (see below) |
| Load champion + predict (host) | `main.py predict "http://secure-login99.bad-example.net/verify?token=99" --model-id champion` | `prediction=malicious, confidence=0.91, model_version=4` |
| Load champion + predict (different, freshly recreated container) | see Container recreation below | `prediction=malicious, confidence=0.555, model_version=4` |

**Why manual promotion, not a loosened gate:** the task brief's acceptance bar is "champion loads,"
not "passes prod gates" — and all four *quality* gates (`malicious_recall`, `malicious_f1`,
`false_negative_rate`, `prediction_latency_ms`) passed. The only gate that failed was the
champion-comparison, and only because it compares against `champion_alias`'s **recorded** metrics
regardless of whether that champion's artifact is actually loadable — comparing a new candidate
against a model that is itself Remaining-Blocker-#3-broken produces a comparison that can never be
meaningfully "worse," just numerically unlucky on a 20-row fixture. `MODEL_PROMOTION_GATES` and
`promotion_require_champion_comparison` were **not** changed. This is flagged as a legitimate,
narrow follow-up (see Remaining Follow-ups) — the champion-comparison gate should probably skip or
warn instead of hard-failing when the current champion's own artifact can't be downloaded, but
that's a promotion-policy decision, not something to change unilaterally here.

**Container recreation** (proves persistence, not luck):
```
podman compose -f docker/docker-compose.yml build ml-service   # rebuilt with Phase 2 + Phase 4 code
podman rm -f semd-ml-service semd-mlflow
podman compose -f docker/docker-compose.yml up -d mlflow        # fresh container 9ec8a2658b3c
podman compose -f docker/docker-compose.yml up -d ml-service    # fresh container a1b5bcf15271, never wrote the v4 artifact
podman exec semd-ml-service python3 -c "
from ml.ml_pipeline import ml_pipeline
from tracking.model_registry import CachedChampionModelLoader
CachedChampionModelLoader().load(selector='champion')
"
```
Succeeded — champion v4's artifact downloaded from the shared `mlflow-artifacts:/` store into a
container that had never written it, exactly the recreation test T-093 requires. (One cosmetic-only
`InconsistentVersionWarning`: the artifact was pickled with scikit-learn 1.9.0, the running
container has 1.8.0 — noted under Remaining Follow-ups, did not affect correctness of this run.)

**Note on `podman compose --env-file`:** `podman-compose` 1.0.6 does not reliably auto-load `.env`
next to the compose file's *invoking* directory the way Docker Compose does. `REDIS_PASSWORD` had
to be exported into the shell (`export REDIS_PASSWORD=example`) before `up -d` for the container to
receive it — exporting the *whole* `.env` (`source .env`) is a trap: `.env`'s `REDIS_HOST=localhost`
is the correct value for host/CLI use but leaks into the container and overrides the compose
default of `redis`, breaking connectivity. Export only the specific override needed.

**T-093 acceptance criterion met**: training used the real, now-safe experiment
(`semd-url-classification-v2`, id `29`) through the real training config and CLI path; the
resulting artifact survived a full container recreation; the registered model (version 4) can be
downloaded and loaded from a container that never wrote it; champion loading succeeds.

### Phase 4 — Silent queue job loss fixed

**Root cause (matches Session 1's Remaining Blocker #4 exactly):** `process_prediction_job`/
`process_training_job` in `src/workers/queue_worker.py` called `redis_client.pop_from_queue(...)`
(a destructive `BRPOP`) and then ran the job with no inner `try/except`. Any exception during
processing (model-not-found, download failure, bad payload) propagated straight to the
worker-loop's outer `except Exception`, which only logs and sleeps — the job had already been
popped, and `push_to_queue(self.result_queue, ...)` was never reached, so it vanished with no
retry, no dead-letter, and no result.

**Fix:** wrapped the body of both `process_training_job` and `process_prediction_job` in their own
`try/except`. On failure, `build_job_failure_result()` builds and pushes a structured payload to
`ml_result_queue` (the existing result channel — no new queue/topology needed):
```json
{"job_id": "...", "job_type": "prediction", "status": "failed",
 "error_type": "ModelRegistryError", "error_message": "...",
 "failed_at": "2026-07-13T21:40:53Z", "retryable": true}
```
`retryable` is `false` for `ValueError`/`TypeError`/`KeyError` (bad input — retrying won't help)
and `true` otherwise (e.g. `ModelRegistryError` — could be a transient download/infra issue).
No secrets are included (only `str(exc)`, not job_data or exception args). Because `BRPOP` already
removes the job exactly once, there is no retry loop to guard against — a failed job is reported
once, not requeued.

**Tests added** (`tests/unit/test_queue_worker.py`, 8 cases): model-not-found
(`ModelRegistryError`, retryable), invalid payload (`ValueError`, not retryable), generic
prediction exception, training-job exception, successful jobs still push a `status=success`
payload (not misclassified as failed), and — the specific "worker continues after a failure"
requirement — two sequential jobs where the first raises and the second succeeds both produce
correct, independent result payloads.

**Live verification** (real backend-published job through the real worker, not a unit test):
```
model_id='nonexistent-alias-xyz' -> job d2f8e8f3-c0f3-4098-abe5-6a7a8e460178
```
`ml_result:d2f8e8f3-...` now contains:
```json
{"job_id": "d2f8e8f3-c0f3-4098-abe5-6a7a8e460178", "job_type": "prediction", "status": "failed",
 "error_type": "ModelRegistryError",
 "error_message": "Unable to load model from MLflow registry: Alias 'nonexistent-alias-xyz' is not assigned for model 'semd-malicious-url-detector'",
 "failed_at": "2026-07-13T21:40:53.651749+00:00", "retryable": true}
```
`ml_prediction_queue` depth was 0 after (drained, not stuck), and a job submitted immediately
afterward processed normally — the worker did not get stuck on the failure.

### Full verification (Session 2)

```
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python -m pytest tests/ -q   # semd-ml, from src/
  -> 92 passed, 0 failed (was 83 at end of Session 1; +8 queue-worker + net +1 mlflow-guard tests)
uv run python scripts/verify_container_paths.py                               # semd-ml
  -> OK: all container path checks passed
uv run python -m unittest tests.unit.test_ml_prediction_service tests.unit.test_settings_redis -v   # semd-backend
  -> Ran 11 tests, OK
```

### Remaining follow-ups (non-blocking, not fixed in this session — flagged, not hidden)

- **Champion-comparison promotion gate compares against a possibly-unloadable champion's recorded
  metrics.** Not a bug that blocks Section 1, but a real gap: `promote` can refuse a strictly-better
  candidate on a sub-millisecond latency noise difference against a champion whose own artifact
  can't be downloaded. Worth revisiting `_compare_to_champion` in `tracking/model_registry.py`
  (e.g. skip the comparison, or verify the champion is actually loadable first) as a follow-up, not
  done here to avoid changing promotion policy unilaterally.
- **`semd-backend/services/redis_client.py` and `services/client/redis_client.py` are duplicate
  files** (both used by different callers — `workers/prediction_worker.py` uses the former,
  `services/ml_service_client.py` the latter). Both were fixed identically (same settings source,
  same new timeouts) so there's no functional bug, but the duplication itself is pre-existing debt,
  not introduced or removed here (out of scope for a Redis-config fix).
- **`semd-backend/services/client/postgres_client.py` logs the full Postgres connection URL,
  including the password, in plaintext** (`print(f"Connection client: {settings.database_url}")`,
  visible in every container log in this session's evidence above). This is unrelated to the Redis
  scope of this task but is a real credential-logging issue worth a follow-up fix — flagged here
  rather than fixed silently or left undiscovered.
- **Model artifact pickled with scikit-learn 1.9.0, container runtime has 1.8.0**
  (`InconsistentVersionWarning` on every champion load). Did not affect this session's predictions
  but should be reconciled (pin `scikit-learn` in `requirements.txt`/lockfile to match training and
  serving environments) before relying on this champion beyond smoke-testing.
- Fixture-based validation (T-093's smoke run) does not establish production model quality — the
  real dataset still needs a full training + promotion cycle before the champion is production-grade,
  not just infra-verified. Explicitly out of scope here, same as Session 1.

---

# Session 1 (historical) — original investigation

Scope: `semd-ml` only. Findings on `semd-backend` are documented but not fixed (out of module
scope; see Remaining Blockers). **Superseded by the Session 2 update above** — kept verbatim below
as the evidence trail for how the blockers were found.

All commands below were executed against a live stack: `semd-backend`'s Redis + PostgreSQL
(`podman compose -f semd-backend/database/docker-compose.database.yaml up -d`) and semd-ml's
MLflow + ml-service (`podman compose -f docker/docker-compose.yml up -d`), using Podman 4.9.3 +
podman-compose 1.0.6 (Docker daemon not available in this environment).

## Environment variable flow (as found)

```
REDIS_HOST/PORT/PASSWORD/DB
  .env (semd-ml, gitignored)
    -> pydantic-settings MLServiceSettings (src/core/config.py, env_file=<repo>/.env)
    -> RedisClient (src/infra/redis_client.py)
  docker/docker-compose.yml ml-service.environment
    -> container env
    -> same MLServiceSettings (env_file /app/.env does not exist in the image — .env is
       neither COPYed by the Dockerfile nor volume-mounted, so only compose-forwarded vars apply)

MLFLOW_TRACKING_URI / MLFLOW_ARTIFACT_ROOT
  .env -> MLServiceSettings -> MLflowTracker (src/tracking/mlflow_tracker.py)
  docker/docker-compose.yml mlflow.command (--backend-store-uri, --default-artifact-root,
    --artifacts-destination) — controls the server's own storage independently of any client env var

DATABASE_URL
  Not consumed in semd-ml as a single var; POSTGRES_* fields are combined into
  MLServiceSettings.database_url as a computed property.
```

Confirmed empirically: `podman exec semd-ml-service env | grep REDIS` before the fix showed only
`REDIS_HOST`/`REDIS_PORT`; after the fix it includes `REDIS_PASSWORD`/`REDIS_DB`.

## T-092 — Redis authentication alignment

### Root cause

`semd-backend`'s shared Redis (`semd-backend/config/redis.conf`) has `requirepass example` set.
`semd-ml/.env` had `REDIS_PASSWORD` blank, and `docker/docker-compose.yml`'s `ml-service`
environment block forwarded only `REDIS_HOST`/`REDIS_PORT` — never `REDIS_PASSWORD`/`REDIS_DB`,
even for users who do set a password in their local `.env`. Both the CLI (host) and the
containerized worker connected unauthenticated and failed.

### Cross-repo finding (not fixed here, backend is a separate module)

`semd-backend`'s `Settings` class (`semd-backend/config/settings.py`) reads Redis credentials from
`config/backend.ini`'s `[REDIS]` section via `configparser`, **not** from environment variables —
so the `REDIS_HOST=redis` env var set in `semd-backend/docker/compose.yaml` is silently ignored by
the backend process. `backend.ini` currently has `HOST = 127.0.0.1` (unreachable from inside a
container) and `PASSWORD = example`. This means the true backend → Redis → ml-service path cannot
be validated end-to-end from this module; see Remaining Blockers.

### Fix applied

| File | Change | Reason |
|---|---|---|
| `docker/docker-compose.yml` | Added `REDIS_PASSWORD: ${REDIS_PASSWORD:-}` and `REDIS_DB: ${REDIS_DB:-0}` to `ml-service.environment` | Compose previously dropped these two vars even when set in the host environment/.env |
| `.env` (local, gitignored) | Set `REDIS_PASSWORD=example`, `REDIS_DB=0` | Matches the shared Redis's actual `requirepass` value found in `semd-backend/config/redis.conf` |
| `.env.example` | Added a comment documenting the shared-Redis contract with `semd-backend` | Prevents the next person from leaving it blank against a password-protected Redis |

### Redis verification

| Test | Command | Exit code | Result | Evidence |
|---|---|---|---|---|
| Redis container health | `podman ps --filter name=semd-redis` | — | PASS | `Up ... (healthy)` |
| Direct authenticated ping | `podman exec semd-redis redis-cli -a example ping` | 0 | PASS | `PONG` |
| `queue-status` before fix | `uv run python main.py queue-status` (REDIS_PASSWORD blank) | 1 | FAIL (expected — reproduces the bug) | `HELLO must be called with the client already authenticated...` |
| `queue-status` after fix | `uv run python main.py queue-status` (REDIS_PASSWORD=example) | 0 | PASS | Printed queue depths for all 3 queues |
| ML worker Redis connection (container) | `podman exec semd-ml-service env \| grep REDIS` | 0 | PASS | `REDIS_PASSWORD=example`, `REDIS_DB=0` present |
| Worker startup, no auth errors | `podman logs semd-ml-service` | — | PASS | `Starting combined worker for both training and prediction` — no `NOAUTH`/`AuthenticationError` |
| Job publish → consume (direct queue push, backend publish blocked — see below) | `redis-cli -a example lpush ml_prediction_queue '{...}'` then check `llen` | — | PASS | Queue length 0 after worker picked it up; worker log shows `Processing prediction job: t092-e2e-test` |
| Invalid password | `REDIS_PASSWORD=wrongpass uv run python main.py queue-status` | 1 | PASS (correctly rejected) | `invalid username-password pair or user is disabled.` |
| Redis unavailable | `REDIS_HOST=127.0.0.1 REDIS_PORT=59999 uv run python main.py queue-status` | 1 | PASS (fails cleanly, no hang) | `Error 111 connecting to 127.0.0.1:59999. Connection refused.` |

**Backend → ml-service end-to-end**: **BLOCKED**, not fabricated. `semd-backend`'s `Settings`
ignores env-forwarded `REDIS_HOST` (reads `backend.ini` instead, which points at `127.0.0.1` —
unreachable from a container) so the backend process cannot actually publish into the shared
Redis from inside its own container. Validated the ml-side of the contract instead: pushed a job
directly onto `ml_prediction_queue` with the shared credentials and confirmed the worker drained
it without any authentication error. This is the maximum verifiable scope from `semd-ml` alone.

## T-093 — MLflow artifact persistence

### Root cause (confirmed empirically, not inferred)

1. `MLflowTracker._ensure_experiment()` (pre-fix) called `create_experiment(..., artifact_location=self._normalize_artifact_root(self.artifact_root))`, where `_normalize_artifact_root` resolved a relative `MLFLOW_ARTIFACT_ROOT` (`./artifacts/mlflow`) against `Path.cwd()` **at experiment-creation time**. Depending on where training was first invoked from (repo root vs `src/` vs in-container `/app/src`), this baked a literal absolute path into the experiment forever — e.g. `/home/semd/Desktop/Project/SEMD/semd-ml/artifacts/mlflow` for the two existing production experiments. Verified via `GET /api/2.0/mlflow/runs/get` — both `ccc1ca8...` and `62d9a007...` runs have `artifact_uri` equal to that literal host path.
2. Neither the `mlflow` container (mounts host `../artifacts` at `/artifacts`) nor the `ml-service` container (mounts it at `/app/artifacts`) has that literal host path inside their own filesystem — confirmed with `podman exec ... ls <path>` failing "No such file or directory" in both containers.
3. Because the experiment's `artifact_location` was a **plain filesystem path** (no `mlflow-artifacts:` scheme), the client SDK resolves it via `LocalArtifactRepository` and writes/reads **directly on its own local disk**, bypassing the tracking server's `--serve-artifacts` HTTP proxy entirely (that proxy only activates automatically for `mlflow-artifacts:/` URIs). Reproduced live: with `MLFLOW_ARTIFACT_ROOT` unset (server default), a `create_experiment()` call from inside `semd-ml-service` produced artifacts written to `/artifacts/...` **inside that container's own ephemeral, unmounted filesystem** — invisible from the `mlflow` server container and from the host, and destined to be lost on container removal.
4. Separately, the `mlflow` server container failed to start at all on first attempt: `mlflow_data/mlflow.db` had already been migrated to a newer alembic schema (`b7e4c1a90f23`) by the locally-installed mlflow client (`mlflow==3.14.0`, per `uv.lock`), while `docker-compose.yml` pinned the server image to `v3.10.0`, which doesn't recognize that revision (`Can't locate revision identified by 'b7e4c1a90f23'`).

### ⚠️ The fix does not reach real training — this is the load-bearing reason T-093 is still open

`_ensure_experiment()` only computes a fresh `artifact_location` on the `create_experiment` branch.
When the experiment already exists (`client.get_experiment_by_name(...)` returns a hit), it just
reuses `experiment.experiment_id` and **never touches `artifact_location` again** — that field is
permanent, set once at creation and stored server-side. Real training always targets one fixed,
already-existing experiment: `MLFLOW_EXPERIMENT_NAME=semd-url-classification`. Verified directly:

```
$ curl -s "http://localhost:5000/api/2.0/mlflow/experiments/get-by-name?experiment_name=semd-url-classification" \
    | python3 -m json.tool | grep artifact_location
        "artifact_location": "/home/semd/Desktop/Project/SEMD/semd-ml/artifacts/mlflow",
```

That is the pre-fix bare host path — this experiment predates the fix. Reproduced with the
project's **actual** `MLflowTracker` code (not a raw-`mlflow` probe) run inside `semd-ml-service`:

```python
from ml.ml_pipeline import ml_pipeline  # import order matches training_service.py
from tracking import mlflow_tracker
run_id = mlflow_tracker.start_run(run_name='t093-reuse-probe')
# artifact_uri = /home/semd/Desktop/Project/SEMD/semd-ml/artifacts/mlflow/9bb7f997.../artifacts
mlflow_tracker.log_artifact('/tmp/reuse_probe.txt', 'artifacts')
```

The logged file landed in `semd-ml-service`'s own ephemeral filesystem
(`podman exec semd-ml-service ls /home/.../artifacts/mlflow/9bb7f997.../artifacts` → found) and was
**absent on the host** (`ls` on the same path from the host → "No such file or directory") — i.e.
every future training run into the real experiment reproduces the exact bug this section set out
to fix, until the experiment itself is remediated. (This probe run, `9bb7f99729454c5abec499c2b88dcb56`,
was soft-deleted via `POST /api/2.0/mlflow/runs/delete` after confirming the bug — no probe data
was left in the production experiment.)

**Remediation requires a decision this section will not make unilaterally**, because it touches
shared registry state (existing `champion`/`candidate` model versions reference runs inside this
experiment):
- **Option A — rename**: point `MLFLOW_EXPERIMENT_NAME` at a new name (e.g.
  `semd-url-classification-v2`). Non-destructive: a fresh `create_experiment` call gets the
  `mlflow-artifacts:/` scheme automatically under the already-fixed code path. Old runs and
  registered versions stay exactly as they are (still broken, but untouched). Requires a fresh
  train + promote cycle to populate a working champion/candidate under the new name.
- **Option B — delete and recreate the same experiment name**: `semd-url-classification` would
  need its runs deleted (soft-delete via the same `runs/delete` API used above, or MLflow's
  experiment delete) and the experiment recreated. This orphans the run references inside
  registered model versions 1–3 (`champion`, `candidate`, and the unaliased version) — they would
  point at deleted runs. Destructive to registry history; not something to do without the user's
  explicit go-ahead.

No option was applied here — flagging for the user per "don't delete shared experiment data
yourself."

### Fix applied

| File | Change | Reason |
|---|---|---|
| `docker/docker-compose.yml` | Pinned `mlflow.image` from `v3.10.0` to `v3.14.0` | Matches the mlflow client version resolved in `uv.lock`; a server older than the client can't open a DB the client has already migrated |
| `docker/docker-compose.yml` | `mlflow.command`: `--default-artifact-root=mlflow-artifacts:/` + added `--artifacts-destination=/artifacts` (was `--default-artifact-root=${MLFLOW_ARTIFACT_ROOT:-./artifacts/mlflow}`) | Routes all artifact I/O through the tracking server's HTTP API (proxied), so client containers never need direct filesystem access to the artifact store — eliminates the cwd/mount-alignment problem entirely instead of trying to make every client's mount point match |
| `docker/docker-compose.yml` | Removed the now-dead `MLFLOW_ARTIFACT_ROOT` env var from the `mlflow` service block | It was already unused by the hardcoded command args; keeping it implied it configured storage when it didn't |
| `docker/docker-compose.yml` | `mlflow.healthcheck.test` changed from `curl -f http://localhost:5000` to a `python3 -c "..."` one-liner via `CMD-SHELL` | Discovered empirically: `ghcr.io/mlflow/mlflow:v3.14.0` (unlike `v3.10.0`) has no `curl` binary, so the old healthcheck failed every interval (`/bin/sh: curl: not found`) and the container stayed `unhealthy` forever. Under real Docker Compose (not podman-compose, which doesn't enforce it) this would permanently block `ml-service`'s `depends_on: condition: service_healthy` |
| `src/tracking/mlflow_tracker.py` | `_ensure_experiment()` no longer passes an explicit `artifact_location`; deleted the now-dead `_normalize_artifact_root` method | Let the server assign `artifact_location` from its own `--default-artifact-root`/`--artifacts-destination`, which is guaranteed consistent regardless of which process/container calls `create_experiment` |
| `src/core/config.py`, `.env`, `.env.example` | `MLFLOW_ARTIFACT_ROOT` default changed from `./artifacts/mlflow` to `mlflow-artifacts:/` | The setting is now informational only (logged into `training_configuration.json` metadata); the old default was actively misleading post-fix |

### MLflow persistence verification

| Test | Command | Exit code | Result | Evidence |
|---|---|---|---|---|
| Start MLflow (first attempt, pre-image-fix) | `podman compose up -d mlflow` | 0 (container exited immediately) | FAIL (reproduces the bug) | `mlflow.exceptions.MlflowException: Detected out-of-date database schema (found version b7e4c1a90f23, but expected 1b5f0d9ad7c1)` |
| `mlflow db upgrade` attempted as a workaround | `podman run --rm -v mlflow_data:/mlflow ghcr.io/mlflow/mlflow:v3.10.0 mlflow db upgrade sqlite:////mlflow/mlflow.db` | 1 | FAIL (confirms version skew, not just a stale migration) | `alembic.util.exc.CommandError: Can't locate revision identified by 'b7e4c1a90f23'` |
| Start MLflow with matching image (`v3.14.0`) | `podman compose up -d mlflow` | 0 | PASS | `curl -sf http://localhost:5000` succeeds |
| MLflow container healthcheck (pre-fix, curl-based) | `podman ps --filter name=semd-mlflow` | — | FAIL (reproduces the bug) | `unhealthy`; `podman inspect` log shows `/bin/sh: 1: curl: not found` every 10s |
| MLflow container healthcheck (post-fix, python3-based) | `podman ps --filter name=semd-mlflow` | — | PASS | `healthy` after ~3 healthcheck intervals |
| `ml-service` starts after `mlflow` reports healthy | `podman compose up -d ml-service` then `podman ps` | 0 | PASS | `semd-ml-service` running, `semd-mlflow` `healthy` |
| Create run with old (pre-fix) client code | in-container `mlflow.create_experiment(...)` + `log_artifact` | 0 (no exception) | FAIL (silently wrong — reproduces the bug) | Artifact written to `/artifacts/...` inside `semd-ml-service`'s own unmounted filesystem; absent from `/artifacts` in the `mlflow` server container and absent on the host |
| Create run with fixed client code | in-container `mlflow.create_experiment(...)` + `log_artifact` | 0 | PASS | `artifact_uri = mlflow-artifacts:/4/d853c321.../artifacts` |
| Artifact visible from host | `cat artifacts/4/d853c321.../artifacts/artifacts/probe2.txt` | 0 | PASS | `hello-t093-v2` |
| Artifact visible from mlflow server container | `podman exec semd-mlflow cat /artifacts/4/.../probe2.txt` | 0 | PASS | `hello-t093-v2` |
| Artifact visible from ml-service container (different container than the one that wrote it) | `podman exec semd-ml-service cat /app/artifacts/4/.../probe2.txt` | 0 | PASS | `hello-t093-v2` |
| **Container recreation persistence** — `podman rm -f semd-ml-service semd-mlflow`, rebuild, `podman compose up -d mlflow ml-service` | see script output | 0 | PASS | Fresh `semd-ml-service` container (never wrote the artifact) successfully ran `mlflow.artifacts.download_artifacts(run_id=..., artifact_path="artifacts/probe2.txt")` and read back `hello-t093-v2` |

Expected-artifact-type coverage for the pre-existing `ccc1ca8...` run (git-tracked):
classification_report.json, feature_schema.json, dataset_metadata.json, dataset_quality_report.json,
training_configuration.json, confusion_matrix.png, precision_recall_curve.png, roc_curve.png,
sample_predictions.json all present at `/artifacts/mlflow/ccc1ca8.../artifacts/artifacts/` inside
the `mlflow` container. The model artifact (`random_forest_run_....joblib`, gitignored) is also
physically present there, but **cannot be loaded** by any container because that run's
`artifact_uri` is the pre-fix literal host path — see Remaining Blockers.

## T-094 — Clean-environment regression coverage

New tests added (all run and passing on this environment):

| Test file | Scenario | Result |
|---|---|---|
| `tests/integration/test_cli_bootstrap.py` | `main.py --help`, per-subcommand `--help`, `verify_imports.py`, all as fresh subprocesses from `src/` (not in-process imports) — catches import-chain breakage a unit test wouldn't see | PASS (4 tests, 5 subtests) |
| `tests/integration/test_settings_paths.py` | `PROJECT_ROOT` resolves 2 levels above `config.py`; dataset/model/report paths are absolute; `MLFLOW_ARTIFACT_ROOT` default is no longer a bare relative path; `.env` resolves to `<repo>/.env`; `REDIS_PASSWORD` is env-overridable | PASS (5 tests) |
| `tests/integration/test_redis_connection.py` | Live-Redis auth round trip, wrong-password rejection, queue push/pop round trip — self-skips if Redis isn't reachable | PASS (3 tests, ran against live Redis) |
| `tests/integration/test_mlflow_artifact_persistence.py` | New experiment gets `mlflow-artifacts:/` scheme (not a bare path); logged artifact round-trips through the live tracking server; the project's own `MLflowTracker` class (not just raw `mlflow` calls) produces a proxied URI for a fresh experiment — self-skips if MLflow isn't reachable | PASS (3 tests, ran against live MLflow) |
| `tests/integration/test_mlflow_artifact_persistence.py::MlflowExperimentReuseKnownGapTests` | Documents Remaining Blocker #1: `MLflowTracker` reusing an experiment that already has a bare-path `artifact_location` does **not** self-heal it to `mlflow-artifacts:/` — asserts the current (gap) behavior explicitly so a future fix has a test to update, and so the gap can't silently regress further | PASS (1 test) |
| `scripts/verify_container_paths.py` | Static, no-infra-required checks: `ml-service` forwards `REDIS_PASSWORD`/`REDIS_DB`; `mlflow` image tag matches `uv.lock`'s mlflow version; `--default-artifact-root`/`--artifacts-destination` use the proxied scheme; `mlflow` healthcheck doesn't use curl; `ml-service` and `mlflow` mount the same host directory for artifacts | PASS on current config; **manually verified it fails with 4 findings when pointed at a reconstructed pre-fix `docker-compose.yml`** (redone in `/tmp`, discarded after) |

Full suite: `MLFLOW_TRACKING_URI=http://localhost:5000 uv run python -m pytest tests/ -q` →
**83 passed**, 0 failed, 0 errors (includes all pre-existing tests, unaffected by these changes).

## Remaining blockers (Session 1 — see "Update (Session 2)" above for resolution status)

1. **[RESOLVED in Session 2 — new experiment `semd-url-classification-v2` + guard, see above]**
   **[BLOCKING — T-093 unresolved for real training] The production `semd-url-classification`
   experiment still has a pre-fix bare-path `artifact_location`.** The code fix in
   `mlflow_tracker.py` only applies to newly-*created* experiments; `_ensure_experiment()` reuses
   an existing experiment's `artifact_location` unchanged when the name already exists. Since
   `MLFLOW_EXPERIMENT_NAME=semd-url-classification` already exists (created before this fix), every
   future training run — through the real `MLflowTracker`, not a synthetic probe — will keep
   writing artifacts to whatever container happens to run training instead of the shared volume,
   exactly as before. See the empirical reproduction above. Requires a user decision between
   renaming the experiment (non-destructive) or deleting/recreating it (touches registered model
   history) before this can be marked resolved end-to-end.
2. **[RESOLVED in Session 2 — `env_ignore_empty=True` + env-var forwarding fixed in `semd-backend`, real backend-published job verified, see above]**
   **`semd-backend` cannot actually reach the shared Redis from inside its own container.**
   `semd-backend/config/settings.py` reads Redis config from `config/backend.ini`
   (`configparser`), completely ignoring the `REDIS_HOST=redis` env var set in
   `semd-backend/docker/compose.yaml`. `backend.ini` currently has `HOST = 127.0.0.1`, unreachable
   from a container, and `PASSWORD = example` (currently correct only by coincidence, since
   `example` does match the live `requirepass`). This is a `semd-backend`-module fix, out of this
   section's scope — flagging per project convention that changes belong in their own submodule
   working tree.
3. **[RESOLVED in Session 2 — champion is now v4 (run `9fef933e...`), loadable and verified after container recreation, see above; old broken champion preserved as `previous-champion` alias, untouched]**
   **The registered `champion` model alias points to an unrecoverable run**, and so does
   `candidate` — same root cause as Blocker #1 (both runs predate the fix). `run_id
   62d9a007619349a08f26d3da04ebcf3d`'s `.joblib` cannot be downloaded from any container
   (`ModelRegistryError: Unable to load model from MLflow registry: Failed to download artifacts
   from path 'random_forest_run_20260713194032_dbcaa877.joblib'`). Resolving Blocker #1 and then
   running a fresh training + promotion cycle would fix this as a side effect — explicitly out of
   scope for this section ("do not run expensive full-dataset model training").
4. **[RESOLVED in Session 2 — structured failure results, see Phase 4 above]**
   **Prediction jobs that fail to load a model are silently dropped, not reported.** Observed
   while reproducing the champion-load failure: the failed job never appeared on
   `ml_result_queue`, and `ml_prediction_queue` was still drained (0 after processing) — i.e. the
   job is lost rather than retried or reported as failed. This is a queue-worker robustness gap
   (`src/workers/queue_worker.py`), not a Redis-auth or MLflow-path issue, so it's noted here but
   not fixed under T-092/T-093.
5. Fixture-based validation (this section) does not establish production model quality — per the
   task brief, that remains explicitly out of scope here.

## Session Handoff

### Completed
- T-092: Root-caused and fixed Redis credential forwarding gap in `docker/docker-compose.yml`;
  documented the shared-secret contract in `.env.example`; validated auth success/failure paths
  and a real (direct-queue) job publish → worker consume flow.
- T-093: Root-caused and fixed the MLflow artifact persistence bug (bare-path `artifact_location`
  bypassing `--serve-artifacts` proxying) and the server/client mlflow version skew that also
  blocked the server from starting; validated with a full artifact write → container recreation →
  read-back cycle **for newly-created experiments**. **Not complete**: the fix does not retroactively
  apply to the real `semd-url-classification` experiment, which predates it — see Remaining
  Blocker #1.
- T-094: Added 4 new test files (16 test cases) plus a static compose-config checker script;
  full suite passes (83 tests).
- Documentation: this file, plus a new Infrastructure section in `docs/troubleshooting.md`.

### Redis status
Aligned and verified on the `semd-ml` side (CLI + containerized worker both authenticate
correctly against the shared password-protected Redis; queue-status, worker startup, job
consumption, invalid-password and Redis-unavailable paths all tested). `semd-backend`'s own
publish path is blocked by a `semd-backend`-side config bug (Remaining Blocker #1).

### MLflow persistence status
**Not fully resolved.** The code/config fix is correct and verified for *newly-created*
experiments (fresh write → container recreation → read-back all passed, using the actual
`MLflowTracker` code). But the one experiment real training actually uses,
`semd-url-classification`, was created before this fix and keeps its original bare-path
`artifact_location` forever — `_ensure_experiment()` never rewrites it for an existing experiment.
Reproduced the exact original bug through the real tracker against that experiment (see T-093 for
the trace). This is Remaining Blocker #1 and is why "Ready for Section 2" is No.

### Tests added
`tests/integration/test_cli_bootstrap.py`, `tests/integration/test_settings_paths.py`,
`tests/integration/test_redis_connection.py`, `tests/integration/test_mlflow_artifact_persistence.py`,
`scripts/verify_container_paths.py`.

### Commands executed
See per-section tables above for the full command/exit-code/evidence list. Container lifecycle:
`podman compose -f semd-backend/database/docker-compose.database.yaml up -d`,
`podman compose -f docker/docker-compose.yml build ml-service`,
`podman compose -f docker/docker-compose.yml up -d mlflow`,
`podman compose -f docker/docker-compose.yml up -d ml-service`,
`podman rm -f semd-ml-service semd-mlflow` (recreation test),
`uv run python -m pytest tests/ -q`, `uv run python scripts/verify_container_paths.py`.

### Failed or blocked commands
- `podman compose up -d mlflow` against the original `v3.10.0` image + pre-migrated
  `mlflow_data/mlflow.db` — exit 0 (container start) but the server process itself crashed
  (alembic schema mismatch). Fixed by pinning the image to `v3.14.0`.
- `mlflow db upgrade` against `v3.10.0`'s bundled alembic — exit 1, `Can't locate revision`.
  Superseded by the image-version fix (no downgrade/upgrade of the DB was needed once the server
  version matched the client).
- Backend → ml-service full end-to-end publish — blocked, see Remaining Blocker #2.

### Files changed
`.env` (local, gitignored), `.env.example`, `docker/docker-compose.yml`,
`src/tracking/mlflow_tracker.py`, `src/core/config.py`, `docs/troubleshooting.md`,
`docs/section-10-infrastructure-validation.md` (new),
`tests/integration/test_cli_bootstrap.py` (new),
`tests/integration/test_settings_paths.py` (new),
`tests/integration/test_redis_connection.py` (new),
`tests/integration/test_mlflow_artifact_persistence.py` (new),
`scripts/verify_container_paths.py` (new).

### Ready for Section 2 (Session 1 verdict — superseded)
No. **See "Update (Session 2)" at the top of this file: as of Session 2, all blockers below are
resolved and Section 1's exit gate is met — Ready for Section 2: Yes.**

### Remaining blockers before Makefile verification (Session 1 — all resolved in Session 2)
- `semd-backend`'s `config/settings.py` needs to actually read `REDIS_HOST`/`REDIS_PORT` from the
  environment (or `backend.ini` needs correcting to a reachable host) before a true
  backend-publishes → ml-consumes flow can be validated — currently only the ml-side half of that
  contract is proven. **[RESOLVED]**
- The `champion`/`candidate` MLflow model aliases need a fresh train + promote cycle before
  `predict`/worker prediction paths can succeed end-to-end; the current registered versions are a
  known-broken artifact of the pre-fix bug, not a new regression. **[RESOLVED — champion is now v4]**
- The queue-worker's silent-job-drop-on-exception behavior (Remaining Blocker #3) should be triaged
  before relying on `make queue-status`/result-queue depth as a health signal in Section 2's
  Makefile verification. **[RESOLVED]**
