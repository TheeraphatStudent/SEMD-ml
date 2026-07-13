# Section 1 — Infrastructure Validation (T-092, T-093, T-094)

Scope: `semd-ml` only. Findings on `semd-backend` are documented but not fixed (out of module
scope; see Remaining Blockers).

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
| `tests/integration/test_mlflow_artifact_persistence.py` | New experiment gets `mlflow-artifacts:/` scheme (not a bare path); logged artifact round-trips through the live tracking server — self-skips if MLflow isn't reachable | PASS (2 tests, ran against live MLflow) |
| `scripts/verify_container_paths.py` | Static, no-infra-required checks: `ml-service` forwards `REDIS_PASSWORD`/`REDIS_DB`; `mlflow` image tag matches `uv.lock`'s mlflow version; `--default-artifact-root`/`--artifacts-destination` use the proxied scheme; `ml-service` and `mlflow` mount the same host directory for artifacts | PASS on current config; **manually verified it fails with 4 findings when pointed at a reconstructed pre-fix `docker-compose.yml`** (redone in `/tmp`, discarded after) |

Full suite: `MLFLOW_TRACKING_URI=http://localhost:5000 uv run python -m pytest tests/ -q` →
**81 passed**, 0 failed, 0 errors (includes all pre-existing tests, unaffected by these changes).

## Remaining blockers

1. **`semd-backend` cannot actually reach the shared Redis from inside its own container.**
   `semd-backend/config/settings.py` reads Redis config from `config/backend.ini`
   (`configparser`), completely ignoring the `REDIS_HOST=redis` env var set in
   `semd-backend/docker/compose.yaml`. `backend.ini` currently has `HOST = 127.0.0.1`, unreachable
   from a container, and `PASSWORD = example` (currently correct only by coincidence, since
   `example` does match the live `requirepass`). This is a `semd-backend`-module fix, out of this
   section's scope — flagging per project convention that changes belong in their own submodule
   working tree.
2. **The registered `champion` model alias points to an unrecoverable run.** `run_id
   62d9a007619349a08f26d3da04ebcf3d`'s `artifact_uri` is a pre-fix literal host path; its
   `.joblib` cannot be downloaded from any container (`ModelRegistryError: Unable to load model
   from MLflow registry: Failed to download artifacts from path
   'random_forest_run_20260713194032_dbcaa877.joblib'`). The same is true for the `candidate`
   alias (`ccc1ca8...`, same pre-fix bug). Both need a fresh training run + promotion under the
   fixed tracker to produce a genuinely loadable champion — explicitly out of scope for this
   section ("do not run expensive full-dataset model training").
3. **Prediction jobs that fail to load a model are silently dropped, not reported.** Observed
   while reproducing the champion-load failure: the failed job never appeared on
   `ml_result_queue`, and `ml_prediction_queue` was still drained (0 after processing) — i.e. the
   job is lost rather than retried or reported as failed. This is a queue-worker robustness gap
   (`src/workers/queue_worker.py`), not a Redis-auth or MLflow-path issue, so it's noted here but
   not fixed under T-092/T-093.
4. Fixture-based validation (this section) does not establish production model quality — per the
   task brief, that remains explicitly out of scope here.

## Session Handoff

### Completed
- T-092: Root-caused and fixed Redis credential forwarding gap in `docker/docker-compose.yml`;
  documented the shared-secret contract in `.env.example`; validated auth success/failure paths
  and a real (direct-queue) job publish → worker consume flow.
- T-093: Root-caused and fixed the MLflow artifact persistence bug (bare-path `artifact_location`
  bypassing `--serve-artifacts` proxying) and the server/client mlflow version skew that also
  blocked the server from starting; validated with a full artifact write → container recreation →
  read-back cycle.
- T-094: Added 4 new test files (14 test cases) plus a static compose-config checker script;
  full suite passes (81 tests).
- Documentation: this file, plus a new Infrastructure section in `docs/troubleshooting.md`.

### Redis status
Aligned and verified on the `semd-ml` side (CLI + containerized worker both authenticate
correctly against the shared password-protected Redis; queue-status, worker startup, job
consumption, invalid-password and Redis-unavailable paths all tested). `semd-backend`'s own
publish path is blocked by a `semd-backend`-side config bug (Remaining Blocker #1).

### MLflow persistence status
Fixed and verified end-to-end including full container recreation. Two pre-existing model
versions (`champion`, `candidate` aliases) remain unrecoverable because they were created before
this fix (Remaining Blocker #2) — new training runs will not have this problem.

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
- Backend → ml-service full end-to-end publish — blocked, see Remaining Blocker #1.

### Files changed
`.env` (local, gitignored), `.env.example`, `docker/docker-compose.yml`,
`src/tracking/mlflow_tracker.py`, `src/core/config.py`, `docs/troubleshooting.md`,
`docs/section-10-infrastructure-validation.md` (new),
`tests/integration/test_cli_bootstrap.py` (new),
`tests/integration/test_settings_paths.py` (new),
`tests/integration/test_redis_connection.py` (new),
`tests/integration/test_mlflow_artifact_persistence.py` (new),
`scripts/verify_container_paths.py` (new).

### Ready for Section 2
No.

### Remaining blockers before Makefile verification
- `semd-backend`'s `config/settings.py` needs to actually read `REDIS_HOST`/`REDIS_PORT` from the
  environment (or `backend.ini` needs correcting to a reachable host) before a true
  backend-publishes → ml-consumes flow can be validated — currently only the ml-side half of that
  contract is proven.
- The `champion`/`candidate` MLflow model aliases need a fresh train + promote cycle before
  `predict`/worker prediction paths can succeed end-to-end; the current registered versions are a
  known-broken artifact of the pre-fix bug, not a new regression.
- The queue-worker's silent-job-drop-on-exception behavior (Remaining Blocker #3) should be triaged
  before relying on `make queue-status`/result-queue depth as a health signal in Section 2's
  Makefile verification.
