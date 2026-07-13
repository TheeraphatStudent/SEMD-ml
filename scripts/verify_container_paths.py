#!/usr/bin/env python3
"""Static regression check for docker/docker-compose.yml + uv.lock alignment.

Catches, without needing any container running, the classes of bug found
during T-092/T-093 infrastructure validation:

  1. Compose forwards REDIS_HOST/REDIS_PORT into ml-service but silently drops
     REDIS_PASSWORD/REDIS_DB -- the worker then connects unauthenticated even
     when .env has a real password configured, and fails with NOAUTH/
     AuthenticationError against a Redis that requires one.
  2. The mlflow server image tag is pinned independently of the mlflow client
     version resolved in uv.lock. A server older than the client refuses to
     open a database the client has already migrated (alembic "Can't locate
     revision" error) -- and even when it starts, a bare filesystem
     --default-artifact-root defeats --serve-artifacts proxying so client
     containers write artifacts to their own unmounted disk instead of the
     shared volume.
  3. The mlflow server's healthcheck used curl, which isn't installed in the
     v3.14.0 image -- the container never reports healthy, which would block
     depends_on: condition: service_healthy under real Docker Compose
     (podman-compose doesn't enforce this, which is how it went unnoticed).

Run: uv run python scripts/verify_container_paths.py
Exit code 0 = all checks passed. Exit code 1 = at least one check failed
(details printed to stderr).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPOSE_PATH = PROJECT_ROOT / "docker" / "docker-compose.yml"
UV_LOCK_PATH = PROJECT_ROOT / "uv.lock"

failures: list[str] = []


def fail(message: str) -> None:
    failures.append(message)


def load_compose() -> dict:
    return yaml.safe_load(COMPOSE_PATH.read_text(encoding="utf-8"))


def resolved_mlflow_client_version() -> str | None:
    text = UV_LOCK_PATH.read_text(encoding="utf-8")
    match = re.search(r'name = "mlflow"\nversion = "([^"]+)"', text)
    return match.group(1) if match else None


def check_redis_env_forwarded(compose: dict) -> None:
    env = compose["services"]["ml-service"].get("environment", {})
    for key in ("REDIS_PASSWORD", "REDIS_DB"):
        if key not in env:
            fail(
                f"docker/docker-compose.yml: ml-service.environment is missing '{key}'. "
                "REDIS_HOST/REDIS_PORT alone are not enough -- an authenticated Redis "
                "will reject the worker with NOAUTH/AuthenticationError."
            )


def check_mlflow_image_matches_client(compose: dict) -> None:
    image = compose["services"]["mlflow"]["image"]
    image_version_match = re.search(r":v?([0-9]+\.[0-9]+\.[0-9]+)$", image)
    client_version = resolved_mlflow_client_version()
    if image_version_match is None:
        fail(f"docker/docker-compose.yml: mlflow.image '{image}' has no parseable version tag")
        return
    if client_version is None:
        fail("uv.lock: could not resolve the locked mlflow package version")
        return
    image_version = image_version_match.group(1)
    if image_version != client_version:
        fail(
            f"docker/docker-compose.yml pins mlflow server image v{image_version}, but "
            f"uv.lock resolves the mlflow client to v{client_version}. A server older than "
            "the client can refuse to open a DB the client already migrated "
            "(alembic 'Can't locate revision' error)."
        )


def check_artifact_root_uses_proxied_scheme(compose: dict) -> None:
    mlflow_command = compose["services"]["mlflow"]["command"]
    joined = " ".join(mlflow_command)
    if "--default-artifact-root=mlflow-artifacts:/" not in joined:
        fail(
            "docker/docker-compose.yml: mlflow server command's --default-artifact-root is "
            "not 'mlflow-artifacts:/'. A bare filesystem path defeats --serve-artifacts "
            "proxying: clients write directly to their own local (often unmounted) disk "
            "instead of the shared artifact volume."
        )
    if "--artifacts-destination=" not in joined:
        fail(
            "docker/docker-compose.yml: mlflow server command is missing "
            "--artifacts-destination; without it --default-artifact-root=mlflow-artifacts:/ "
            "has no backing physical storage path."
        )


def check_mlflow_healthcheck_does_not_use_curl(compose: dict) -> None:
    healthcheck = compose["services"]["mlflow"].get("healthcheck", {})
    test = healthcheck.get("test", [])
    joined = " ".join(test) if isinstance(test, list) else str(test)
    if "curl" in joined:
        fail(
            "docker/docker-compose.yml: mlflow.healthcheck.test uses curl, which is not "
            "installed in ghcr.io/mlflow/mlflow:v3.14.0 (it was in v3.10.0). The healthcheck "
            "fails every interval and the container never reports healthy, permanently "
            "blocking any depends_on: condition: service_healthy consumer under real Docker "
            "Compose."
        )


def check_ml_service_and_mlflow_share_artifact_host_dir(compose: dict) -> None:
    def host_dir_for(service: str, container_path: str) -> str | None:
        for mount in compose["services"][service].get("volumes", []):
            host, _, container = mount.partition(":")
            if container.rstrip("/") == container_path.rstrip("/"):
                return host
        return None

    ml_service_host = host_dir_for("ml-service", "/app/artifacts")
    mlflow_host = host_dir_for("mlflow", "/artifacts")
    if ml_service_host is None:
        fail("docker/docker-compose.yml: ml-service has no volume mounted at /app/artifacts")
    if mlflow_host is None:
        fail("docker/docker-compose.yml: mlflow has no volume mounted at /artifacts")
    if ml_service_host and mlflow_host and ml_service_host != mlflow_host:
        fail(
            f"docker/docker-compose.yml: ml-service mounts host '{ml_service_host}' at "
            f"/app/artifacts but mlflow mounts host '{mlflow_host}' at /artifacts -- "
            "these must be the same host directory for any direct-filesystem artifact "
            "access (e.g. joblib downloads) to see the same files."
        )


def main() -> int:
    compose = load_compose()
    check_redis_env_forwarded(compose)
    check_mlflow_image_matches_client(compose)
    check_artifact_root_uses_proxied_scheme(compose)
    check_mlflow_healthcheck_does_not_use_curl(compose)
    check_ml_service_and_mlflow_share_artifact_host_dir(compose)

    if failures:
        print(f"FAILED: {len(failures)} container path check(s) failed\n", file=sys.stderr)
        for message in failures:
            print(f"  - {message}\n", file=sys.stderr)
        return 1

    print("OK: all container path checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
