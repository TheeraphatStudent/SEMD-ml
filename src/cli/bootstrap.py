from __future__ import annotations

import os
import shutil


def ensure_env_file(root: str = ".") -> None:
    env_path = os.path.join(root, ".env")
    example_path = os.path.join(root, ".env.example")
    if not os.path.exists(env_path) and os.path.exists(example_path):
        shutil.copy(example_path, env_path)


def run() -> int:
    ensure_env_file()
    from cli.main import main

    return main()
