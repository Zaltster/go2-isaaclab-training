#!/usr/bin/env python3
"""Import local Go2 tasks, then hand off to IsaacLab's RSL-RL trainer."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


def find_isaaclab_root() -> Path:
    candidates = [
        os.environ.get("ISAACLAB_ROOT"),
        "/workspace/IsaacLab",
        "/IsaacLab",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate)
        if (root / "isaaclab.sh").exists():
            return root
    raise FileNotFoundError("Set ISAACLAB_ROOT to the directory containing isaaclab.sh.")


def main() -> None:
    # Import side effect: registers local Gym task ids before train.py reads --task.
    import go2_isaaclab_tasks  # noqa: F401

    root = find_isaaclab_root()
    train_script = root / "scripts" / "reinforcement_learning" / "rsl_rl" / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Missing IsaacLab RSL-RL train script: {train_script}")

    sys.path.insert(0, str(train_script.parent))
    sys.argv = [str(train_script), *sys.argv[1:]]
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
