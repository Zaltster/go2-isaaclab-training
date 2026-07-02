#!/usr/bin/env python3
"""Run the frozen-WTW hierarchical navigation trainer."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    target = ["/workspace/IsaacLab/isaaclab.sh", "-p", "scripts/train_wtw_hierarchical_nav.py", *sys.argv[1:]]
    print("starting:", " ".join(target), flush=True)
    return subprocess.call(target)


if __name__ == "__main__":
    raise SystemExit(main())
