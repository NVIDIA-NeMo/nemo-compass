# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run Compass with a hard time limit that holds even if Hermes exits.

The plugin starts this script as the run's detached process; it starts Compass, kills it at the
deadline, and records how it ended. Standard library only, so it runs on Hermes's bare interpreter.

Usage: python watchdog.py <timeout_s> <exit_file> -- <compass command...>
"""

from __future__ import annotations

import json
import subprocess
import sys


def main(argv: list[str]) -> int:
    timeout_s, exit_file = int(argv[1]), argv[2]
    cmd = argv[argv.index("--") + 1 :]
    proc = subprocess.Popen(cmd)  # inherits cwd, env and the log file from the plugin
    timed_out = False
    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        proc.kill()
        rc = proc.wait()
    with open(exit_file, "w", encoding="utf-8") as f:
        json.dump({"rc": rc, "timed_out": timed_out}, f)
    return 0 if rc == 0 and not timed_out else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
