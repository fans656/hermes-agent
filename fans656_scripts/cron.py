#!/usr/bin/env python3
"""Cron helpers — tail output, check status, etc.

Usage::

    ./cli cron tail <job_id>    # follow latest output, auto-rotate
"""

import os
import sys
import time
from pathlib import Path

OUTPUT_ROOT = Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes"))) / "cron" / "output"


def _cmd_tail(job_id: str) -> None:
    output_dir = OUTPUT_ROOT / job_id
    if not output_dir.exists():
        print(f"Output dir not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    last_file = ""
    while True:
        files = sorted(output_dir.glob("*.md"),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        latest = files[0] if files else None

        if latest is None:
            time.sleep(10)
            continue

        latest_str = str(latest)
        is_new = latest_str != last_file
        if is_new:
            print(f"--- following: {latest.name} ---")
            last_file = latest_str

        try:
            with open(latest, "r") as f:
                if is_new:
                    f.seek(0)  # start from beginning of new file
                else:
                    f.seek(0, os.SEEK_END)  # tail existing file
                while True:
                    line = f.readline()
                    if not line:
                        break
                    print(line, end="")
        except FileNotFoundError:
            pass
        time.sleep(1)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: cron <subcommand> [...]", file=sys.stderr)
        print("  tail <job_id>   follow latest output, auto-rotate", file=sys.stderr)
        sys.exit(0 if len(sys.argv) >= 2 else 1)

    sub = sys.argv[1]
    if sub == "tail":
        if len(sys.argv) < 3:
            print("Usage: cron tail <job_id>", file=sys.stderr)
            sys.exit(1)
        _cmd_tail(sys.argv[2])
    else:
        print(f"Unknown subcommand: {sub}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
