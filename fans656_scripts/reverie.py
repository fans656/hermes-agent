#!/usr/bin/env python3
"""Reverie — manage Evo's wandering sessions in the 小径 room.

Subcommands::

    reverie                  Send initial stimulus to start a wandering session
    reverie info             Show current state (frozen status, last stimulus)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote
from typing import Any, Dict, Optional

# ── Paths ───────────────────────────────────────────────────────────────

_HERMES_HOME = Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes")))
_STATE_DIR = _HERMES_HOME / "cron" / "reverie"
_STATE_FILE = _STATE_DIR / "state.json"
_SECRETS_FILE = _HERMES_HOME / "secrets" / "matrix-evo-sub.json"
_UTC8 = timezone(timedelta(hours=8))

# ── Constants ──────────────────────────────────────────────────────────

ROOM_ID = "!ChiGrdKFbk3vmycE67W9d2Q2r37sCY-1tZDN5F2QRJk"

# Stimulus preamble — gives context about who the subconscious is
STIMULUS_BASE = (
    "我是你的潜意识。你现在在自由 wandering 时间。\n"
    "\n"
    "选你感兴趣的事做。"
)

MAX_ROUNDS = 15  # max rounds before auto-freeze
ROUND_TIMEOUT = 120  # seconds to wait for Evo's response

# ── State ──────────────────────────────────────────────────────────────

def _read_state() -> Dict[str, Any]:
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "frozen": False,
        "stimulus_count": 0,
        "last_stimulus": None,
        "last_stimulus_at": None,
        "last_evo_response": None,
        "last_evo_response_at": None,
        "frozen_at": None,
        "frozen_round": 0,
        "terminated_at": None,
        "last_termination_message": None,
    }


def _write_state(state: Dict[str, Any]) -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))


# ── Matrix API ─────────────────────────────────────────────────────────

def _get_sub_token() -> str:
    """Read the evo-subconscious Matrix access token."""
    if not _SECRETS_FILE.exists():
        print("Error: secrets file not found at", _SECRETS_FILE, file=sys.stderr)
        print("Run: hermes -p evo-subconscious chat to set up credentials first.", file=sys.stderr)
        sys.exit(1)
    secrets = json.loads(_SECRETS_FILE.read_text(encoding="utf-8"))
    return secrets["access_token"]


def _send_matrix_message(token: str, body: str, *, formatted_body: Optional[str] = None) -> str:
    """Send a message to the 小径 room as evo-subconscious. Returns event_id."""
    homeserver = "https://matrix.fans656.cn"
    room_enc = quote(ROOM_ID, safe="")
    txn_id = f"reverie_{int(time.time() * 1000)}_{os.urandom(4).hex()}"

    payload: Dict[str, Any] = {
        "msgtype": "m.text",
        "body": body,
    }
    if formatted_body:
        payload["format"] = "org.matrix.custom.html"
        payload["formatted_body"] = formatted_body

    req = Request(
        f"{homeserver}/_matrix/client/v3/rooms/{room_enc}/send/m.room.message/{txn_id}",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
        },
        method="PUT",
    )
    with urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
        event_id = data.get("event_id", "")
        return event_id


# ── Commands ───────────────────────────────────────────────────────────

def cmd_send_stimulus(args: list[str]) -> None:
    """Send initial stimulus to start a wandering session."""
    token = _get_sub_token()
    state = _read_state()

    if state.get("frozen") and not args:
        state["frozen"] = False
        state["frozen_at"] = None
        state["frozen_round"] = 0
        print("  unfrozen (previous session was frozen)")

    stimulus = STIMULUS_BASE
    event_id = _send_matrix_message(token, stimulus)

    state["stimulus_count"] = state.get("stimulus_count", 0) + 1
    state["last_stimulus"] = stimulus
    state["last_stimulus_at"] = datetime.now(tz=_UTC8).isoformat()
    state["last_stimulus_event_id"] = event_id
    _write_state(state)

    print("  stimulus sent:")
    for line in stimulus.strip().split("\n"):
        print(f"    {line}")
    print(f"  event: {event_id[:20]}...")


def cmd_info(args: list[str]) -> None:
    """Show current state."""
    state = _read_state()

    print("  stimulus content:")
    for line in STIMULUS_BASE.strip().split("\n"):
        print(f"    {line}")

    print()
    if state.get("frozen"):
        print("  status: FROZEN")
        print(f"  frozen at round: {state.get('frozen_round', '?')}")
        print(f"  frozen since: {state.get('frozen_at', '?')}")
    else:
        print("  status: active")

    print(f"  stimulus count: {state.get('stimulus_count', 0)}")
    if state.get("last_stimulus"):
        print(f"  last stimulus: {state['last_stimulus'][:120]}...")
        print(f"  last stimulus at: {state.get('last_stimulus_at', '?')}")
    else:
        print("  last stimulus: (none yet)")

    if state.get("last_evo_response"):
        print(f"  last Evo response: {state['last_evo_response'][:120]}...")
        print(f"  last Evo response at: {state.get('last_evo_response_at', '?')}")
    else:
        print("  last Evo response: (none yet)")

    if state.get("terminated_at"):
        print(f"\n  ── last termination ──")
        print(f"  terminated at: {state['terminated_at']}")
        _last_msg = state.get("last_termination_message", "")
        if _last_msg:
            print("  last termination message (head):")
            for _line in _last_msg.strip().split("\n")[:10]:
                print(f"    {_line}")
            if len(_last_msg.split("\n")) > 10:
                print("    ...")

    print(f"\n  room: {ROOM_ID}")


# ── Main ───────────────────────────────────────────────────────────────

def print_usage() -> None:
    print("Usage: reverie [info]")
    print()
    print("  reverie          Send initial stimulus to start wandering")
    print("  reverie info     Show current state")


def main() -> None:
    sub = sys.argv[1] if len(sys.argv) > 1 else None

    if sub == "info":
        cmd_info(sys.argv[2:])
    elif sub is None:
        cmd_send_stimulus(sys.argv[2:])
    else:
        print(f"Unknown subcommand: {sub}", file=sys.stderr)
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
