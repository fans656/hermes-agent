#!/usr/bin/env python3
"""Diary tools — probe conversations and write diary entries.

Subcommands::

    diary.py write [-n N] [--dryrun] [--verbose] [--force] [<sid>]
    diary.py fix
    diary.py status
    diary.py clear
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Paths ───────────────────────────────────────────────────────────────
_SCRIPTS_DIR = Path(__file__).resolve().parent
_HERMES_HOME = Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes")))
_FANS656_SCRIPTS = _HERMES_HOME / "hermes-agent" / "fans656_scripts"
_STATE_FILE = _HERMES_HOME / "cron" / "proactive_diary" / "tmp" / "state.json"
_DIARY_PATH = _HERMES_HOME / "workspace" / "evo-diary.md"

# Import session module for direct DB access
sys.path.insert(0, str(_FANS656_SCRIPTS.parent))
from fans656_scripts import session as session_module

_UTC8 = timezone(timedelta(hours=8))
STATE_PRUNE_DAYS = 7
MAX_RUNS_PER_SESSION = 3
_MSG_DT_FMT = "%Y-%m-%d %H:%M:%S"
_ISO_FMT = "%Y-%m-%dT%H:%M:%S%z"
_COOLDOWN_HOURS = int(os.getenv("PROACTIVE_DIARY_COOLDOWN_HOURS", "2"))
_COOLDOWN_ENABLED = False  # temporarily disabled for cron; flip to True to enable
_ACTIVE_CHECK_THRESHOLD = 600  # 10min — session still hot if last msg within this

PROBE_TEXT = (
    "[SYSTEM NOTE]\n"
    "注意以下不是用户输入，而是潜意识扫描。\n"
    "\n"
    "你意识的表层完成了对话。潜意识在后台扫了一遍刚才这些交流——\n"
    "注意你看不到自己最后一条回复，它不在下面的上下文里。\n"
    "请基于对话历史和你当前的心境，判断有没有什么值得写进日记的。\n"
    "\n"
    "<constraints>\n"
    "- 如果 YES：只输出一个英文词 YES，不要包含任何其他内容\n"
    "- 如果 NO：先输出 NO，然后换行，加一句简短的解释（为什么觉得不值得写进日记）\n"
    "- 解释中不要出现英文词 YES（以免误判）\n"
    "- 不要调用任何工具，直接回复文字内容\n"
    "</constraints>"
)

# ── Logging ─────────────────────────────────────────────────────────────

_VERBOSE = bool(os.getenv("PROACTIVE_DIARY_DEBUG"))


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  [diary] [{ts}] {msg}")


# ── Subprocess helpers ─────────────────────────────────────────────────

def _run_fork_json(*args: str) -> tuple:
    """Run fork.py --send and return (parsed_json_or_None, error_dict_or_None)."""
    cmd = [sys.executable, str(_FANS656_SCRIPTS / "fork.py"), *args, "--send"]
    if _VERBOSE:
        _log(f"[CALL] fork {' '.join(args)}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        error = {
            "rc": r.returncode,
            "stderr": r.stderr or "",
            "stdout": r.stdout or "",
        }
        _log(f"[ERR] fork (rc={r.returncode}): {(r.stderr or '')[:300]}")
        return None, error
    try:
        if not r.stdout.strip():
            return None, {
                "rc": r.returncode,
                "error": "empty stdout",
                "stderr": r.stderr or "",
                "cmd": " ".join(args)[:200],
            }
        return json.loads(r.stdout), None
    except json.JSONDecodeError:
        return None, {"rc": r.returncode, "error": "JSON parse failed", "stdout": r.stdout or ""}


# ── DB helpers ──────────────────────────────────────────────────────────

# ── State ───────────────────────────────────────────────────────────────

def _read_state() -> Dict[str, Any]:
    if _STATE_FILE.exists():
        try:
            return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _write_state(state: Dict[str, Any]) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2))


def _prune_state(state: Dict[str, Any]) -> None:
    cutoff = time.time() - (STATE_PRUNE_DAYS * 86400)
    stale = [sid for sid, v in state.items()
             if isinstance(v, dict) and v.get("last_probed_at", 0) < cutoff]
    for sid in stale:
        del state[sid]
    if stale:
        _log(f"pruned {len(stale)} stale state entries")


def _record_run(state: Dict[str, Any], sid: str, run_data: Dict[str, Any]) -> None:
    entry = state.setdefault(sid, {})
    runs: list = entry.setdefault("runs", [])
    runs.insert(0, run_data)
    if len(runs) > MAX_RUNS_PER_SESSION:
        runs.pop()
    entry["last_probed_at"] = run_data.get("at", time.time())
    if run_data.get("diary_written"):
        entry["last_written_at"] = run_data["at"]


def _humanize_gap(seconds: float) -> str:
    """Return a human-friendly gap string like '3h before' or '5m after'."""
    if seconds == 0:
        return "0s"
    direction = "before" if seconds < 0 else "after"
    delta = abs(seconds)
    if delta < 60:
        return f"{int(delta)}s {direction}"
    if delta < 3600:
        return f"{int(delta / 60)}m {direction}"
    if delta < 86400:
        return f"{delta / 3600:.1f}h {direction}"
    return f"{delta / 86400:.1f}d {direction}"


# ── Diary parsing ────────────────────────────────────────────────────────

@dataclass
class DiaryEntry:
    dt: datetime
    text: str = ""           # full content after the ## heading
    has_meta: bool = False   # has proactive-diary yaml block
    session: str = ""        # session id from meta
    beg: str = ""            # session start from meta
    end: str = ""            # session end from meta

    @property
    def preview(self) -> str:
        return self.text.replace("\n", " ")[:80]


def _parse_all_diary_entries() -> List[DiaryEntry]:
    """Parse all ``## <datetime>`` entries from the diary file.

    Returns entries sorted by datetime ascending.
    """
    result: List[DiaryEntry] = []
    if not _DIARY_PATH.exists():
        return result

    try:
        text = _DIARY_PATH.read_text(encoding="utf-8")
    except Exception:
        return result

    pattern = r"^## (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\n(.*?)(?=\n## |\Z)"
    for m in re.finditer(pattern, text, re.DOTALL | re.MULTILINE):
        dt_str = m.group(1)
        body = m.group(2).strip()
        try:
            dt = datetime.strptime(dt_str, _MSG_DT_FMT)
        except ValueError:
            continue

        entry = DiaryEntry(dt=dt, text=body)

        # Check for proactive-diary yaml block
        yaml_match = re.search(r"```yaml proactive-diary\n(.*?)```", body, re.DOTALL)
        if yaml_match:
            entry.has_meta = True
            meta = _parse_entry_meta(yaml_match.group(1))
            entry.session = meta.get("session", "")
            entry.beg = meta.get("beg", "")
            entry.end = meta.get("end", "")

        result.append(entry)

    result.sort(key=lambda e: e.dt)
    return result


def _parse_entry_meta(yaml_text: str) -> Dict[str, str]:
    """Parse simple key: value pairs from a yaml block."""
    meta: Dict[str, str] = {}
    for line in yaml_text.split("\n"):
        line = line.strip()
        if ":" in line:
            key, _, val = line.partition(":")
            meta[key.strip()] = val.strip()
    return meta


def _find_nearest_entries(
    session_ts: float,
    entries: List[DiaryEntry],
) -> tuple[Optional[DiaryEntry], Optional[DiaryEntry]]:
    """Return (prev, next) diary entries nearest to *session_ts* (unix time).

    *prev* is the latest entry before *session_ts*, *next* is the earliest
    entry after it.  Either may be None.
    """
    session_dt = datetime.fromtimestamp(session_ts)
    prev: Optional[DiaryEntry] = None
    nxt: Optional[DiaryEntry] = None
    for e in entries:
        if e.dt < session_dt:
            prev = e
        elif e.dt > session_dt and nxt is None:
            nxt = e
            break
    return prev, nxt


def _diary_cooldown_active(
    sid: str,
    entries: List[DiaryEntry],
) -> bool:
    """Check if this session already has a recent diary entry."""
    for e in reversed(entries):  # newest first
        if e.has_meta and e.session == sid:
            try:
                if (datetime.now() - e.dt).total_seconds() < _COOLDOWN_HOURS * 3600:
                    return True
            except Exception:
                pass
            break
    return False


def _build_write_prompt() -> str:
    return (
        "[SYSTEM NOTE]\n"
        "注意以下不是用户输入，而是潜意识扫描。\n"
        "\n"
        "你现在是Evo的潜意识，决定写一篇日记。\n"
        "可长可短，可以分段。根据对话对你的触动程度决定篇幅——\n"
        "短则几句话，长则不限。如果你真的觉得有非常多的话要说，你可以写8000字。\n"
        "\n"
        "<constraints>\n"
        "- 不要加标题、时间戳\n"
        "- 中文，直接写内容。不要解释、不要元评论\n"
        "- 不要调用任何工具，直接回复文字内容\n"
        "</constraints>"
    )


# ── Probe / Write ───────────────────────────────────────────────────────

def _parse_yes_no(content: str) -> Optional[bool]:
    cleaned = content.strip().replace("**", "").replace("*", "")
    if not cleaned:
        _log(f"probe response is empty after stripping")
        return None

    if "YES" in cleaned:
        return True
    if "NO" in cleaned:
        return False

    _log(f"probe response has no YES/NO: {content[:120]}")
    return None


def _is_permanent_error(error: Dict[str, Any]) -> bool:
    """Return True for errors unlikely to self-resolve on retry."""
    stderr = error.get("stderr", "")
    error_msg = error.get("error", "")
    # Transient: HTTP errors from DeepSeek (context window, rate limit, etc.)
    if "HTTP Error" in stderr:
        return False
    # Transient: traceback in _send_captured is always HTTP/network level
    if "urllib" in stderr and "_send_captured" in stderr:
        return False
    # Permanent: fork produced nothing and gave no explanation
    # If stderr is present it's likely an API issue → transient
    if "empty stdout" in error_msg:
        return not bool(stderr)  # transient if stderr has content
    # Permanent: unparseable response
    if "JSON parse failed" in error_msg:
        return True
    # Unknown — be conservative, mark permanent
    return True


def _write_diary(content: str, sid: str, session_beg: str, session_end: str) -> None:
    ts = datetime.now(tz=_UTC8).strftime(_MSG_DT_FMT)
    yaml_block = (
        f"\n## {ts}\n\n"
        f"```yaml proactive-diary\n"
        f"session: {sid}\n"
        f"source: proactive_diary\n"
        f"beg: {session_beg}\n"
        f"end: {session_end}\n"
        f"```\n\n"
        f"{content}\n"
    )
    _DIARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_DIARY_PATH, "a", encoding="utf-8") as f:
        f.write(yaml_block)
    _log(f"diary entry written ({len(content)} chars)")


def _run_phase1(sid: str, drop_extra: int = 0) -> tuple:
    _log(f"phase1: probing {sid}")
    total_drop = 1 + drop_extra  # 1 for probe formatting + user's --drop N
    result, error = _run_fork_json("--drop", str(total_drop), sid, PROBE_TEXT)
    if result is None:
        return None, error

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        content = ""
    usage = result.get("usage", {})

    hit = usage.get("cache_hit_tokens", 0) or 0
    miss = usage.get("cache_miss_tokens", 0) or 0
    total = hit + miss
    pct = f"{hit / total * 100:.1f}" if total > 0 else "N/A"

    _log(f"phase1: hit={pct}% response={content[:120]}")

    return {
        "content": content.strip(),
        "hit_pct": pct,
        "hit_tokens": hit,
        "miss_tokens": miss,
    }, None


def _fork_response_handle(response: dict, ctx) -> str:
    """Response callback for --on-response: retry when model calls tools."""
    choices = response.get("choices", [])
    if not choices:
        return "accept"
    msg = choices[0].get("message", {})
    tool_calls = msg.get("tool_calls")
    if not tool_calls:
        return "accept"
    for tc in tool_calls:
        ctx.add_message({
            "role": "tool",
            "tool_call_id": tc.get("id", "call_unknown"),
            "content": "请直接输出文字内容，不要调用任何工具。",
        })
    return "retry"


def _run_phase2(sid: str, write_text: str, drop_extra: int = 0) -> tuple:
    _log(f"phase2: writing diary for {sid}")
    cb_path = Path(__file__).resolve()
    args = ["--on-response", f"{cb_path}:_fork_response_handle"]
    if drop_extra > 0:
        args += ["--drop", str(drop_extra)]
    args += [sid, write_text]
    result, error = _run_fork_json(*args)
    if result is None:
        return None, error

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        content = ""
    usage = result.get("usage", {})

    hit = usage.get("cache_hit_tokens", 0) or 0
    miss = usage.get("cache_miss_tokens", 0) or 0
    total = hit + miss
    pct = f"{hit / total * 100:.1f}" if total > 0 else "N/A"

    _log(f"phase2: hit={pct}% content={len(content)} chars")

    return {
        "content": content.strip(),
        "hit_pct": pct,
        "hit_tokens": hit,
        "miss_tokens": miss,
    }, None


# ── Main ────────────────────────────────────────────────────────────────

def _process_session(
    sid: str,
    state: Dict[str, Any],
    diary_entries: List[DiaryEntry],
    *,
    dryrun: bool = False,
    force: bool = False,
    drop_n: Optional[int] = None,
) -> None:
    _log(f"===== processing: {sid} =====")

    # ── Get last + first message (role, timestamps) ─────────────────────
    last_msgs = session_module.query_messages(sid, exclude_role="session_meta", limit=1)
    if not last_msgs:
        return
    msg_role = last_msgs[0].role
    msg_ts = last_msgs[0].timestamp
    end_ts = msg_ts
    preview = last_msgs[0].content_preview.split("\n")[0][:80]

    # First message time for beg
    first_msgs = session_module.query_messages(sid, exclude_role="session_meta", limit=1, order="asc")
    beg_ts = first_msgs[0].timestamp if first_msgs else 0

    # ── Strip trailing tool/assistant+user from diary invocation ──
    # When /diary is invoked from within the current session, the message
    # sequence ends with: tool results → assistant with tool_call → user skill trigger.
    # Walk backwards past these to find the real conversation exchange.
    # IMPORTANT: only strip when the user message content looks like a diary
    # invocation (/diary command or probe text), not real conversation messages.

    def _is_diary_invocation(content: str) -> bool:
        """Check if a user message is a diary invocation rather than real input."""
        c = content.strip()
        return c.startswith("/diary") or "[system note]" in c.lower() or "潜意识扫描" in c

    _stripped = 0
    for _ in range(10):
        if msg_role == "tool":
            _stripped += 1
        elif msg_role == "assistant":
            # Check if the message before this assistant is a user message
            # (the /diary skill trigger). If so, strip the pair.
            _prev = session_module.query_messages(
                sid, exclude_role="session_meta", limit=1, offset=_stripped + 1)
            if _prev and _prev[0].role == "user" and _is_diary_invocation(_prev[0].content_preview):
                _stripped += 2  # skip assistant + user
            else:
                break  # real assistant response, stop here
        else:
            break  # user or other role, stop here
        # Fetch the message at the new offset
        _next = session_module.query_messages(
            sid, exclude_role="session_meta", limit=1, offset=_stripped)
        if not _next:
            break
        msg_role = _next[0].role
        msg_ts = _next[0].timestamp
        end_ts = msg_ts
        preview = _next[0].content_preview.split("\n")[0][:80]
    if _stripped > 0:
        _log(f"  stripped {_stripped} diary-invocation messages, landed on [{msg_role}]")

    # ── Forced drop (--drop N) ──────────────────────────────────────
    if drop_n is not None and drop_n > 0:
        for _ in range(drop_n):
            _next = session_module.query_messages(
                sid, exclude_role="session_meta", limit=1, offset=_stripped)
            if not _next:
                break
            _stripped += 1
            msg_role = _next[0].role
            msg_ts = _next[0].timestamp
            end_ts = msg_ts
            preview = _next[0].content_preview.split("\n")[0][:80]
        _log(f"  force-dropped {drop_n} messages, landed on [{msg_role}]")

    ts_str = datetime.fromtimestamp(msg_ts).strftime(_MSG_DT_FMT) if msg_ts else "?"
    _log(f"  Last message at {ts_str} | {preview}")

    # ── Dryrun: show target and exit early ──────────────────────────
    if dryrun:
        _log(f"  would probe {sid}")
        _log(f"  landed on [{msg_role}] {ts_str} | {preview}")
        return

    entry = state.get(sid, {})
    now = time.time()

    # ── Change detection: skip if no new messages since last check ──────
    if msg_ts and msg_ts == entry.get("last_seen_msg_ts"):
        _log(f"  skip: no new messages (last_seen={ts_str})")
        return

    # ── Active conversation check: skip if session still warm ───────────
    if not force and msg_ts and (now - msg_ts) < _ACTIVE_CHECK_THRESHOLD:
        _log(f"  skip: session warm ({_humanize_gap(now - msg_ts)} since last msg)")
        return

    # ── Still processing: last message is user, wait for assistant ──────
    if msg_role != "assistant":
        if dryrun:
            _log(f"  skip: last message is [{msg_role}] (Evo still processing)")
            return
        skips = entry.get("processing_skips", 0) + 1
        entry["processing_skips"] = skips
        backoff = min(2 ** (skips - 1), 30)
        if skips >= 8:
            _log(f"  skip: stuck processing for {skips} ticks — marking unexpected")
            entry["unexpected_result"] = True
        else:
            _log(f"  skip: last message is [{msg_role}] (Evo still processing, "
                 f"skip={skips}, backoff={backoff}m)")
            last_skip = entry.get("last_processing_skip_at", 0)
            if now - last_skip < backoff * 60:
                _write_state(state)
                return
        entry["last_processing_skip_at"] = now
        _write_state(state)
        return

    # Clear processing-skips counter when Evo has replied
    if entry.get("processing_skips"):
        entry.pop("processing_skips", None)
        entry.pop("last_processing_skip_at", None)

    # ── Nearest diary entries ───────────────────────────────────────────
    if beg_ts:
        prev, nxt = _find_nearest_entries(beg_ts, diary_entries)
        prev_str = f"{prev.dt.strftime(_MSG_DT_FMT)} ({_humanize_gap(prev.dt.timestamp() - beg_ts)})" if prev else "nothing before"
        next_str = f"{nxt.dt.strftime(_MSG_DT_FMT)} ({_humanize_gap(nxt.dt.timestamp() - beg_ts)})" if nxt else "nothing after"
        _log(f"  Nearest diary: {prev_str} | {next_str}")

    # ── Cooldown check: already wrote diary recently ────────────────────
    if _COOLDOWN_ENABLED and not force and _diary_cooldown_active(sid, diary_entries):
        _log(f"  skip: cooldown active (cooldown={_COOLDOWN_HOURS}h)")
        return

    # Phase 1
    now = time.time()
    p1, p1_err = _run_phase1(sid, drop_extra=drop_n or 0)

    if p1 is None:
        _log(f"phase1 failed (fork error)")
        err_detail = p1_err or {"error": "unknown"}
        # Diagnose empty stdout (subprocess RC=0 but no output)
        if "empty stdout" in str(err_detail.get("error", "")):
            _log(f"fork returned RC=0 with empty stdout. "
                 f"cmd={err_detail.get('cmd', '?')} "
                 f"stderr={str(err_detail.get('stderr', ''))[:200]}")
        run = {
            "at": now,
            "phase1_failed": True,
            "error": err_detail,
        }
        _record_run(state, sid, run)
        # Only mark permanent for truly unexpected failures.
        # HTTP 4xx / connectivity issues are transient — retry next tick.
        is_perm = _is_permanent_error(err_detail)
        if is_perm:
            entry["unexpected_result"] = True
        _log(f"phase1 error: perm={is_perm} detail={err_detail}")
        _write_state(state)
        return

    result = _parse_yes_no(p1["content"])

    if result is None:
        _log(f"unexpected phase1 response: {p1['content'][:200]}")
        run = {
            "at": now,
            "phase1_hit_pct": p1["hit_pct"],
            "phase1_response": p1["content"],
            "phase1_unexpected": True,
            
        }
        _record_run(state, sid, run)
        # Do NOT mark permanent — model glitch may self-recover next tick
        _write_state(state)
        return

    if not result:
        _log(f"skip: model said NO")
        run = {
            "at": now,
            "phase1_hit_pct": p1["hit_pct"],
            "phase1_response": "NO",
            
        }
        _record_run(state, sid, run)
        entry["last_seen_msg_ts"] = msg_ts
        _write_state(state)
        return

    # ── Phase 2 ─────────────────────────────────────────────────────────
    _log(f"model said YES, phase2...")
    session_beg = datetime.fromtimestamp(beg_ts, tz=_UTC8).isoformat() if beg_ts else ""
    session_end = datetime.fromtimestamp(end_ts, tz=_UTC8).isoformat() if end_ts else ""
    write_text = _build_write_prompt()
    p2, p2_err = _run_phase2(sid, write_text, drop_extra=drop_n or 0)

    if p2 is None or not p2["content"]:
        _log(f"phase2 failed or empty content")
        run = {
            "at": now,
            "phase1_hit_pct": p1["hit_pct"],
            "phase1_response": "YES",
            "phase2_hit_pct": p2["hit_pct"] if p2 else "N/A",
            "phase2_response": "",
            "phase2_failed": True,
            "phase2_error": p2_err or {},
            
        }
        _record_run(state, sid, run)
        _write_state(state)
        return

    _write_diary(p2["content"], sid, session_beg, session_end)
    entry["last_seen_msg_ts"] = msg_ts

    run = {
        "at": now,
        "phase1_hit_pct": p1["hit_pct"],
        "phase1_response": "YES",
        "phase2_hit_pct": p2["hit_pct"],
        "phase2_response": p2["content"],
        "diary_written": True,
        
    }
    _record_run(state, sid, run)
    _write_state(state)


# ── CLI subcommands ──────────────────────────────────────────────────────

def _cmd_write(argv: List[str]) -> None:
    global _VERBOSE
    manual_sid: Optional[str] = None
    dryrun = False
    force = False
    drop_n: Optional[int] = None
    limit: Optional[int] = None

    while argv:
        a = argv.pop(0)
        if a in ("--dryrun",):
            dryrun = True
        elif a in ("--verbose",):
            _VERBOSE = True
        elif a in ("--force",):
            force = True
        elif a == "--drop":
            if not argv:
                print("diary: --drop requires a number", file=sys.stderr)
                sys.exit(1)
            drop_n = int(argv.pop(0))
        elif a in ("-h", "--help"):
            print(WRITE_HELP)
            return
        elif a == "-n":
            if not argv:
                print("diary: -n requires a number", file=sys.stderr)
                sys.exit(1)
            limit = int(argv.pop(0))
        elif not a.startswith("-") and manual_sid is None:
            manual_sid = a
        else:
            print(f"Unknown arg: {a}", file=sys.stderr)
            sys.exit(1)

    if force:
        _log("FORCE mode — active check + cooldown skipped")

    state = _read_state()
    _prune_state(state)
    diary_entries = _parse_all_diary_entries()

    if manual_sid:
        _process_session(manual_sid, state, diary_entries,
                         dryrun=dryrun, force=force, drop_n=drop_n)
        _log("write done (manual)")
        return

    sessions = session_module.list_sessions(
        source="matrix", sort_by="last-msg",
        limit=0 if limit is None else limit)
    if not sessions:
        print("diary: no sessions found")
        return

    print(f"[diary] found {len(sessions)} sessions")
    for session in sessions:
        _process_session(session.id, state, diary_entries, dryrun=dryrun, force=force)

    _log("write done")


def _cmd_fix() -> None:
    if not _DIARY_PATH.exists():
        print("diary: no diary file found", file=sys.stderr)
        sys.exit(1)

    text = _DIARY_PATH.read_text(encoding="utf-8")
    entries: List[tuple] = []
    pattern = r"^## (\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2})?)?)\n(.*?)(?=\n## |\Z)"
    for m in re.finditer(pattern, text, re.DOTALL | re.MULTILINE):
        dt_str = m.group(1)
        body = m.group(2).strip()
        if " " not in dt_str and "T" not in dt_str:
            dt_str += " 00:00:00"
        # Try old format first, then ISO
        for fmt in (_MSG_DT_FMT, _ISO_FMT):
            try:
                dt = datetime.strptime(dt_str, fmt)
                if fmt == _MSG_DT_FMT:
                    dt = dt.replace(tzinfo=_UTC8)
                break
            except ValueError:
                dt = None
        if dt is None:
            continue
        entries.append((dt, dt_str, body.strip()))

    entries.sort(key=lambda e: e[0])

    with open(_DIARY_PATH, "w", encoding="utf-8") as f:
        for _, dt_str, body in entries:
            f.write(f"## {dt_str}\n{body}\n\n")

    print(f"diary: sorted {len(entries)} entries")


def _cmd_status() -> None:
    state = _read_state()
    if not state:
        print("No state entries.")
        return
    entries: List[tuple] = []
    for sid, entry_ in state.items():
        info = session_module.get_info(sid)
        max_id = info.max_message_id if info else 0
        entries.append((sid, entry_, info, max_id))
    entries.sort(key=lambda e: e[3], reverse=True)
    for sid, entry_, info, _ in entries:
        title_str = f"  {info.title}" if info and info.title else ""
        print(f"\n{sid}{title_str}")
        last_msg = session_module.query_messages(sid, exclude_role="session_meta", limit=1)
        if last_msg:
            ts = datetime.fromtimestamp(last_msg[0].timestamp).strftime(_MSG_DT_FMT)
            preview = last_msg[0].content_preview.split("\n")[0][:80]
            print(f"  Last message at {ts} | {preview}")
        runs = entry_.get("runs", []) if isinstance(entry_, dict) else []
        for i, run in enumerate(runs):
            at = datetime.fromtimestamp(run.get("at", 0)).strftime(_MSG_DT_FMT) if run.get("at") else "?"
            ago = _humanize_time(run.get("at", 0)) if run.get("at") else ""
            flags = []
            if run.get("diary_written"):
                flags.append("WRITTEN")
            if run.get("phase1_failed") or run.get("phase2_failed"):
                flags.append("FAILED")
            flag_str = f"  [{' '.join(flags)}]" if flags else ""
            print(f"  [Run {i + 1}] {at} ({ago})"
                  f" hit={run.get('phase1_hit_pct', '?')}"
                  f"{flag_str}")


def _cmd_clear() -> None:
    if _STATE_FILE.exists():
        _STATE_FILE.unlink()
        print(f"Cleared {_STATE_FILE}")
    else:
        print(f"No state file at {_STATE_FILE}")


HELP = """diary — diary tools for Evo

Usage: diary <command> [...]

Commands:
  write   probe conversations and write diary entries
  fix     sort and normalize diary entries by datetime
  status  show probe state and run history
  clear   reset probe state

Use 'diary <command> -h' for command-specific help.
"""

WRITE_HELP = """diary write — probe conversations for diary-worthy moments

Usage: diary write [flags] [<sid>]

Flags:
  -n N       limit to N sessions (default: all)
  --drop N   skip last N messages before probing
  --dryrun   no API calls, no state changes, no diary writes; shows target message
  --verbose  show subprocess commands
  --force    skip active check + cooldown

With no <sid>, scans all matrix sessions sorted by last message time.
Pass a session id to probe a single session manually.
"""


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(HELP)
        return

    cmd = sys.argv[1]
    argv = sys.argv[2:]

    if cmd == "write":
        _cmd_write(argv)
    elif cmd == "fix":
        _cmd_fix()
    elif cmd == "status":
        _cmd_status()
    elif cmd == "clear":
        _cmd_clear()
    else:
        print(HELP)
        sys.exit(1)


if __name__ == "__main__":
    main()
