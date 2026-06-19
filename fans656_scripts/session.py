#!/usr/bin/env python3
"""Unified session discovery, polling, and state query — DB + captured aware.

Operates at the `state.db` level with awareness of captured
request/response pairs under ``sessions/cache/captured/{id}/``.

Usage as CLI::

    session.py list --source matrix --captured --limit 5
    session.py info <session_id>
    session.py find "keyword" --source matrix
    session.py poll <session_id> --after-msg 12345
    session.py messages <sid> --role assistant --exclude-role session_meta --last 1 --json

Usage as library::

    from fans656_scripts.session import list_sessions, poll, get_info
    sessions = list_sessions(source="matrix", captured_only=True, limit=1)
    if sessions:
        info = poll(sessions[0].id, after_msg_id=100)
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["SessionInfo", "MessageInfo", "list_sessions", "get_info",
           "query_messages", "find_by_content", "poll"]

# ── Path setup ──────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _hermes_home() -> Path:
    env = os.getenv("HERMES_HOME")
    if env:
        return Path(env)
    return Path.home() / ".hermes"


_STATE_DB = _hermes_home() / "state.db"
_CAPTURED_ROOT = _hermes_home() / "sessions" / "cache" / "captured"


# ── Dataclass ───────────────────────────────────────────────────────────

@dataclass
class SessionInfo:
    id: str
    source: str
    message_count: int
    max_message_id: int
    started_at: float
    ended_at: Optional[float]
    captured_count: int = 0
    captured_last_at: float = 0.0
    title: str = ""
    model: str = ""
    preview: str = ""
    last_msg_preview: str = ""

    @property
    def has_captured(self) -> bool:
        return self.captured_count > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "message_count": self.message_count,
            "max_message_id": self.max_message_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "captured_count": self.captured_count,
            "captured_last_at": self.captured_last_at,
            "has_captured": self.has_captured,
            "title": self.title,
            "model": self.model,
            "preview": self.preview,
            "last_msg_preview": self.last_msg_preview,
        }


# ── Internal helpers ────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_STATE_DB))
    conn.row_factory = sqlite3.Row
    return conn


def _build_exclude_clause(exclude_sources: Optional[List[str]]) -> tuple[str, list]:
    """Return (SQL clause, params) for excluding sources, or ('', [])."""
    ex_list = [s for s in (exclude_sources or []) if s]
    if not ex_list:
        return "", []
    placeholders = ",".join("?" for _ in ex_list)
    return f"AND source NOT IN ({placeholders})", ex_list


def _captured_info(session_id: str) -> tuple[int, float]:
    """Return (count, latest_mtime) of captured req files for a session."""
    d = _CAPTURED_ROOT / session_id
    if not d.is_dir():
        return 0, 0.0
    reqs = sorted(d.glob("*_req.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return len(reqs), reqs[0].stat().st_mtime if reqs else 0.0


def _row_to_info(row: Any) -> SessionInfo:
    d = dict(row) if not isinstance(row, dict) else row
    sid = d.get("id") or d.get("session_id") or ""
    count, mtime = _captured_info(sid)
    return SessionInfo(
        id=sid,
        source=d.get("source", ""),
        message_count=d.get("message_count", 0),
        max_message_id=d.get("max_message_id", 0),
        started_at=d.get("started_at", 0.0),
        ended_at=d.get("ended_at"),
        captured_count=count,
        captured_last_at=mtime,
        title=d.get("title", "") or "",
        model=d.get("model", ""),
        preview=d.get("preview", ""),
        last_msg_preview=d.get("last_msg_preview", ""),
    )


def _get_max_msg_id_raw(session_id: str) -> int:
    try:
        conn = _connect()
        row = conn.execute(
            "SELECT MAX(id) as max_id FROM messages WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        conn.close()
        if row and row["max_id"] is not None:
            return row["max_id"]
    except Exception as exc:
        print(f"session: _get_max_msg_id_raw({session_id}) failed: {exc}",
              file=sys.stderr)
    return 0


def _get_preview(session_id: str) -> str:
    try:
        conn = _connect()
        row = conn.execute(
            "SELECT substr(content, 1, 80) as preview FROM messages "
            "WHERE session_id = ? AND role = 'user' "
            "ORDER BY id ASC LIMIT 1",
            (session_id,),
        ).fetchone()
        conn.close()
        if row:
            return row["preview"] or ""
    except Exception as exc:
        print(f"session: _get_preview({session_id}) failed: {exc}",
              file=sys.stderr)
    return ""
    try:
        conn = _connect()
        row = conn.execute(
            "SELECT substr(content, 1, 80) as preview FROM messages "
            "WHERE session_id = ? AND role = 'user' "
            "ORDER BY id ASC LIMIT 1",
            (session_id,),
        ).fetchone()
        conn.close()
        if row:
            return row["preview"] or ""
    except Exception as exc:
        print(f"session: _get_preview({session_id}) failed: {exc}",
              file=sys.stderr)
    return ""


# ── Core API ────────────────────────────────────────────────────────────

def list_sessions(
    source: Optional[str] = None,
    exclude_sources: Optional[List[str]] = None,
    limit: int = 20,
    captured_only: bool = False,
    sort_by: str = "started",
) -> List[SessionInfo]:
    """List recent sessions from state.db, with captured directory awareness.

    Args:
        source: Filter by platform source (e.g. ``"matrix"``).
        exclude_sources: Exclude these sources (e.g. ``["cron"]``).
        limit: Maximum results (0 = unlimited).
        captured_only: Only return sessions that have captured req/resp pairs.
        sort_by: "started" (default) or "last-msg".
    """
    try:
        conn = _connect()

        exclude_clause, ex_params = _build_exclude_clause(exclude_sources)
        where_parts = ["1=1"]
        params: list = []

        if source:
            where_parts.append("AND source = ?")
            params.append(source)
        if exclude_clause:
            where_parts.append(exclude_clause)
            params.extend(ex_params)
        where_parts.append("AND message_count > 0")

        order_clause = "ORDER BY s.started_at DESC"
        if sort_by == "last-msg":
            order_clause = "ORDER BY max_message_id DESC"

        if limit <= 0:
            limit_clause = ""
        else:
            limit_clause = "LIMIT ?"

        if limit > 0:
            params.append(limit * 4 if captured_only else limit)

        rows = conn.execute(
            f"""SELECT s.id, s.source, s.message_count, s.model, s.title,
                       s.started_at, s.ended_at,
                       COALESCE((SELECT MAX(m.id) FROM messages m
                                  WHERE m.session_id = s.id), 0) as max_message_id,
                       COALESCE((SELECT substr(REPLACE(REPLACE(m2.content, X'0A', ' '), X'0D', ' '), 1, 80) FROM messages m2
                                  WHERE m2.session_id = s.id AND m2.role = 'user'
                                  ORDER BY m2.id ASC LIMIT 1), '') as preview,
                       COALESCE((SELECT substr(REPLACE(REPLACE(m3.content, X'0A', ' '), X'0D', ' '), 1, 80) FROM messages m3
                                  WHERE m3.session_id = s.id AND m3.content IS NOT NULL
                                  ORDER BY m3.id DESC LIMIT 1), '') as last_msg_preview
                FROM sessions s
                WHERE {' '.join(where_parts)}
                {order_clause} {limit_clause}""",
            params,
        ).fetchall()
        conn.close()
    except Exception as exc:
        print(f"session: list_sessions failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return []

    result: List[SessionInfo] = []
    for row in rows:
        info = _row_to_info(row)
        if captured_only and not info.has_captured:
            continue
        result.append(info)
        if limit > 0 and len(result) >= limit:
            break

    return result


def get_info(session_id: str) -> Optional[SessionInfo]:
    """Return full SessionInfo including captured directory status."""
    try:
        conn = _connect()
        row = conn.execute(
            "SELECT id, source, message_count, model, title, started_at, ended_at "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        conn.close()
        if row:
            info = _row_to_info(row)
            info.max_message_id = _get_max_msg_id_raw(session_id)
            info.preview = _get_preview(session_id)
            return info
    except Exception as exc:
        print(f"session: get_info({session_id}) failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
    return None


def find_by_content(
    text: str,
    *,
    source: Optional[str] = None,
    within_seconds: int = 300,
    exact: bool = False,
) -> Optional[str]:
    """Search recent messages for *text* and return the session_id."""
    min_id = _approx_id_before_seconds(within_seconds)
    try:
        conn = _connect()
        clause = "m.content = ?" if exact else "m.content LIKE ?"
        param = text if exact else f"%{text}%"

        if source:
            row = conn.execute(
                f"SELECT m.session_id FROM messages m "
                f"JOIN sessions s ON s.id = m.session_id "
                f"WHERE {clause} AND s.source = ? AND m.id > ? "
                f"ORDER BY m.id DESC LIMIT 1",
                (param, source, min_id),
            ).fetchone()
        else:
            row = conn.execute(
                f"SELECT m.session_id FROM messages m "
                f"WHERE {clause} AND m.id > ? "
                f"ORDER BY m.id DESC LIMIT 1",
                (param, min_id),
            ).fetchone()
        conn.close()
        return row["session_id"] if row else None
    except Exception as exc:
        print(f"session: find_by_content failed: {exc}", file=sys.stderr)
        return None


def _approx_id_before_seconds(seconds: int) -> int:
    """Heuristic: return an approximate message id from N seconds ago."""
    if seconds <= 0:
        return 0
    try:
        conn = _connect()
        row = conn.execute("SELECT MAX(id) as max_id FROM messages").fetchone()
        conn.close()
        if row and row["max_id"]:
            return max(0, row["max_id"] - (seconds * 3))
    except Exception as exc:
        print(f"session: _approx_id_before_seconds({seconds}) failed: {exc}",
              file=sys.stderr)
    return 0


def poll(
    session_id: str,
    *,
    after_msg_id: Optional[int] = None,
    after_ts: Optional[float] = None,
    timeout: int = 120,
    interval: int = 1,
) -> Optional[SessionInfo]:
    """Poll state.db until the session has new messages.

    Returns updated SessionInfo or None on timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            conn = _connect()
            row = conn.execute(
                "SELECT id, source, message_count, model, title, started_at, ended_at "
                "FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            conn.close()

            if row is None:
                time.sleep(interval)
                continue

            info = _row_to_info(row)
            info.max_message_id = _get_max_msg_id_raw(session_id)

            if after_msg_id is not None:
                if info.max_message_id > after_msg_id:
                    return info
            elif after_ts is not None:
                if info.started_at > after_ts:
                    return info
            else:
                if info.message_count > 0:
                    return info
        except Exception as exc:
            print(f"session: poll({session_id}) iteration failed: {exc}",
                  file=sys.stderr)
        time.sleep(interval)
    return None


def get_max_msg_id(session_id: str) -> int:
    """Return the maximum message.id for a session."""
    return _get_max_msg_id_raw(session_id)


@dataclass
class MessageInfo:
    id: int
    role: str
    timestamp: float = 0.0
    content_preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "role": self.role,
                "timestamp": self.timestamp, "content_preview": self.content_preview}


def query_messages(
    session_id: str,
    *,
    role: Optional[str] = None,
    exclude_role: Optional[str] = None,
    limit: int = 1,
    offset: int = 0,
    order: str = "desc",
) -> List[MessageInfo]:
    """Return messages for a session, optionally filtered by role.

    Args:
        session_id: Session to query.
        role: Only return messages with this role.
        exclude_role: Exclude messages with this role.
        limit: Max messages to return (default 1).
        offset: Skip first N results (default 0).
        order: \"desc\" (latest first, default) or \"asc\" (oldest first).
    """
    where_parts = ["session_id = ?"]
    params: list = [session_id]

    if role:
        where_parts.append("AND role = ?")
        params.append(role)
    if exclude_role:
        where_parts.append("AND role != ?")
        params.append(exclude_role)

    order_clause = "DESC" if order == "desc" else "ASC"

    where_clause = " ".join(where_parts)
    try:
        conn = _connect()
        rows = conn.execute(
            f"SELECT id, role, timestamp, substr(content, 1, 120) as content_preview "
            f"FROM messages "
            f"WHERE {where_clause} "
            f"ORDER BY id {order_clause} LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        conn.close()
        if order == "desc":
            rows = list(reversed(rows))  # oldest first for caller convenience
        return [
            MessageInfo(
                id=row["id"],
                role=row["role"],
                timestamp=row["timestamp"] or 0.0,
                content_preview=row["content_preview"] or "",
            )
            for row in rows
        ]
    except Exception as exc:
        print(f"session: query_messages({session_id}) failed: {exc}",
              file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return []


# ── CLI ─────────────────────────────────────────────────────────────────

def _parse_exclude(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [s.strip() for s in raw.split(",") if s.strip()]


def _print_info(info: SessionInfo, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(info.to_dict(), ensure_ascii=False, indent=2))
        return

    ts = ""
    if info.started_at:
        ts = datetime.fromtimestamp(info.started_at).strftime("%Y-%m-%d %H:%M")
    cap = (
        f"{info.captured_count} pairs / latest "
        f"{datetime.fromtimestamp(info.captured_last_at).strftime('%H:%M:%S')}"
    ) if info.has_captured else "no"

    print("===")
    print(f"session:       {info.id}")
    print(f"source:        {info.source}")
    print(f"messages:      {info.message_count}")
    print(f"max_msg_id:    {info.max_message_id}")
    print(f"started:       {ts}")
    print(f"ended:         {'yes' if info.ended_at else 'active'}")
    if info.title:
        print(f"title:         {info.title}")
    if info.model:
        print(f"model:         {info.model}")
    print(f"captured:      {cap}")
    print("===")

    head, tail = _get_content_previews(info.id)
    if head:
        for line in _wrap_preview(head, width=80):
            print(f"> {line}")
    if tail:
        print("  ...")
        for line in _wrap_preview(tail, width=80):
            print(f"> {line}")


def _wrap_preview(text: str, width: int = 80) -> list[str]:
    """Wrap long preview text into lines prefixed for display."""
    lines = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        while len(para) > width:
            lines.append(para[:width])
            para = para[width:]
        lines.append(para)
    return lines


def _get_content_previews(session_id: str) -> tuple[str, str]:
    """Return (head, tail) preview content — first user msg and last assistant msg."""
    head = ""
    tail = ""
    try:
        conn = _connect()
        r = conn.execute(
            "SELECT substr(content, 1, 500) as content FROM messages "
            "WHERE session_id = ? AND role = 'user' "
            "ORDER BY id ASC LIMIT 1",
            (session_id,),
        ).fetchone()
        if r:
            head = r["content"] or ""
        r = conn.execute(
            "SELECT substr(content, 1, 500) as content FROM messages "
            "WHERE session_id = ? AND role = 'assistant' "
            "ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        if r:
            tail = r["content"] or ""
        conn.close()
    except Exception as exc:
        print(f"session: _get_content_previews({session_id}) failed: {exc}",
              file=sys.stderr)
    return head, tail


def _cmd_list(args: argparse.Namespace) -> None:
    sessions = list_sessions(
        source=args.source or None,
        exclude_sources=_parse_exclude(args.exclude),
        limit=0 if args.all else args.limit,
        captured_only=args.captured,
        sort_by=args.sort,
    )
    if args.json:
        print(json.dumps([s.to_dict() for s in sessions],
                         ensure_ascii=False, indent=2))
    elif not sessions:
        print("No sessions found.")
    else:
        count = len(sessions)
        sql_count = _count_sessions(
            source=args.source or None,
            exclude_sources=_parse_exclude(args.exclude),
        )
        total_str = f" / {sql_count}" if sql_count > count else ""
        print(f"Showing {count}{total_str} results")

        # Build header
        cols = [f"{'SESSION_ID':<42}", f"{'SRC':>6}", f"{'MSGS':>5}"]
        if args.show_capture_detail:
            cols.extend([f"{'CAPT':>5}", "  LAST_CAPTURED"])
        cols.append("LAST MSG")
        print("  ".join(cols))
        print("-" * (len("  ".join(cols))))

        for s in sessions:
            row = [
                f"{s.id:<42}",
                f"{s.source[:6]:>6}",
                f"{s.message_count:>5}",
            ]
            if args.show_capture_detail:
                cap_str = str(s.captured_count) if s.has_captured else "-"
                last_str = (
                    datetime.fromtimestamp(s.captured_last_at).strftime("%m-%d %H:%M:%S")
                    if s.has_captured
                    else "-"
                )
                row.extend([f"{cap_str:>5}", f"  {last_str}"])
            title = s.last_msg_preview[:60] if s.last_msg_preview else "-"
            row.append(title)
            print("  ".join(row))


def _count_sessions(
    source: Optional[str],
    exclude_sources: Optional[List[str]],
) -> int:
    """Return total matching sessions (without limit / captured filter)."""
    try:
        conn = _connect()
        exclude_clause, ex_params = _build_exclude_clause(exclude_sources)
        where_parts = ["1=1"]
        params: list = []
        if source:
            where_parts.append("AND source = ?")
            params.append(source)
        if exclude_clause:
            where_parts.append(exclude_clause)
            params.extend(ex_params)
        where_parts.append("AND message_count > 0")

        row = conn.execute(
            f"SELECT COUNT(*) as cnt FROM sessions WHERE {' '.join(where_parts)}",
            params,
        ).fetchone()
        conn.close()
        return row["cnt"] if row else 0
    except Exception as exc:
        print(f"session: _count_sessions failed: {exc}", file=sys.stderr)
        return 0


def _cmd_info(args: argparse.Namespace) -> None:
    info = get_info(args.session_id)
    if info is None:
        print(f"Session not found: {args.session_id}", file=sys.stderr)
        sys.exit(1)
    _print_info(info, args.json)


def _cmd_find(args: argparse.Namespace) -> None:
    sid = find_by_content(
        args.text,
        source=args.source or None,
        within_seconds=args.within,
        exact=args.exact,
    )
    if sid:
        print(sid)
    else:
        sys.exit(1)


def _cmd_poll(args: argparse.Namespace) -> None:
    after_msg = args.after_msg
    after_ts = args.after_ts

    # Default: start from current max message id
    if after_msg is None and after_ts is None:
        after_msg = _get_max_msg_id_raw(args.session_id)

    while True:
        info = poll(
            args.session_id,
            after_msg_id=after_msg,
            after_ts=after_ts,
            timeout=args.timeout,
            interval=args.interval,
        )
        if info is None:
            if after_msg is not None:
                print(f"Timeout after {args.timeout}s (after_msg={after_msg})", file=sys.stderr)
            else:
                print(f"Timeout after {args.timeout}s", file=sys.stderr)
            sys.exit(1)
        _print_info(info, args.json)
        if not args.follow:
            return
        # Update baseline for next poll
        after_msg = info.max_message_id
        after_ts = None


def _cmd_messages(args: argparse.Namespace) -> None:
    msgs = query_messages(
        args.session_id,
        role=args.role,
        exclude_role=args.exclude_role,
        limit=args.limit,
        offset=args.offset,
        order=args.order,
    )
    if args.json:
        print(json.dumps([m.to_dict() for m in msgs],
                         ensure_ascii=False, indent=2))
    else:
        for m in msgs:
            print(f"[{m.role}] {m.content_preview}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified session discovery, polling, and state query — DB + captured aware.",
    )
    sub = parser.add_subparsers(dest="command", help="Subcommand")

    p_list = sub.add_parser("list", help="List recent sessions")
    p_list.add_argument("-s", "--source", help="Filter by platform source")
    p_list.add_argument("-x", "--exclude",
                        help="Comma-separated sources to exclude")
    p_list.add_argument("-n", "--limit", type=int, default=20,
                        help="Max results (0=unlimited, default 20)")
    p_list.add_argument("--all", action="store_true",
                        help="Return all sessions (overrides -n)")
    p_list.add_argument("--sort", choices=["started", "last-msg"],
                        default="last-msg",
                        help="Sort by started_at or last message time (default: last-msg)")
    p_list.add_argument("--captured", action="store_true",
                        help="Only sessions with captured req/resp pairs")
    p_list.add_argument("--show-capture-detail", action="store_true",
                        help="Show capture count and last captured time columns")
    p_list.add_argument("--json", action="store_true",
                        help="Output as JSON array")

    p_info = sub.add_parser("info", help="Show session state")
    p_info.add_argument("session_id", help="Session ID")
    p_info.add_argument("--json", action="store_true",
                        help="Output as JSON")

    p_find = sub.add_parser("find", help="Find session by message content")
    p_find.add_argument("text", help="Text to search for in messages")
    p_find.add_argument("-s", "--source", help="Filter by platform source")
    p_find.add_argument("-w", "--within", type=int, default=300,
                        help="Search within last N seconds (default 300)")
    p_find.add_argument("--exact", action="store_true",
                        help="Exact match instead of LIKE")

    p_poll = sub.add_parser("poll", help="Poll session for new messages")
    p_poll.add_argument("session_id", help="Session ID to watch")
    p_poll.add_argument("--after-msg", type=int,
                        help="Wait for message id > N (default: current max)")
    p_poll.add_argument("--after-ts", type=float,
                        help="Wait for session with started_at > N")
    p_poll.add_argument("-t", "--timeout", type=int, default=120,
                        help="Seconds before giving up (default 120)")
    p_poll.add_argument("-i", "--interval", type=int, default=1,
                        help="Seconds between polls (default 1)")
    p_poll.add_argument("-f", "--follow", action="store_true",
                        help="Keep polling after first result (default: quit)")
    p_poll.add_argument("--json", action="store_true",
                        help="Output as JSON")

    p_msgs = sub.add_parser("messages", help="Query messages")
    p_msgs.add_argument("session_id", help="Session ID")
    p_msgs.add_argument("-r", "--role", help="Filter by role")
    p_msgs.add_argument("-x", "--exclude-role",
                        help="Exclude this role (e.g. session_meta)")
    p_msgs.add_argument("-l", "--limit", type=int, default=1,
                        help="Max messages (default 1)")
    p_msgs.add_argument("-o", "--offset", type=int, default=0,
                        help="Skip first N results")
    p_msgs.add_argument("--order", choices=["desc", "asc"], default="desc",
                        help="Sort order: desc (latest first) or asc")
    p_msgs.add_argument("--json", action="store_true",
                        help="Output as JSON array")

    # Shorthand: bare session_id → info
    if len(sys.argv) > 1 and sys.argv[1] not in (
        "list", "info", "find", "poll", "messages", "-h", "--help",
    ) and not sys.argv[1].startswith("-"):
        sys.argv.insert(1, "info")

    args = parser.parse_args()

    if args.command == "list":
        _cmd_list(args)
    elif args.command == "info":
        _cmd_info(args)
    elif args.command == "find":
        _cmd_find(args)
    elif args.command == "poll":
        _cmd_poll(args)
    elif args.command == "messages":
        _cmd_messages(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
