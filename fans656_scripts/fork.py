#!/usr/bin/env python3
"""Replay or continue a past Hermes conversation.

Load the recorded conversation for a session and produce a curl-ready
JSON request body that mirrors exactly what was sent to the API.  No
network calls are made by default — you review the payload first, then
optionally send it with ``--send``.

Examples::

    fork.py <sid>                              # curl-ready JSON
    fork.py <sid> --pretty                     # message tail preview
    fork.py <sid> --send --pretty              # send + human-readable response
    fork.py <sid> --drop 1 "worth a diary?"    # drop last, append probe
    fork.py <sid> --drop 2 --send --pretty     # redo last 2 turns
    fork.py <sid> --rebuild --send             # DB fallback
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Response callback ──────────────────────────────────────────────────
# A callback is loaded via --on-response <path>:<funcname> and receives:
#   callback(response: dict, ctx: ResponseContext) -> "accept" | "retry"
#
# The caller can call ctx.add_message(msg) to append messages before a retry.

class ResponseContext:
    """Mutable context passed to --on-response callbacks."""
    def __init__(self, messages: List[Dict[str, Any]]):
        self._messages = list(messages)
        self._pending: List[Dict[str, Any]] = []

    def add_message(self, msg: Dict[str, Any]) -> None:
        self._pending.append(msg)

    @property
    def messages(self) -> List[Dict[str, Any]]:
        return list(self._messages) + list(self._pending)

    def commit_pending(self) -> None:
        self._messages.extend(self._pending)
        self._pending = []


def load_callback(spec: str) -> Optional[Callable]:
    """Load a callback from '<path>:<funcname>' or '<modulename>:<funcname>'."""
    try:
        path, funcname = spec.rsplit(":", 1)
    except ValueError:
        print(f"fork: --on-response: expected '<path>:<funcname>', got '{spec}'", file=sys.stderr)
        return None

    # Try as file path first
    p = Path(path)
    if p.exists():
        modname = p.stem.replace("-", "_")
        spec_loader = importlib.util.spec_from_file_location(modname, p)
        if spec_loader is None:
            print(f"fork: could not load module from '{p}'", file=sys.stderr)
            return None
        mod = importlib.util.module_from_spec(spec_loader)
        sys.modules[modname] = mod
        spec_loader.loader.exec_module(mod)
    else:
        print(f"fork: callback file not found: '{p}'", file=sys.stderr)
        return None

    fn = getattr(mod, funcname, None)
    if not callable(fn):
        print(f"fork: '{funcname}' is not callable in '{p}'", file=sys.stderr)
        return None
    print(f"fork: loaded response callback: {p}:{funcname}", file=sys.stderr)
    return fn

# ── Path ────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _hermes_home() -> Path:
    return Path(os.getenv("HERMES_HOME", str(Path.home() / ".hermes")))


_CAPTURED_ROOT = _hermes_home() / "sessions" / "cache" / "captured"
_STATE_DB = _hermes_home() / "state.db"
DEFAULT_MODEL = "deepseek-v4-flash"


# ── ForkSource ───────────────────────────────────────────────────────────

@dataclass
class ForkSource:
    kind: str                         # "captured" | "rebuild"
    session_id: str
    model: str
    messages: List[Dict[str, Any]]    # API-format messages
    assistant_final: str              # raw text of last assistant response
    api_kwargs: Optional[Dict[str, Any]] = None  # full curl-ready body (captured only)
    usage: Optional[Dict[str, Any]] = None       # from resp.json (captured only)


# ── Load: captured ──────────────────────────────────────────────────────

def _load_captured(session_id: str, seq: Optional[int] = None) -> Optional[ForkSource]:
    d = _CAPTURED_ROOT / session_id
    if not d.is_dir():
        return None
    if seq is not None:
        req_path = d / f"{seq:03d}_req.json"
        if not req_path.exists():
            print(f"fork: --capture {seq}: {req_path.name} not found", file=sys.stderr)
            return None
        reqs = [req_path]
    else:
        reqs = sorted(d.glob("*_req.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not reqs:
            return None
    seq = int(reqs[0].stem.split("_")[0])
    try:
        req = json.loads(reqs[0].read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"fork: failed to read {reqs[0].name}: {exc}", file=sys.stderr)
        return None

    resp_path = d / f"{seq:03d}_resp.json"
    assistant_final = ""
    usage: Optional[Dict[str, Any]] = None
    if resp_path.exists():
        try:
            resp = json.loads(resp_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"fork: failed to parse {resp_path.name}: {exc}", file=sys.stderr)
            resp = None
        if isinstance(resp, dict):
            inner = resp.get("response") or resp
            choices = inner.get("choices")
            if isinstance(choices, list) and len(choices) > 0:
                first = choices[0]
                assistant_final = (
                    first.get("message", {}).get("content", "")
                    if isinstance(first, dict) else ""
                ) or ""
            usage = inner.get("usage")

    messages = req.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return None

    model = req.get("model", DEFAULT_MODEL)
    api_kwargs = {k: v for k, v in req.items() if k not in ("messages",)}

    return ForkSource(
        kind="captured",
        session_id=session_id,
        model=model,
        messages=messages,
        assistant_final=assistant_final,
        api_kwargs=api_kwargs,
        usage=usage,
    )


# ── Load: rebuild ───────────────────────────────────────────────────────

def _sanitize_message(msg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert internal-format message to API format (strip Hermes internals)."""
    if not isinstance(msg, dict):
        return None
    role = msg.get("role")
    if not role or role in ("session_meta", "compression_summary"):
        return None
    clean: Dict[str, Any] = {"role": role}

    if "content" in msg:
        c = msg["content"]
        if isinstance(c, str):
            clean["content"] = c.strip() if c else c
        elif c is not None:
            clean["content"] = json.dumps(c, ensure_ascii=False, default=str, separators=(",", ":"))
    elif role == "assistant" and msg.get("tool_calls"):
        clean["content"] = None

    if role == "assistant" and msg.get("tool_calls"):
        tcs = []
        for tc in msg["tool_calls"]:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {})
            args = fn.get("arguments", "")
            if isinstance(args, str) and args:
                try:
                    args_obj = json.loads(args)
                    args = json.dumps(args_obj, separators=(",", ":"), sort_keys=True)
                except Exception:
                    pass
            tcs.append({
                "id": tc.get("id") or tc.get("call_id", ""),
                "type": "function",
                "function": {"name": fn.get("name", ""), "arguments": args},
            })
        if tcs:
            clean["tool_calls"] = tcs

    if role == "tool":
        clean["tool_call_id"] = msg.get("tool_call_id", "")

    if msg.get("reasoning_content"):
        clean["reasoning_content"] = msg["reasoning_content"]

    return clean


def _load_rebuild(session_id: str) -> Optional[ForkSource]:
    if not _STATE_DB.exists():
        return None
    try:
        from hermes_state import SessionDB
        db = SessionDB(_STATE_DB)
    except Exception as exc:
        print(f"fork: failed to open SessionDB: {exc}", file=sys.stderr)
        return None

    session = db.get_session(session_id)
    if not session:
        return None

    system_prompt = session.get("system_prompt") or ""
    raw_msgs = db.get_messages_as_conversation(session_id) or []

    messages: List[Dict[str, Any]] = []
    for msg in raw_msgs:
        clean = _sanitize_message(msg)
        if clean is None:
            continue
        has_content = clean.get("content") is not None
        has_tc = bool(clean.get("tool_calls"))
        if has_content or has_tc:
            messages.append(clean)

    if system_prompt and messages:
        messages.insert(0, {"role": "system", "content": system_prompt})

    # Last assistant response
    assistant_final = ""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            assistant_final = msg["content"]
            break

    return ForkSource(
        kind="rebuild",
        session_id=session_id,
        model=session.get("model") or DEFAULT_MODEL,
        messages=messages,
        assistant_final=assistant_final,
    )


# ── Load (unified) ──────────────────────────────────────────────────────

def load_source(session_id: str, *, rebuild: bool = False, capture_seq: Optional[int] = None) -> Optional[ForkSource]:
    if not rebuild:
        src = _load_captured(session_id, seq=capture_seq)
        if src:
            return src
    return _load_rebuild(session_id)


# ── Build messages ──────────────────────────────────────────────────────

def build_messages(
    source: ForkSource,
    *,
    drop: int = 0,
    append_text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Build the complete message array: source messages + assistant_final,
    then apply --drop from the tail, then optionally append new text."""
    messages = list(source.messages)

    if source.assistant_final:
        messages.append({"role": "assistant", "content": source.assistant_final})

    if drop > 0:
        messages = messages[:-drop] if drop < len(messages) else []

    if append_text:
        messages.append({"role": "user", "content": append_text})

    return messages


def build_request(source: ForkSource, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build curl-ready request body. Forces non-streaming for simple response parsing."""
    if source.kind == "captured" and source.api_kwargs:
        body = dict(source.api_kwargs)
        body["messages"] = messages
        body["model"] = body.get("model") or source.model
    else:
        body = {"model": source.model or DEFAULT_MODEL, "messages": messages}
    body["stream"] = False
    body.pop("stream_options", None)
    return body


# ── Output ──────────────────────────────────────────────────────────────

def _print_tail(source: ForkSource, n: int) -> None:
    msgs = source.messages
    total = len(msgs)
    tail = msgs[-n:] if n < total else msgs
    for i, m in enumerate(tail):
        idx = total - len(tail) + i + 1
        role = m.get("role", "?")
        c = _msg_preview(m)
        print(f"  msg-{idx:<5} [{role:<11}] {c}")


def _msg_preview(m: Dict[str, Any]) -> str:
    tc = m.get("tool_calls")
    if tc:
        fns = [t.get("function", {}).get("name", "?") for t in tc]
        return f"*tool_calls* {', '.join(fns)}"
    if m.get("role") == "tool":
        return str(m.get("content", ""))[:120]
    return str(m.get("content", ""))[:120]


def _msg_header(m: Dict[str, Any]) -> str:
    role = m.get("role", "?")
    line = "=" if role == "system" else "-"
    return f"{role}\n{line * 80}"


def _msg_body(m: Dict[str, Any]) -> str:
    role = m.get("role", "")
    tc = m.get("tool_calls")
    rc = m.get("reasoning_content", "")

    lines = []
    if rc:
        lines.append("")
        lines.append("*thinking*")
        lines.append("")
        for line in rc.split("\n")[:5]:
            lines.append(line)
        if len(rc.split("\n")) > 5:
            lines.append("...")

    if tc:
        lines.append("")
        lines.append("*tool_calls*")
        lines.append("")
        for t in tc:
            fn = t.get("function", {})
            name = fn.get("name", "?")
            args_str = fn.get("arguments", "")
            try:
                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                args_str = json.dumps(args, ensure_ascii=False, indent=2)
            except Exception:
                pass
            lines.append(f"{name}({args_str})")
    elif role == "tool":
        content = m.get("content", "")
        lines.append("")
        lines.append("```json")
        if isinstance(content, str) and len(content) > 500:
            content = content[:500] + "\n... (truncated)"
        lines.append(str(content))
        lines.append("```")
    else:
        content = str(m.get("content", ""))
        if role == "system":
            sp_lines = [l for l in content.split("\n") if l.strip()][:3]
            lines.extend(sp_lines)
            lines.append("...")
        elif content:
            lines.append(content)

    return "\n".join(lines)


def _print_messages_markdown(messages: List[Dict[str, Any]], tail_n: int = 0) -> None:
    msgs = messages[-tail_n:] if tail_n > 0 else messages
    for m in msgs:
        print()
        print(_msg_header(m))
        print()
        print(_msg_body(m))


def _print_request(curl_body: Dict[str, Any]) -> None:
    print(json.dumps(curl_body, ensure_ascii=False, default=str, separators=(",", ":")))


def _print_pretty(source: ForkSource, messages: List[Dict[str, Any]],
                  response_content: str = "", usage: Optional[Dict[str, Any]] = None) -> None:
    """Markdown-style output with === frontmatter."""
    bar = "=" * 80
    print(bar)
    print(f"session: {source.session_id}")
    print(f"source: {source.kind}")
    print(f"model: {source.model}")
    print(f"messages_in: {len(source.messages)}")
    print(f"messages_out: {len(messages)}")
    if usage:
        hit = usage.get("cache_hit_tokens", 0) or 0
        miss = usage.get("cache_miss_tokens", 0) or 0
        total = hit + miss
        if total > 0:
            print(f"cache_hit_pct: {hit / total * 100:.1f}")
        print(f"cache_hit_tokens: {hit}")
        print(f"cache_miss_tokens: {miss}")
        print(f"prompt_tokens: {usage.get('prompt_tokens', 'N/A')}")
        print(f"completion_tokens: {usage.get('completion_tokens', 'N/A')}")
    elif source.kind == "rebuild":
        print("cache_hit_pct: N/A")
    print(bar)

    if response_content:
        print()
        print("Response")
        print("-" * 80)
        for line in response_content.split("\n"):
            print(line)
    else:
        _print_messages_markdown(messages, tail_n=5)


# ── Send ────────────────────────────────────────────────────────────────

def _send_captured(
    source: ForkSource,
    messages: List[Dict[str, Any]],
    response_callback: Optional[Callable] = None,
    max_retries: int = 1,
) -> Tuple[Any, str, Dict[str, Any]]:
    """POST exact HTTP body bytes to DeepSeek. Returns (response, content, usage).

    If response_callback is set, it's called with (response, ResponseContext) after
    each API call. Return "accept" to finish or "retry" to loop (up to max_retries).
    """
    api_key = _deepseek_api_key()
    body = build_request(source, messages)

    # -- Response callback loop ------------------------------------------------
    current_messages = list(messages)
    import urllib.request
    import urllib.error
    for attempt in range(max_retries + 1):
        if response_callback:
            print(f"fork: callback attempt {attempt + 1}/{max_retries + 1}", file=sys.stderr)
        # Build and send request
        b = dict(body)
        b["messages"] = current_messages
        bj = json.dumps(b, ensure_ascii=False, default=str, separators=(",", ":"))
        try:
            raw = urllib.request.urlopen(urllib.request.Request(
                "https://api.deepseek.com/v1/chat/completions",
                data=bj.encode("utf-8"),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            ), timeout=120).read().decode("utf-8")
            resp = json.loads(raw)
        except (urllib.error.HTTPError, json.JSONDecodeError, ValueError) as exc:
            print(f"fork: request failed: {exc}", file=sys.stderr)
            return None, "", {}

        choices = resp.get("choices", [])
        if choices and len(choices) > 0:
            content = choices[0].get("message", {}).get("content", "") or ""
        usage_data = resp.get("usage") or {}
        usage = _extract_usage_from_dict({"usage": usage_data} if usage_data else {})

        # If no callback or last attempt, accept
        if not response_callback or attempt >= max_retries:
            break

        # Ask callback: accept or retry
        ctx = ResponseContext(current_messages)
        try:
            decision = response_callback(resp, ctx)
        except Exception as exc:
            print(f"fork: response_callback error: {exc}", file=sys.stderr)
            break

        if decision == "retry":
            ctx.commit_pending()
            n_pending = len(ctx._pending) if hasattr(ctx, '_pending') else 0
            print(f"fork: callback returned retry ({n_pending} tool responses queued)", file=sys.stderr)
            # Add assistant tool_calls message before the tool responses
            assistant_tc = choices[0].get("message", {}).get("tool_calls")
            if assistant_tc:
                current_messages = ctx.messages
                # Insert assistant's tool_calls message right before the tool responses
                assistant_msg = {"role": "assistant", "tool_calls": assistant_tc}
                tool_count = len(assistant_tc)
                current_messages.insert(-tool_count, assistant_msg)
            else:
                current_messages = ctx.messages
            continue
        # "accept" → break
        print(f"fork: callback returned accept", file=sys.stderr)
        break

    return {"choices": [{"message": {"content": content}}]}, content, usage


def _send_rebuild(
    source: ForkSource,
    messages: List[Dict[str, Any]],
    response_callback: Optional[Callable] = None,
    max_retries: int = 1,
) -> Tuple[Any, str, Dict[str, Any]]:
    """POST to DeepSeek without captured api_kwargs. Returns (response, content, usage).

    If response_callback is set, it's called with (response, ResponseContext) after
    each API call. Return "accept" to finish or "retry" to loop (up to max_retries).
    """
    api_key = _deepseek_api_key()

    current_messages = list(messages)
    for attempt in range(max_retries + 1):
        body = build_request(source, current_messages)
        body_json = json.dumps(body, ensure_ascii=False, default=str, separators=(",", ":"))

        import urllib.request
        import urllib.error
        try:
            raw = urllib.request.urlopen(urllib.request.Request(
                "https://api.deepseek.com/v1/chat/completions",
                data=body_json.encode("utf-8"),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            ), timeout=120).read().decode("utf-8")
            resp = json.loads(raw)
        except (urllib.error.HTTPError, json.JSONDecodeError, ValueError) as exc:
            print(f"fork: request failed: {exc}", file=sys.stderr)
            return None, "", {}

        choices = resp.get("choices", [])
        content = ""
        if choices and len(choices) > 0:
            content = choices[0].get("message", {}).get("content", "") or ""
        usage_data = resp.get("usage") or {}
        usage = _extract_usage_from_dict({"usage": usage_data} if usage_data else {})

        if not response_callback or attempt >= max_retries:
            break

        print(f"fork: callback attempt {attempt + 1}/{max_retries + 1}", file=sys.stderr)
        ctx = ResponseContext(current_messages)
        try:
            decision = response_callback(resp, ctx)
        except Exception as exc:
            print(f"fork: response_callback error: {exc}", file=sys.stderr)
            break

        if decision == "retry":
            ctx.commit_pending()
            n_pending = len(ctx._pending) if hasattr(ctx, '_pending') else 0
            print(f"fork: callback returned retry ({n_pending} tool responses queued)", file=sys.stderr)
            assistant_tc = choices[0].get("message", {}).get("tool_calls")
            if assistant_tc:
                current_messages = ctx.messages
                tool_count = len(assistant_tc)
                current_messages.insert(-tool_count, {"role": "assistant", "tool_calls": assistant_tc})
            else:
                current_messages = ctx.messages
            continue
        print(f"fork: callback returned accept", file=sys.stderr)
        break

    return {"choices": [{"message": {"content": content}}]}, content, usage


def _extract_usage_from_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    u = data.get("usage", {})
    if not isinstance(u, dict):
        return {}
    return {
        "prompt_tokens": u.get("prompt_tokens", 0) or 0,
        "completion_tokens": u.get("completion_tokens", 0) or 0,
        "cache_hit_tokens": u.get("prompt_cache_hit_tokens", 0) or 0,
        "cache_miss_tokens": u.get("prompt_cache_miss_tokens", 0) or 0,
    }


def _hit_pct(usage: Dict[str, Any]) -> str:
    hit = usage.get("cache_hit_tokens", 0) or 0
    miss = usage.get("cache_miss_tokens", 0) or 0
    if hit + miss > 0:
        return f"{hit / (hit + miss) * 100:.1f}"
    return "N/A"


def _deepseek_api_key() -> str:
    env_file = _hermes_home() / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("DEEPSEEK_API_KEY", "")


# ── CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("session_id", nargs="?", help="Session ID (omit with --new)")
    parser.add_argument("text", nargs="?", help="Text to append as user message")
    parser.add_argument("--new", action="store_true",
                        help="New session mode (no session_id required)")
    parser.add_argument("--drop", type=int, default=0,
                        help="Drop last N messages from the full conversation")
    parser.add_argument("--tail", type=int,
                        help="Show last N messages (no network)")
    parser.add_argument("--send", action="store_true",
                        help="Send request to DeepSeek")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force DB rebuild (ignore captured files)")
    parser.add_argument("--capture", type=int, default=None,
                        help="Use specific captured req file by seq (e.g. --capture 134)")
    parser.add_argument("--pretty", action="store_true",
                        help="Markdown-style output with frontmatter")
    parser.add_argument("--on-response", metavar="PATH:FUNC",
                        help="Load a callback: fn(response, ctx) -> 'accept'|'retry'")
    parser.add_argument("--max-retries", type=int, default=1,
                        help="Max retries when callback returns 'retry' (default: 1)")

    args = parser.parse_args()

    # ── --new mode ──────────────────────────────────────────────────────
    if args.new:
        user_text = args.text or args.session_id or ""
        messages: List[Dict[str, Any]] = []
        if user_text:
            messages.append({"role": "user", "content": user_text})
        body = {"model": DEFAULT_MODEL, "messages": messages}

        if args.send:
            body_json = json.dumps(body, ensure_ascii=False, default=str, separators=(",", ":"))
            api_key = _deepseek_api_key()
            import urllib.request
            req = urllib.request.Request(
                "https://api.deepseek.com/v1/chat/completions",
                data=body_json.encode("utf-8"),
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
            resp_data = json.loads(urllib.request.urlopen(req, timeout=120).read().decode("utf-8"))
            content = ""
            choices = resp_data.get("choices")
            if choices and len(choices) > 0:
                content = choices[0].get("message", {}).get("content", "") or ""
            if args.pretty:
                usage = _extract_usage_from_dict(resp_data)
                bar = "=" * 80
                print(bar)
                print("session: (new)")
                print(f"model: {DEFAULT_MODEL}")
                print(f"cache_hit_pct: {_hit_pct(usage)}")
                print(f"cache_hit_tokens: {usage.get('cache_hit_tokens', 'N/A')}")
                print(f"cache_miss_tokens: {usage.get('cache_miss_tokens', 'N/A')}")
                print(bar)
                print()
                print("Response")
                print("-" * 80)
                for line in content.split("\n"):
                    print(line)
            else:
                print(json.dumps(resp_data, ensure_ascii=False, indent=2))
        else:
            _print_request(body)
        return

    # ── Normal mode (requires session_id) ───────────────────────────────
    if not args.session_id:
        parser.error("session_id is required (or use --new)")
    sid: str = args.session_id

    # ── Load source ─────────────────────────────────────────────────────
    source = load_source(sid, rebuild=args.rebuild, capture_seq=args.capture)
    if source is None:
        print(f"No source found for session: {sid}", file=sys.stderr)
        sys.exit(1)

    # ── --tail mode ─────────────────────────────────────────────────────
    if args.tail:
        _print_tail(source, args.tail)
        return

    # ── Build messages ──────────────────────────────────────────────────
    messages = build_messages(
        source,
        drop=args.drop,
        append_text=args.text,
    )

    # ── --send ──────────────────────────────────────────────────────────
    if args.send:
        cb = load_callback(args.on_response) if args.on_response else None
        if source.kind == "captured":
            resp_data, content, usage = _send_captured(source, messages, response_callback=cb, max_retries=args.max_retries)
        else:
            resp_data, content, usage = _send_rebuild(source, messages, response_callback=cb, max_retries=args.max_retries)

        if args.pretty:
            _print_pretty(source, messages, content, usage)
        else:
            resp_data["usage"] = usage
            print(json.dumps(resp_data, ensure_ascii=False, indent=2))
        return

    # ── Default: curl-ready JSON ────────────────────────────────────────
    if args.pretty:
        _print_pretty(source, messages)
    else:
        body = build_request(source, messages)
        _print_request(body)


if __name__ == "__main__":
    main()
