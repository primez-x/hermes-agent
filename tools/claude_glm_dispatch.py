#!/usr/bin/env python3
"""Native Hermes tool and CLI runner for structured claude-glm dispatch.

Hermes remains the dispatcher. This module launches Claude Code through the
local ``claude-glm`` wrapper, captures ``stream-json`` output to disk, extracts
recoverable session metadata, and exposes a small native tool that uses Hermes'
tracked background process registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

from hermes_cli.config import get_hermes_home
from tools.registry import registry, tool_error
from tools.terminal_tool import terminal_tool


HERMES_HOME = get_hermes_home()
DISPATCH_ROOT = HERMES_HOME / "claude-glm-dispatch"
WRAPPER_PATH = HERMES_HOME / "bin" / "claude-glm-dispatch"
DEFAULT_CLAUDE_GLM_BIN = os.getenv(
    "CLAUDE_GLM_BIN", str(Path.home() / ".local/bin/claude-glm")
)
DEFAULT_DISPATCH_TURNS = {
    "quick": 40,
    "small": 70,
    "normal": 100,
    "complex": 150,
    "hard": 200,
}


def _json_default(value: Any) -> str:
    return str(value)


def _load_dispatch_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception:
        return {}
    section = config.get("claude_glm_dispatch", {})
    return section if isinstance(section, dict) else {}


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return default
    return candidate if candidate > 0 else default


def _normalize_max_turns(value: Any) -> int | str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.lower() == "auto":
            return "auto"
        try:
            return int(stripped)
        except ValueError:
            return "auto"
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _prompt_has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _auto_turn_tier(prompt: str, workdir: str) -> str:
    # Classify by the curated task prompt. Workdir is only a stable project
    # identity signal; including arbitrary temp paths makes pytest directories
    # such as /tmp/pytest-* accidentally trigger the complex tier.
    context = str(prompt or "").lower()
    workdir_l = str(workdir or "").lower()
    if workdir_l.endswith("/titan") or workdir_l == "/home/user/titan" or workdir_l.startswith("/home/user/titan/"):
        return "complex"

    complex_terms = (
        "all tests",
        "database",
        "debug",
        "deployment",
        "end-to-end",
        "e2e",
        "failing test",
        "failing tests",
        "flaky",
        "integration",
        "migration",
        "multi-file",
        "performance",
        "production",
        "pytest",
        "race",
        "refactor",
        "schema",
        "service",
        "test suite",
    )
    small_terms = (
        "formatting",
        "lint",
        "one-line",
        "quick fix",
        "simple patch",
        "single-file",
        "small fix",
        "typo",
    )
    quick_terms = (
        "analysis",
        "diagnose",
        "explain",
        "inspect",
        "logs",
        "read-only",
        "review",
        "status",
        "summarize",
    )

    if _prompt_has_any(context, complex_terms):
        return "complex"
    if _prompt_has_any(context, small_terms):
        return "small"
    if _prompt_has_any(context, quick_terms):
        return "quick"
    return "normal"


def resolve_max_turns(
    max_turns: Any = None,
    *,
    prompt: str = "",
    workdir: str = "",
    config: dict[str, Any] | None = None,
) -> tuple[int, str]:
    """Resolve explicit/config/auto max-turn policy for claude-glm dispatch."""
    dispatch_config = config if isinstance(config, dict) else _load_dispatch_config()
    hard = _positive_int(
        dispatch_config.get("hard_max_turns"), DEFAULT_DISPATCH_TURNS["hard"]
    )

    requested = _normalize_max_turns(max_turns)
    if isinstance(requested, int):
        resolved = min(max(1, requested), hard)
        suffix = ":clamped" if resolved != requested else ""
        return resolved, f"explicit{suffix}"

    configured = _normalize_max_turns(dispatch_config.get("max_turns"))
    if requested is None and isinstance(configured, int):
        resolved = min(max(1, configured), hard)
        suffix = ":clamped" if resolved != configured else ""
        return resolved, f"config{suffix}"

    tier = _auto_turn_tier(prompt, workdir)
    default_turns = DEFAULT_DISPATCH_TURNS[tier]
    if tier == "normal":
        configured_turns = dispatch_config.get("normal_max_turns")
        if configured_turns is None:
            configured_turns = dispatch_config.get("default_max_turns")
    else:
        configured_turns = dispatch_config.get(f"{tier}_max_turns")
    target = _positive_int(configured_turns, default_turns)
    resolved = min(target, hard)
    suffix = ":clamped" if resolved != target else ""
    return resolved, f"auto:{tier}{suffix}"


def _new_dispatch_dir(prompt: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    digest = hashlib.sha256(f"{time.time_ns()}\0{prompt}".encode("utf-8")).hexdigest()[
        :8
    ]
    return DISPATCH_ROOT / f"{stamp}_{digest}"


def _tool_detail(name: str, tool_input: Any) -> str:
    if isinstance(tool_input, dict):
        for key in (
            "command",
            "file_path",
            "path",
            "pattern",
            "query",
            "url",
            "description",
        ):
            value = tool_input.get(key)
            if value:
                return str(value)[:240]
        try:
            return json.dumps(tool_input, ensure_ascii=False, sort_keys=True)[:240]
        except TypeError:
            return str(tool_input)[:240]
    if tool_input:
        return str(tool_input)[:240]
    return name


def _content_blocks(event: dict[str, Any]) -> Iterable[dict[str, Any]]:
    message = event.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    yield block

    stream_event = event.get("event")
    if isinstance(stream_event, dict):
        block = stream_event.get("content_block")
        if isinstance(block, dict):
            yield block


def _update_summary(
    summary: dict[str, Any], event: dict[str, Any]
) -> list[dict[str, str]]:
    new_tools: list[dict[str, str]] = []

    session_id = event.get("session_id")
    if session_id and not summary.get("session_id"):
        summary["session_id"] = str(session_id)

    event_type = event.get("type")
    if event_type == "result":
        subtype = event.get("subtype") or event.get("status")
        if subtype:
            summary["status"] = str(subtype)
        for key in (
            "session_id",
            "result",
            "num_turns",
            "total_cost_usd",
            "duration_ms",
            "stop_reason",
            "terminal_reason",
            "usage",
            "modelUsage",
        ):
            if key in event and event[key] is not None:
                summary[key] = event[key]

    if event_type in {"error", "error_event"}:
        summary["status"] = "error"
        summary["error"] = event.get("error") or event.get("message") or event

    for block in _content_blocks(event):
        if block.get("type") != "tool_use":
            continue
        name = str(block.get("name") or "tool")
        detail = _tool_detail(name, block.get("input"))
        item = {"name": name, "detail": detail}
        summary["tools"].append(item)
        new_tools.append(item)

    return new_tools


def parse_stream_json_lines(lines: Iterable[str]) -> dict[str, Any]:
    """Parse Claude Code ``stream-json`` lines into a compact run summary."""
    summary: dict[str, Any] = {
        "session_id": None,
        "status": "unknown",
        "tools": [],
        "result": "",
        "invalid_json_lines": 0,
    }
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            summary["invalid_json_lines"] += 1
            continue
        if isinstance(event, dict):
            _update_summary(summary, event)
    return summary


def build_claude_glm_command(
    *,
    prompt_file: Path,
    log_dir: Path,
    max_turns: int = DEFAULT_DISPATCH_TURNS["normal"],
    effort: str = "max",
    resume_session_id: str = "",
    continue_latest: bool = False,
    wrapper_path: Path = WRAPPER_PATH,
) -> str:
    """Build the Hermes terminal command for the wrapper.

    The curated prompt stays in ``prompt_file``. The Hermes-visible shell
    command contains paths and options only, not the prompt body.
    """
    parts = [
        shlex.quote(str(wrapper_path)),
        "--prompt-file",
        shlex.quote(str(prompt_file)),
        "--log-dir",
        shlex.quote(str(log_dir)),
        "--max-turns",
        str(int(max_turns)),
        "--effort",
        shlex.quote(str(effort)),
    ]
    if resume_session_id:
        parts.extend(["--resume", shlex.quote(str(resume_session_id))])
    elif continue_latest:
        parts.append("--continue")
    return " ".join(parts)


def build_claude_process_args(
    *,
    prompt: str,
    max_turns: int,
    effort: str,
    resume_session_id: str = "",
    continue_latest: bool = False,
    claude_bin: str = DEFAULT_CLAUDE_GLM_BIN,
) -> list[str]:
    """Build the claude argv for a hot, steerable stream-json session.

    ``prompt`` is intentionally unused: in stream-json input mode the task is
    fed as the first user-message envelope on stdin (see
    ``run_claude_glm_dispatch``), which keeps the session alive and steerable
    across multiple messages instead of exiting after one response.
    """
    args = [
        claude_bin,
        "-p",
        "--verbose",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--max-turns",
        str(int(max_turns)),
        "--effort",
        effort,
    ]
    if resume_session_id:
        args.extend(["--resume", resume_session_id])
    elif continue_latest:
        args.append("--continue")
    return args


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def run_claude_glm_dispatch(
    *,
    prompt_file: Path,
    log_dir: Path,
    max_turns: int | str | None,
    effort: str,
    resume_session_id: str = "",
    continue_latest: bool = False,
    claude_bin: str = DEFAULT_CLAUDE_GLM_BIN,
    workdir: str | None = None,
    idle_timeout: int = 300,
) -> int:
    """Run claude-glm as a persistent, steerable stream-json session.

    Launches the initial task, then stays hot (stdin open) so additional
    messages can be appended/steered at the next tool-call break. Steering
    envelopes are dropped as JSON files in ``log_dir/steer/`` (one per message;
    see ``claude_glm_steer``) and forwarded to claude's stdin. After
    ``idle_timeout`` seconds with no stdout activity following a completed turn,
    the session winds itself down (stdin closed -> claude exits cleanly).
    """
    import threading

    prompt = prompt_file.read_text(encoding="utf-8")
    resolved_max_turns, max_turns_source = resolve_max_turns(
        max_turns, prompt=prompt, workdir=workdir or ""
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    events_path = log_dir / "events.jsonl"
    summary_path = log_dir / "summary.json"
    final_path = log_dir / "final.txt"
    steer_dir = log_dir / "steer"
    steer_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "hot").write_text("1", encoding="utf-8")  # live-session marker

    args = build_claude_process_args(
        prompt="",
        max_turns=resolved_max_turns,
        effort=effort,
        resume_session_id=resume_session_id,
        continue_latest=continue_latest,
        claude_bin=claude_bin,
    )
    summary: dict[str, Any] = {
        "session_id": None,
        "status": "running",
        "tools": [],
        "result": "",
        "invalid_json_lines": 0,
        "started_at": time.time(),
        "log_dir": str(log_dir),
        "requested_max_turns": max_turns,
        "resolved_max_turns": resolved_max_turns,
        "max_turns_source": max_turns_source,
        "steerable": True,
        "idle_timeout": idle_timeout,
    }
    seen_session = ""
    seen_tools: set[tuple[str, str]] = set()
    lock = threading.Lock()
    last_activity = {"t": time.time()}
    armed = {"v": False}
    stop = threading.Event()

    def _touch() -> None:
        with lock:
            last_activity["t"] = time.time()

    init_msg = (
        json.dumps(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                },
            }
        )
        + "\n"
    )

    print(f"[claude-glm] started log_dir={log_dir} (steerable hot session)", flush=True)
    proc = subprocess.Popen(
        args,
        cwd=workdir or None,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdin is not None and proc.stdout is not None

    try:
        proc.stdin.write(init_msg)
        proc.stdin.flush()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[claude-glm] failed writing initial message: {exc}", flush=True)
    _touch()

    def _forward_steers() -> None:
        while not stop.is_set():
            for entry in sorted(steer_dir.glob("*.json")):
                try:
                    payload = entry.read_text(encoding="utf-8").strip()
                    if payload.startswith("{"):
                        try:
                            envelope = json.loads(payload).get("envelope") or json.loads(payload)
                        except json.JSONDecodeError:
                            envelope = None
                    else:
                        envelope = None
                    if not envelope:
                        envelope = {
                            "type": "user",
                            "message": {
                                "role": "user",
                                "content": [{"type": "text", "text": payload}],
                            },
                        }
                    proc.stdin.write(json.dumps(envelope, ensure_ascii=False) + "\n")
                    proc.stdin.flush()
                    _touch()
                    print(f"[claude-glm] steer forwarded: {entry.name}", flush=True)
                except Exception as exc:  # pragma: no cover - defensive
                    print(f"[claude-glm] steer error {entry.name}: {exc}", flush=True)
                finally:
                    try:
                        entry.unlink()
                    except OSError:
                        pass
            stop.wait(1.0)

    def _idle_watch() -> None:
        while not stop.is_set():
            stop.wait(5.0)
            if stop.is_set():
                break
            with lock:
                idle = time.time() - last_activity["t"]
                is_armed = armed["v"]
            if is_armed and idle >= idle_timeout:
                print(
                    f"[claude-glm] idle {int(idle)}s >= {idle_timeout}s, winding down",
                    flush=True,
                )
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                stop.set()
                return

    threading.Thread(target=_forward_steers, daemon=True).start()
    threading.Thread(target=_idle_watch, daemon=True).start()

    exit_code = 0
    events_file = events_path.open("a", encoding="utf-8")
    try:
        for raw in proc.stdout:
            events_file.write(raw)
            events_file.flush()
            _touch()
            line = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                summary["invalid_json_lines"] += 1
                continue
            if not isinstance(event, dict):
                continue
            new_tools = _update_summary(summary, event)
            if summary.get("session_id") and summary["session_id"] != seen_session:
                seen_session = str(summary["session_id"])
                try:
                    (log_dir / "session_id").write_text(seen_session, encoding="utf-8")
                    _write_json(summary_path, summary)
                except OSError:
                    pass
                print(f"[claude-glm] session_id={seen_session}", flush=True)
            for tool in new_tools:
                key = (tool["name"], tool["detail"])
                if key in seen_tools:
                    continue
                seen_tools.add(key)
                print(f"[claude-glm] tool {tool['name']}: {tool['detail']}", flush=True)
            if event.get("type") == "result":
                with lock:
                    armed["v"] = True
                _touch()
                # Persist immediately: the hot process lingers after the task
                # finishes (to accept steers), so callers must read the result
                # from summary.json rather than waiting for process exit.
                try:
                    _write_json(summary_path, summary)
                except OSError:
                    pass
    finally:
        events_file.close()
        exit_code = proc.wait()
        stop.set()

    summary["exit_code"] = exit_code
    summary["completed_at"] = time.time()
    summary["duration_seconds"] = round(
        summary["completed_at"] - summary["started_at"], 2
    )
    if summary.get("status") == "running":
        summary["status"] = "success" if exit_code == 0 else "error"
    if exit_code != 0 and not summary.get("error"):
        summary["error"] = f"claude-glm exited with code {exit_code}"

    _write_json(summary_path, summary)
    final_path.write_text(str(summary.get("result") or ""), encoding="utf-8")
    try:
        (log_dir / "hot").unlink()
    except OSError:
        pass

    status = summary.get("status") or "unknown"
    turns = summary.get("num_turns")
    cost = summary.get("total_cost_usd")
    suffix = []
    if turns is not None:
        suffix.append(f"turns={turns}")
    if cost is not None:
        suffix.append(f"cost=${cost}")
    if summary.get("session_id"):
        suffix.append(f"session_id={summary['session_id']}")
    suffix_text = " " + " ".join(suffix) if suffix else ""
    print(f"[claude-glm] result {status}{suffix_text}", flush=True)
    if summary.get("error"):
        print(f"[claude-glm] error {summary['error']}", flush=True)
    print(f"[claude-glm] summary={summary_path}", flush=True)
    return int(exit_code or 0)


def claude_glm_dispatch_tool(
    *,
    prompt: str,
    workdir: str = "",
    max_turns: int | str | None = "auto",
    effort: str = "max",
    resume_session_id: str = "",
    continue_latest: bool = False,
    background: bool = True,
    notify_on_complete: bool = True,
    timeout: int = 180,
) -> str:
    prompt = (prompt or "").strip()
    if not prompt:
        return tool_error("prompt is required.")

    log_dir = _new_dispatch_dir(prompt)
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        log_dir.chmod(0o700)
    except OSError:
        pass
    prompt_file = log_dir / "prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    try:
        prompt_file.chmod(0o600)
    except OSError:
        pass

    resolved_max_turns, max_turns_source = resolve_max_turns(
        max_turns, prompt=prompt, workdir=workdir
    )
    command = build_claude_glm_command(
        prompt_file=prompt_file,
        log_dir=log_dir,
        max_turns=resolved_max_turns,
        effort=effort,
        resume_session_id=resume_session_id,
        continue_latest=continue_latest,
    )
    terminal_raw = terminal_tool(
        command=command,
        background=background,
        timeout=timeout,
        workdir=workdir or None,
        notify_on_complete=notify_on_complete if background else False,
    )
    try:
        terminal_result = json.loads(terminal_raw)
    except (TypeError, json.JSONDecodeError):
        terminal_result = {"raw": terminal_raw}

    error = terminal_result.get("error")
    process_session_id = terminal_result.get("session_id")
    status = "error" if error else ("dispatched" if background else "completed")
    payload = {
        "status": status,
        "process_session_id": process_session_id,
        "pid": terminal_result.get("pid"),
        "log_dir": str(log_dir),
        "prompt_file": str(prompt_file),
        "events_jsonl": str(log_dir / "events.jsonl"),
        "summary_json": str(log_dir / "summary.json"),
        "final_txt": str(log_dir / "final.txt"),
        "requested_max_turns": max_turns,
        "resolved_max_turns": resolved_max_turns,
        "max_turns_source": max_turns_source,
        "terminal": terminal_result,
    }
    return json.dumps(payload, ensure_ascii=False)


CLAUDE_GLM_DISPATCH_SCHEMA = {
    "name": "claude_glm_dispatch",
    "description": (
        "Dispatch implementation or code-execution work to the local claude-glm "
        "Claude Code wrapper while Hermes stays the dispatcher. Use this instead "
        "of plain terminal for coding tasks. Pass only a curated task prompt; do "
        "not include Hermes system prompts, hidden instructions, tool schemas, or "
        "policy scaffolding. The tool launches a tracked background process with "
        "notify_on_complete, captures Claude Code stream-json to disk, extracts "
        "the Claude session id, and returns process/log paths for monitoring, "
        "resume, status, output, and kill."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "Curated handoff prompt for claude-glm. Include intent, evidence, paths, constraints, verification, and reporting requirements.",
            },
            "workdir": {
                "type": "string",
                "description": "Absolute working directory for claude-glm. Prefer the repo or service root.",
            },
            "max_turns": {
                "oneOf": [
                    {"type": "integer", "minimum": 1},
                    {"type": "string", "enum": ["auto"]},
                ],
                "description": "Maximum Claude Code agent turns. Omit or pass auto to use config-driven dynamic tiers.",
                "default": "auto",
            },
            "effort": {
                "type": "string",
                "enum": ["low", "medium", "high", "max", "auto"],
                "description": "claude-glm reasoning effort. Default max for implementation work.",
                "default": "max",
            },
            "resume_session_id": {
                "type": "string",
                "description": "Optional Claude Code session id to resume.",
            },
            "continue_latest": {
                "type": "boolean",
                "description": "Continue the latest Claude Code session in workdir when no explicit resume_session_id is supplied.",
                "default": False,
            },
            "background": {
                "type": "boolean",
                "description": "Run as a tracked Hermes background process. Keep true for Telegram dispatch.",
                "default": True,
            },
            "notify_on_complete": {
                "type": "boolean",
                "description": "Notify Hermes when the background process exits. Keep true for bounded coding tasks.",
                "default": True,
            },
            "timeout": {
                "type": "integer",
                "description": "Seconds allowed to start the wrapper command. The child task itself is tracked as a background process.",
                "default": 180,
            },
        },
        "required": ["prompt"],
    },
}


def _handle_claude_glm_dispatch(args: dict[str, Any], **_: Any) -> str:
    return claude_glm_dispatch_tool(
        prompt=str(args.get("prompt") or ""),
        workdir=str(args.get("workdir") or ""),
        max_turns=args.get("max_turns", "auto"),
        effort=str(args.get("effort") or "max"),
        resume_session_id=str(args.get("resume_session_id") or ""),
        continue_latest=bool(args.get("continue_latest", False)),
        background=bool(args.get("background", True)),
        notify_on_complete=bool(args.get("notify_on_complete", True)),
        timeout=int(args.get("timeout") or 180),
    )


def _resolve_hot_log_dir(*, session_id: str = "", log_dir: str = "") -> Path | None:
    """Find the log_dir of a currently-hot (running) steerable dispatch."""
    if log_dir:
        candidate = Path(log_dir).expanduser()
        return candidate if (candidate / "steer").exists() else None
    if session_id:
        for d in sorted(DISPATCH_ROOT.glob("*"), reverse=True):
            if not (d / "hot").exists():
                continue
            sid_file = d / "session_id"
            try:
                sid = (
                    sid_file.read_text(encoding="utf-8").strip()
                    if sid_file.exists()
                    else ""
                )
            except OSError:
                sid = ""
            if sid and sid == session_id:
                return d
    return None


def claude_glm_steer_tool(
    *, session_id: str = "", log_dir: str = "", text: str = ""
) -> str:
    """Steer a RUNNING hot claude-glm session.

    Drops a JSON user-message envelope into ``<log_dir>/steer/``; the running
    dispatch forwards it to claude's stdin, consumed at the next tool-call
    break. Errors if no hot session matches — for a wound-down session, dispatch
    again with ``resume_session_id`` instead.
    """
    text = (text or "").strip()
    if not text:
        return tool_error("text is required.")
    target = _resolve_hot_log_dir(session_id=session_id, log_dir=log_dir)
    if not target:
        return tool_error(
            "No hot steerable claude-glm session found. Pass the session_id of a "
            "running dispatch (or its log_dir). If the session already wound down, "
            "dispatch again with resume_session_id instead of steering."
        )
    steer_dir = target / "steer"
    envelope = {
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }
    stamp = (
        time.strftime("%Y%m%d_%H%M%S")
        + f"_{os.getpid()}_{abs(hash(text)) % 100000:05d}"
    )
    out = steer_dir / f"{stamp}.json"
    out.write_text(
        json.dumps({"envelope": envelope}, ensure_ascii=False), encoding="utf-8"
    )
    return json.dumps(
        {"status": "steered", "log_dir": str(target), "file": str(out)},
        ensure_ascii=False,
    )


CLAUDE_GLM_STEER_SCHEMA = {
    "name": "claude_glm_steer",
    "description": (
        "Steer a RUNNING claude-glm hot session by appending a user message that "
        "claude consumes at its next tool-call break. Use this when a "
        "claude_glm_dispatch is still running and the user sends a follow-up, "
        "correction, or adjustment — do NOT start a parallel dispatch. For a "
        "session that already finished/wound down, dispatch again with "
        "resume_session_id instead."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The steering instruction to append to the running claude-glm session.",
            },
            "session_id": {
                "type": "string",
                "description": "Claude Code session id of the running dispatch (preferred resolver).",
            },
            "log_dir": {
                "type": "string",
                "description": "Absolute log_dir of the running dispatch (alternative resolver).",
            },
        },
        "required": ["text"],
    },
}


def _handle_claude_glm_steer(args: dict[str, Any], **_: Any) -> str:
    return claude_glm_steer_tool(
        session_id=str(args.get("session_id") or ""),
        log_dir=str(args.get("log_dir") or ""),
        text=str(args.get("text") or ""),
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run claude-glm with structured Hermes dispatch capture."
    )
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--max-turns", default="auto")
    parser.add_argument(
        "--effort", default="max", choices=["low", "medium", "high", "max", "auto"]
    )
    parser.add_argument("--resume", default="")
    parser.add_argument("--continue", dest="continue_latest", action="store_true")
    parser.add_argument("--claude-bin", default=DEFAULT_CLAUDE_GLM_BIN)
    parser.add_argument("--workdir", default="")
    parser.add_argument(
        "--idle-timeout", type=int, default=300, dest="idle_timeout",
        help="Seconds of post-turn idle before the hot session winds down (default 300).",
    )
    ns = parser.parse_args(argv)
    return run_claude_glm_dispatch(
        prompt_file=Path(ns.prompt_file),
        log_dir=Path(ns.log_dir),
        max_turns=ns.max_turns,
        effort=ns.effort,
        resume_session_id=ns.resume,
        continue_latest=ns.continue_latest,
        claude_bin=ns.claude_bin,
        workdir=ns.workdir or None,
        idle_timeout=ns.idle_timeout,
    )


registry.register(
    name="claude_glm_dispatch",
    toolset="terminal",
    schema=CLAUDE_GLM_DISPATCH_SCHEMA,
    handler=_handle_claude_glm_dispatch,
    emoji="GLM",
)


registry.register(
    name="claude_glm_steer",
    toolset="terminal",
    schema=CLAUDE_GLM_STEER_SCHEMA,
    handler=_handle_claude_glm_steer,
    emoji="GLM",
)


if __name__ == "__main__":
    raise SystemExit(_main())
