import json


def test_stream_json_parser_extracts_session_tools_and_result():
    from tools.claude_glm_dispatch import parse_stream_json_lines

    summary = parse_stream_json_lines(
        [
            json.dumps({"type": "system", "session_id": "sess-123"}),
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Bash",
                                "input": {"command": "pytest tests/tools"},
                            }
                        ]
                    },
                }
            ),
            json.dumps(
                {
                    "type": "stream_event",
                    "event": {
                        "type": "content_block_start",
                        "content_block": {
                            "type": "tool_use",
                            "name": "Edit",
                            "input": {"file_path": "app.py"},
                        },
                    },
                }
            ),
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "session_id": "sess-123",
                    "result": "Implemented the change and ran tests.",
                    "num_turns": 4,
                    "total_cost_usd": 0.0123,
                    "duration_ms": 23456,
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                }
            ),
        ]
    )

    assert summary["session_id"] == "sess-123"
    assert summary["status"] == "success"
    assert summary["num_turns"] == 4
    assert summary["total_cost_usd"] == 0.0123
    assert summary["tools"] == [
        {"name": "Bash", "detail": "pytest tests/tools"},
        {"name": "Edit", "detail": "app.py"},
    ]
    assert summary["result"] == "Implemented the change and ran tests."


def test_build_claude_glm_command_uses_prompt_file_and_stream_json(tmp_path):
    from tools.claude_glm_dispatch import build_claude_glm_command

    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("secret prompt body", encoding="utf-8")
    log_dir = tmp_path / "logs"

    command = build_claude_glm_command(
        prompt_file=prompt_file,
        log_dir=log_dir,
        max_turns=12,
        effort="max",
        resume_session_id="sess-123",
    )

    assert "--prompt-file" in command
    assert str(prompt_file) in command
    assert "--output-format" not in command
    assert "secret prompt body" not in command
    assert "--max-turns 12" in command
    assert "--effort max" in command
    assert "--resume" in command
    assert str(log_dir) in command


def test_resolve_max_turns_clamps_explicit_value():
    from tools.claude_glm_dispatch import resolve_max_turns

    resolved, source = resolve_max_turns(
        500,
        config={
            "max_turns": "auto",
            "hard_max_turns": 200,
        },
    )

    assert resolved == 200
    assert source == "explicit:clamped"


def test_resolve_max_turns_uses_complex_auto_tier_for_titan_work():
    from tools.claude_glm_dispatch import resolve_max_turns

    resolved, source = resolve_max_turns(
        "auto",
        prompt="Fix failing pytest coverage for the migration.",
        workdir="/home/user/titan",
        config={
            "max_turns": "auto",
            "complex_max_turns": 150,
            "hard_max_turns": 200,
        },
    )

    assert resolved == 150
    assert source == "auto:complex"


def test_resolve_max_turns_defaults_to_normal_auto_tier():
    from tools.claude_glm_dispatch import resolve_max_turns

    resolved, source = resolve_max_turns(
        None,
        prompt="Implement the requested change.",
        workdir="/home/user/example",
        config={
            "max_turns": "auto",
            "normal_max_turns": 100,
            "hard_max_turns": 200,
        },
    )

    assert resolved == 100
    assert source == "auto:normal"


def test_dispatch_tool_starts_tracked_background_process(tmp_path, monkeypatch):
    from tools import claude_glm_dispatch

    calls = []

    def fake_terminal_tool(**kwargs):
        calls.append(kwargs)
        return json.dumps(
            {
                "output": "Background process started",
                "session_id": "proc_abc",
                "pid": 1234,
                "exit_code": 0,
                "error": None,
            }
        )

    monkeypatch.setattr(claude_glm_dispatch, "DISPATCH_ROOT", tmp_path)
    monkeypatch.setattr(claude_glm_dispatch, "terminal_tool", fake_terminal_tool)
    monkeypatch.setattr(
        claude_glm_dispatch,
        "_load_dispatch_config",
        lambda: {
            "max_turns": "auto",
            "normal_max_turns": 100,
            "hard_max_turns": 200,
        },
    )

    result = json.loads(
        claude_glm_dispatch.claude_glm_dispatch_tool(
            prompt="Implement the requested change.",
            workdir=str(tmp_path),
        )
    )

    assert result["status"] == "dispatched"
    assert result["process_session_id"] == "proc_abc"
    assert result["log_dir"].startswith(str(tmp_path))
    assert result["prompt_file"].startswith(str(tmp_path))
    assert calls
    assert calls[0]["background"] is True
    assert calls[0]["notify_on_complete"] is True
    assert calls[0]["workdir"] == str(tmp_path)
    assert "Implement the requested change." not in calls[0]["command"]
    assert "claude-glm-dispatch" in calls[0]["command"]
    assert "--max-turns 100" in calls[0]["command"]
    assert result["resolved_max_turns"] == 100
    assert result["max_turns_source"] == "auto:normal"
