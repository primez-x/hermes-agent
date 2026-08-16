from hermes_cli.model_switch import _apply_picker_allowlist


def test_picker_allowlist_filters_provider_and_model_rows():
    rows = [
        {"slug": "zai", "models": ["glm-5.3", "glm-5v-turbo", "glm-5"]},
        {"slug": "openai-codex", "models": ["gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.5"]},
        {
            "slug": "local-qwen",
            "is_user_defined": True,
            "models": ["qwen3.8-27b-uncensored", "other"],
        },
        {"slug": "moa", "models": ["moa-default"]},
    ]
    allowlist = {
        "ZAI": ["GLM-5.3", "GLM-5V-Turbo"],
        "openai-codex": ["gpt-5.6-sol", "gpt-5.6-luna"],
        "custom:local-qwen": ["qwen3.8-27b-uncensored"],
    }

    filtered = _apply_picker_allowlist(rows, allowlist)

    assert [(row["slug"], row["models"]) for row in filtered] == [
        ("zai", ["glm-5.3", "glm-5v-turbo"]),
        ("openai-codex", ["gpt-5.6-sol", "gpt-5.6-luna"]),
        ("local-qwen", ["qwen3.8-27b-uncensored"]),
    ]


def test_empty_picker_allowlist_fails_closed():
    rows = [{"slug": "zai", "models": ["glm-5.3"]}]
    assert _apply_picker_allowlist(rows, {}) == []
