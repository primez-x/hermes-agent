"""ZAI / GLM provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


class ZaiProviderProfile(ProviderProfile):
    """Z.AI request shaping.

    GLM-4.5+ supports the `thinking` object. The `reasoning_effort` dial is
    only accepted by GLM-5.2+, so keep GLM-4.7 on explicit thinking without
    sending the unsupported effort field.
    """

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context,
    ):
        model_l = str(model or "").strip().lower()
        extra_body = {}
        top_level = {}

        enabled = not (
            isinstance(reasoning_config, dict)
            and reasoning_config.get("enabled") is False
        )
        if model_l.startswith("glm-"):
            extra_body["thinking"] = {
                "type": "enabled" if enabled else "disabled",
            }

        if enabled and model_l.startswith("glm-5.2"):
            effort = "max"
            if isinstance(reasoning_config, dict):
                requested = str(reasoning_config.get("effort") or "").strip().lower()
                if requested in {"max", "xhigh"}:
                    effort = "max"
                elif requested in {"high", "medium", "low", "minimal", "none"}:
                    effort = requested
            top_level["reasoning_effort"] = effort

        return extra_body, top_level


zai = ZaiProviderProfile(
    name="zai",
    aliases=("glm", "z-ai", "z.ai", "zhipu"),
    env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
    display_name="Z.AI (GLM)",
    description="Z.AI / GLM — Zhipu AI models",
    signup_url="https://z.ai/",
    fallback_models=(
        "glm-5.2",
        "glm-5",
        "glm-4-9b",
    ),
    base_url="https://api.z.ai/api/paas/v4",
    default_aux_model="glm-4.5-flash",
)

register_provider(zai)
