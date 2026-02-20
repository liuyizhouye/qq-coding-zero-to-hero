"""openai_api lesson package."""

from .component import (
    DEFAULT_PROMPT,
    ONLINE_ENDPOINTS,
    DemoResult,
    EndpointCallResult,
    build_capability_overview,
    build_parameter_reference,
    run_demo,
)


def build_debug_state(user_text: str = DEFAULT_PROMPT) -> dict[str, object]:
    from .main import build_debug_state as _build_debug_state

    return _build_debug_state(user_text=user_text)


def run_debug(user_text: str = DEFAULT_PROMPT) -> dict[str, object]:
    from .main import run_debug as _run_debug

    return _run_debug(user_text=user_text)

__all__ = [
    "DEFAULT_PROMPT",
    "ONLINE_ENDPOINTS",
    "DemoResult",
    "EndpointCallResult",
    "build_capability_overview",
    "build_parameter_reference",
    "build_debug_state",
    "run_debug",
    "run_demo",
]
