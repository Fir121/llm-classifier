"""Prompt templating helpers."""

from __future__ import annotations

from typing import Any


def safe_prompt_format(template: str, **values: Any) -> str:
    """Replace known placeholders without parsing other braces.

    Unlike ``str.format``, this helper only replaces exact ``{name}`` tokens for
    provided values and leaves all other braces untouched.
    """
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace(f"{{{key}}}", str(value))
    return rendered
