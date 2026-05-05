from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Dict, Iterator, List, Optional


_SAMPLE_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar(
    "medrax_sample_context",
    default={},
)


def get_sample_context() -> Dict[str, Any]:
    context = _SAMPLE_CONTEXT.get()
    return dict(context) if isinstance(context, dict) else {}


@contextmanager
def bind_sample_context(
    *,
    image_paths: Optional[List[str]] = None,
    case_id: str = "",
    figures_dir: str = "",
) -> Iterator[None]:
    token: Token = _SAMPLE_CONTEXT.set(
        {
            "image_paths": list(image_paths or []),
            "case_id": str(case_id or ""),
            "figures_dir": str(figures_dir or ""),
        }
    )
    try:
        yield
    finally:
        _SAMPLE_CONTEXT.reset(token)
