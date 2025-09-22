"""
Lightweight, Pythonic helpers for ergonomic Mapping (dict-like) access.

These utilities keep the public API mapping-based (State/Params/Forcing) while
providing concise, mypy-friendly, and fast access patterns that resemble
destructuring without introducing dotdicts or heavy dependencies.

Guidelines:
- Use take/require in hot paths for brevity and speed.
- Use take_nested for config/params access where dotted paths improve clarity.
- Avoid silent defaults unless the cfg schema supplies them; prefer explicit KeyError.
"""

from __future__ import annotations

from operator import itemgetter
from typing import Any, Mapping, Tuple

__all__ = [
    "take",
    "require",
    "take_nested",
    "split_keys",
]


def take(d: Mapping[str, Any], *keys: str) -> Tuple[Any, ...]:
    """
    Destructure required keys from a mapping in one expression.

    Always returns a tuple of values (length equals the number of keys),
    so single-key access yields a 1-tuple.

    Example:
        T, q, u, v = take(state, "T", "q", "u", "v")
        (dx_min,) = take(params["grid"], "dx_min")

    Raises:
        KeyError: if any key is missing.
    """
    if not keys:
        return ()
    if len(keys) == 1:
        k = keys[0]
        return (d[k],)
    return itemgetter(*keys)(d)


def require(d: Mapping[str, Any], *keys: str) -> Tuple[Any, ...]:
    """
    Like take(), but raises a clearer KeyError message indicating what was expected.

    Example:
        (dx_min,) = require(params["grid"], "dx_min")

    Raises:
        KeyError: with an error message listing the missing key and available keys.
    """
    if not keys:
        return ()
    try:
        if len(keys) == 1:
            k = keys[0]
            return (d[k],)
        return itemgetter(*keys)(d)
    except (
        KeyError
    ) as e:  # pragma: no cover - error branch is covered by tests via message check
        missing = e.args[0]
        raise KeyError(
            f"Missing required key '{missing}'. Expected one of: {keys}. "
            f"Available keys: {list(d.keys())}"
        ) from e


def _get_path(d: Mapping[str, Any], path: str) -> Any:
    cur: Any = d
    for seg in path.split("."):
        try:
            cur = cur[seg]  # type: ignore[index]
        except KeyError as e:
            raise KeyError(
                f"Path segment '{seg}' not found while resolving '{path}'."
            ) from e
        except TypeError as e:
            raise TypeError(
                f"Encountered non-mapping object at segment '{seg}' while resolving '{path}'. "
                f"Current object type: {type(cur).__name__}"
            ) from e
    return cur


def take_nested(d: Mapping[str, Any], *paths: str) -> Tuple[Any, ...]:
    """
    Extract multiple dotted-path values from a nested mapping.

    Example:
        dx_min, theta = take_nested(params, "grid.dx_min", "spectral.semi_implicit.theta")

    Raises:
        KeyError: if any path segment is missing.
        TypeError: if an intermediate object is not a Mapping.
    """
    return tuple(_get_path(d, p) for p in paths)


def split_keys(
    d: Mapping[str, Any], *keys: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Split a mapping into (picked, rest) by a set of keys.

    Example:
        picked, rest = split_keys(params, "grid", "backend")

    Note:
        - Keys must exist in 'd' to be included in 'picked'; missing keys are ignored.
        - 'rest' preserves all other keys.
    """
    picked: dict[str, Any] = {k: d[k] for k in keys if k in d}
    rest: dict[str, Any] = {k: v for k, v in d.items() if k not in picked}
    return picked, rest
