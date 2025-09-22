from __future__ import annotations

from typing import Any, Mapping, Protocol


class XP(Protocol):
    def maximum(self, x: Any, y: Any): ...
    def sum(self, x: Any): ...
    def zeros_like(self, x: Any): ...
    def asarray(self, x: Any): ...


def identity(x: Any, *, xp: XP) -> Any:
    """
    Minimal placeholder operator that returns input unchanged.
    Useful as a no-op default when wiring ops.
    """
    return x


def laplacian(
    field: Any, *, xp: XP, dx: float | None = None, dy: float | None = None
) -> Any:
    """
    Placeholder Laplacian operator.

    For M1, we provide a safe no-op that returns zeros_like(field) if available,
    otherwise returns the input unchanged. This establishes the ops facade and
    call sites; numerical implementation can be filled in subsequent milestones.
    """
    try:
        return xp.zeros_like(field)
    except Exception:
        # Fallback to identity if zeros_like is not available
        return field


def clamp_min(field: Any, lower: float, *, xp: XP) -> Any:
    """
    Clamp to a lower bound in a backend-neutral way.
    Falls back to Python max for scalar values if xp.maximum is unavailable.
    """
    try:
        return xp.maximum(field, lower)  # type: ignore[arg-type]
    except Exception:
        try:
            return max(field, lower)  # type: ignore[type-var]
        except Exception:
            return field


def total(field: Any, *, xp: XP) -> float:
    """
    Compute a total (sum) in a backend-neutral way.
    Returns float where possible; otherwise attempts to cast.
    """
    try:
        s = xp.sum(field)
        try:
            return float(s)  # type: ignore[arg-type]
        except Exception:
            return s  # type: ignore[return-value]
    except Exception:
        # Fallback: if scalar-like
        try:
            return float(field)  # type: ignore[arg-type]
        except Exception:
            return 0.0


def dx_min_from_params(params: Mapping[str, Any]) -> float | None:
    """
    Helper to obtain grid.dx_min from params if present.
    """
    try:
        grid = params.get("grid", {})  # type: ignore[assignment]
        return float(grid.get("dx_min"))  # type: ignore[arg-type]
    except Exception:
        return None
