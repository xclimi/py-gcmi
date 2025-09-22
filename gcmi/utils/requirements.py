"""
Declarative requirements for GCMI components.

Provides:
- Requirement: a declarative spec for required keys/values in state/params/forcing
- requires: decorator to attach Requirement specs to a StepFn or any callable
- get_requirements: retrieve attached requirements (follows __wrapped__ chain)
- validate_requirements: runtime validator producing structured violations
- RequirementError: aggregated error for failing requirements

Intended usage:
- Authors declare what a step function or middleware needs (e.g. params.spectral.radius).
- A check middleware (see gcmi.middleware.requirements) enforces these for the
  first few steps (or always), then skips for performance if desired.

Notes:
- This module is zero-dependency and does not mutate inputs.
- Validation does not modify State/Params/Forcing; it only raises or reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (Any, Callable, List, Literal, Mapping, Optional, Sequence,
                    Tuple)

Where = Literal["state", "params", "forcing"]
Severity = Literal["error", "warn"]


@dataclass(frozen=True)
class Requirement:
    """
    A declarative requirement on State/Params/Forcing.

    Attributes:
        where: which container to check ("state" | "params" | "forcing")
        path: dotted path within that container (e.g., "spectral.radius")
        required: if True, missing path is a violation
        type: optional isinstance check (type or tuple of types)
        predicate: optional predicate(value) -> bool; False is a violation
        message: optional custom message to display on violation
        severity: "error" (default) or "warn"
    """

    where: Where
    path: str
    required: bool = True
    type: Optional[type | tuple[type, ...]] = None
    predicate: Optional[Callable[[Any], bool]] = None
    message: Optional[str] = None
    severity: Severity = "error"


@dataclass(frozen=True)
class Violation:
    where: Where
    path: str
    severity: Severity
    message: str
    requirement: Requirement


class RequirementError(Exception):
    """Aggregated requirements error."""

    def __init__(self, violations: Sequence[Violation]) -> None:
        self.violations = list(violations)
        msgs = [
            f"[{v.severity.upper()}] {v.where}.{v.path}: {v.message}"
            for v in self.violations
        ]
        super().__init__("Requirements not satisfied:\n" + "\n".join(msgs))


def _get_path(d: Mapping[str, Any], path: str) -> Any:
    cur: Any = d
    for seg in path.split("."):
        if not isinstance(cur, Mapping):
            raise KeyError(
                f"Path '{path}' invalid: segment '{seg}' encountered non-mapping type {type(cur).__name__}"
            )
        try:
            cur = cur[seg]  # type: ignore[index]
        except KeyError as e:
            raise KeyError(f"Missing key '{seg}' while resolving '{path}'") from e
    return cur


def requires(*reqs: Requirement):
    """
    Decorator to attach requirement specs to a function.

    Example:
        @requires(
            Requirement("params", "spectral.radius", type=(int, float), predicate=lambda r: r > 0)
        )
        def step_fn(...): ...

    The metadata is stored on attribute '__gcmi_requires__' (tuple[Requirement, ...]).
    """

    def _decorate(fn):
        existing: Tuple[Requirement, ...] = tuple(getattr(fn, "__gcmi_requires__", ()))
        setattr(fn, "__gcmi_requires__", existing + tuple(reqs))
        return fn

    return _decorate


def get_requirements(fn: Any) -> Tuple[Requirement, ...]:
    """
    Retrieve requirement specs from a function, following __wrapped__ chains if present.

    Returns:
        A tuple of Requirement instances (may be empty).
    """
    reqs: List[Requirement] = []
    seen: set[int] = set()
    cur = fn
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        attached = getattr(cur, "__gcmi_requires__", ())
        if attached:
            reqs.extend(attached)
        cur = getattr(cur, "__wrapped__", None)
    return tuple(reqs)


def validate_requirements(
    *,
    state: Mapping[str, Any] | None,
    params: Mapping[str, Any] | None,
    forcing: Mapping[str, Any] | None,
    requirements: Sequence[Requirement],
) -> Tuple[list[Violation], list[Violation]]:
    """
    Validate requirements against provided containers.

    Returns:
        (errors, warnings) as lists of Violation.
    """
    errors: list[Violation] = []
    warns: list[Violation] = []

    def get_container(where: Where) -> Mapping[str, Any] | None:
        if where == "state":
            return state
        if where == "params":
            return params
        if where == "forcing":
            return forcing
        return None

    for r in requirements:
        container = get_container(r.where)
        if container is None:
            v = Violation(
                where=r.where,
                path=r.path,
                severity=r.severity,
                message=f"Container '{r.where}' is None; cannot check path '{r.path}'",
                requirement=r,
            )
            (errors if r.severity == "error" else warns).append(v)
            continue

        try:
            value = _get_path(container, r.path)
        except KeyError as e:
            if r.required:
                v = Violation(
                    where=r.where,
                    path=r.path,
                    severity=r.severity,
                    message=r.message or str(e),
                    requirement=r,
                )
                (errors if r.severity == "error" else warns).append(v)
            # If not required, missing is acceptable; continue.
            continue

        if r.type is not None and not isinstance(value, r.type):
            v = Violation(
                where=r.where,
                path=r.path,
                severity=r.severity,
                message=r.message
                or f"Expected type {r.type}, got {type(value).__name__}",
                requirement=r,
            )
            (errors if r.severity == "error" else warns).append(v)
            continue

        if r.predicate is not None:
            ok = False
            try:
                ok = bool(r.predicate(value))
            except Exception as e:
                v = Violation(
                    where=r.where,
                    path=r.path,
                    severity=r.severity,
                    message=r.message or f"Predicate raised: {e}",
                    requirement=r,
                )
                (errors if r.severity == "error" else warns).append(v)
                continue
            if not ok:
                v = Violation(
                    where=r.where,
                    path=r.path,
                    severity=r.severity,
                    message=r.message or "Predicate returned False",
                    requirement=r,
                )
                (errors if r.severity == "error" else warns).append(v)

    return errors, warns
