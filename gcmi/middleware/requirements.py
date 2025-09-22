"""
Requirements-checking middleware.

Wrap a StepFn to enforce declarative requirements (attached via
gcmi.utils.requirements.requires or provided explicitly) during the first
N invocations, then skip checks for performance.

This follows the design goal:
- Make component needs explicit and declarative.
- Fail fast (or warn) in early steps; avoid overhead later.

Example
-------
from gcmi.utils.requirements import Requirement, requires
from gcmi.middleware.requirements import with_requirements_check

@requires(Requirement("params", "spectral.radius", type=(int, float), predicate=lambda r: r > 0))
def step_fn(state, forcing, params, dt, *, xp):
    ...

step = with_requirements_check(step_fn, max_checks=3)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

from gcmi.utils.requirements import (Requirement, RequirementError,
                                     get_requirements, validate_requirements)

# Loose typing to avoid import-cycle with core types
StepFn = Callable[
    [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], float],
    Tuple[Mapping[str, Any], Dict[str, Any]],
]


def with_requirements_check(
    step: StepFn,
    *,
    extra: Sequence[Requirement] = (),
    max_checks: int = 3,
    raise_on_error: bool = True,
    record_warnings: bool = True,
) -> StepFn:
    """
    Wrap a StepFn to validate attached (and extra) requirements for the first `max_checks` calls.

    Args:
        step: The inner step function.
        extra: Additional requirements to enforce (in addition to those attached via decorator).
        max_checks: Number of initial calls to validate; further calls skip validation.
        raise_on_error: If True, raise RequirementError on errors; otherwise, only record.
        record_warnings: If True, attach warnings/errors summary to diag["gcmi_requirements"].

    Returns:
        A StepFn with early-step requirements validation.

    Notes:
        - The count is based on wrapper invocations (may include sub-steps if upstream middleware performs substepping).
        - The wrapper sets __wrapped__ to allow requirement introspection by downstream tooling.
    """
    call_count = 0

    def wrapped(state, forcing, params, dt, *, xp):
        nonlocal call_count
        reqs = ()
        errors = []
        warns = []
        should_check = call_count < max_checks

        if should_check:
            reqs = get_requirements(step) + tuple(extra)
            if reqs:
                errors, warns = validate_requirements(
                    state=state, params=params, forcing=forcing, requirements=reqs
                )
                if errors and raise_on_error:
                    # Increment call_count to avoid repeated blocking if caller loops
                    call_count += 1
                    raise RequirementError(errors)

        st, diag = step(state, forcing, params, dt, xp=xp)

        if should_check and reqs and record_warnings:
            if errors or warns:
                (diag.setdefault("gcmi_requirements", [])).append(
                    {
                        "call": call_count,
                        "max_checks": max_checks,
                        "checked": len(reqs),
                        "errors": [
                            {"where": v.where, "path": v.path, "message": v.message}
                            for v in errors
                        ],
                        "warnings": [
                            {"where": v.where, "path": v.path, "message": v.message}
                            for v in warns
                        ],
                    }
                )

        call_count += 1
        return st, diag

    setattr(wrapped, "__wrapped__", step)
    return wrapped  # type: ignore[return-value]
