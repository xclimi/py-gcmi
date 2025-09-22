from __future__ import annotations

import math
from typing import (Any, Callable, Dict, Mapping, Sequence, Tuple, TypedDict,
                    cast)

# Loose typing aliases to avoid import cycles with gcmi.core.api
State = Dict[str, Any]
Forcing = Dict[str, Any]
Params = Mapping[str, Any]
Diag = Dict[str, Any]
StepFn = Callable[[State, Forcing, Params, float], Tuple[State, Diag]]


class XPProto(TypedDict, total=False):
    # Minimal structural typing hint for xp-like namespaces if needed by linters
    pass


def _append_mw_meta(diag: Diag, name: str, **meta: Any) -> None:
    (diag.setdefault("gcmi_mw", [])).append({"name": name, **meta})


def with_cfl_guard(
    step: StepFn,
    *,
    cfl_max: float = 0.8,
    wave_speed_cb: Callable[[State, Params, Any], float],
) -> StepFn:
    """
    Stability middleware: enforce dt against a CFL criterion via optional substepping.

    Behavior:
    - Compute vmax = wave_speed_cb(state, params, xp)
    - dx := params['grid']['dx_min'] if present else 1.0
    - cfl := vmax * dt / dx
    - If cfl <= cfl_max: single inner step
    - Else: perform n_sub = ceil(cfl / cfl_max) sub-steps with dt_sub = dt / n_sub

    Notes:
    - This is a generic controller; it does not alter physics except time slicing.
    - Metadata is recorded under diag["gcmi_mw"].
    """

    def wrapped(state: State, forcing: Forcing, params: Params, dt: float, *, xp):
        vmax = float(wave_speed_cb(state, params, xp))
        dx = 1.0
        try:
            grid = cast(Mapping[str, Any], params.get("grid", {}))
            if "dx_min" in grid:
                dx = float(grid["dx_min"])
        except Exception:
            dx = 1.0

        cfl = 0.0 if dx == 0 else vmax * dt / dx
        if cfl <= cfl_max or dx == 0.0 or vmax == 0.0 or dt == 0.0:
            st, dg = step(state, forcing, params, dt, xp=xp)
            _append_mw_meta(
                dg, "cfl_guard", cfl=cfl, n_substeps=1, vmax=vmax, dx=dx, dt=dt
            )
            return st, dg

        n_sub = max(1, int(math.ceil(cfl / cfl_max)))
        dt_sub = dt / n_sub if n_sub > 0 else dt
        st = state
        last_diag: Diag = {}
        for _ in range(n_sub):
            st, last_diag = step(st, forcing, params, dt_sub, xp=xp)
        _append_mw_meta(
            last_diag,
            "cfl_guard",
            cfl=cfl,
            n_substeps=n_sub,
            vmax=vmax,
            dx=dx,
            dt=dt,
            dt_sub=dt_sub,
        )
        return st, last_diag

    setattr(wrapped, "__wrapped__", step)
    return wrapped  # type: ignore[return-value]


def with_hyperdiff(
    step: StepFn,
    *,
    coeff: float = 0.0,
    order: int = 4,
    vars: Sequence[str] = ("T", "u", "v"),
) -> StepFn:
    """
    Add hyperdiffusion (placeholder via ops.grid.laplacian). For M1, laplacian may be a no-op.

    Args:
        coeff: diffusion coefficient (applied as: var <- var - coeff * Laplacian(var))
        order: nominal order (recorded in metadata only for M1)
        vars: variables to diffuse if present in state
    """
    from gcmi.ops import grid as grid_ops  # local import to avoid cycles

    def wrapped(state: State, forcing: Forcing, params: Params, dt: float, *, xp):
        st, dg = step(state, forcing, params, dt, xp=xp)
        if coeff != 0.0:
            for v in vars:
                if v in st:
                    try:
                        lap_v = grid_ops.laplacian(st[v], xp=xp)
                        st[v] = st[v] - coeff * lap_v  # type: ignore[operator]
                    except Exception:
                        # On type incompatibility, skip modification but continue
                        pass
        _append_mw_meta(dg, "hyperdiff", coeff=coeff, order=order, vars=tuple(vars))
        return st, dg

    setattr(wrapped, "__wrapped__", step)
    return wrapped  # type: ignore[return-value]


def with_flux_limiter(
    step: StepFn,
    *,
    scheme: str = "mc",
    vars: Sequence[str] = ("q", "T"),
) -> StepFn:
    """
    Flux limiter (placeholder). For M1, records metadata; no state change.
    """

    def wrapped(state: State, forcing: Forcing, params: Params, dt: float, *, xp):
        st, dg = step(state, forcing, params, dt, xp=xp)
        _append_mw_meta(dg, "flux_limiter", scheme=scheme, vars=tuple(vars))
        return st, dg

    setattr(wrapped, "__wrapped__", step)
    return wrapped  # type: ignore[return-value]


def with_positivity(
    step: StepFn,
    *,
    vars: Sequence[str] = ("q",),
    lower: float = 0.0,
    conserve: str | None = None,  # placeholder: not enforced in M1
) -> StepFn:
    """
    Enforce non-negativity via clamping for selected variables.
    """
    from gcmi.ops import grid as grid_ops  # local import

    def wrapped(state: State, forcing: Forcing, params: Params, dt: float, *, xp):
        st, dg = step(state, forcing, params, dt, xp=xp)
        for v in vars:
            if v in st:
                try:
                    st[v] = grid_ops.clamp_min(st[v], lower, xp=xp)
                except Exception:
                    # If clamp fails, skip
                    pass
        _append_mw_meta(
            dg, "positivity", vars=tuple(vars), lower=lower, conserve=conserve
        )
        return st, dg

    setattr(wrapped, "__wrapped__", step)
    return wrapped  # type: ignore[return-value]


def with_energy_fix(
    step: StepFn,
    *,
    budget: Sequence[str] = ("dry_static", "latent", "kinetic"),
) -> StepFn:
    """
    Global energy budget correction (placeholder). For M1, records metadata only.
    """

    def wrapped(state: State, forcing: Forcing, params: Params, dt: float, *, xp):
        st, dg = step(state, forcing, params, dt, xp=xp)
        _append_mw_meta(dg, "energy_fix", budget=tuple(budget))
        return st, dg

    setattr(wrapped, "__wrapped__", step)
    return wrapped  # type: ignore[return-value]


def with_conservation_projection(
    step: StepFn,
    *,
    conserve: Sequence[str] = ("total_mass", "moist_energy"),
) -> StepFn:
    """
    Projection onto conserved totals (placeholder). For M1, records metadata only.
    """

    def wrapped(state: State, forcing: Forcing, params: Params, dt: float, *, xp):
        st, dg = step(state, forcing, params, dt, xp=xp)
        _append_mw_meta(dg, "conservation_projection", conserve=tuple(conserve))
        return st, dg

    setattr(wrapped, "__wrapped__", step)
    return wrapped  # type: ignore[return-value]
