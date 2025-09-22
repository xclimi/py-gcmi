"""Minimal driver utilities to make GCMI usage feel as simple as WSGI.

WSGI analogy:
- app(environ, start_response)                    ~ step(dt, state, forcing, params, *, xp)
- make_server(host, port, app).serve_forever()    ~ run(step, n_steps, xp, cfg=?, forcing_stream=?, hooks=?)

Design goals
- Keep Core strict and stable (for middleware/hooks/tests), while Drivers offer a
  "sugar" layer with sensible defaults and flexible step signatures for minimal demos.
- Defaults allow a true 20-line "Hello, GCMI" script.

Public (Driver) API
- run(
    step,
    *,
    n_steps: int,
    xp,
    cfg: Mapping[str, Any] | None = None,
    forcing_stream: ForcingStream | None = None,
    hooks: tuple[Hook, ...] = (),
    dt: float | None = None,
  ) -> tuple[State, Mapping[str, Any]]

- make_runner(step, *, xp, cfg=None, hooks=(), dt=None) -> (forcing_stream, n_steps) -> (state, report)

Step signature flexibility (Driver level)
- Preferred minimal:   step(dt, state, *, xp) -> (state, diag)
- Also accepted:       step(dt, state, forcing=None, *, xp) / step(dt, state, forcing=None, params=None, *, xp)
- Back-compatible:     step(state, forcing, params, dt, *, xp)
- The driver inspects available parameter names and passes only what the function accepts.
"""

from __future__ import annotations

import inspect
from time import perf_counter
from typing import (Any, Callable, Iterable, Iterator, Mapping, Protocol, Tuple, Union)

from gcmi.core.api import XP, Diag, Forcing, Params, State, init_fn

# Hook signature (observational only) aligned with core hooks
Hook = Callable[[int, dict[str, Any], dict[str, Any]], None]


class StepCallable(Protocol):
    """Callable contract for a GCMI step function (flexible at Driver level).

    Implementations may use any of the accepted signatures (see module docstring).
    The driver will adapt calls by name (dt/state/forcing/params/xp) and only pass
    those the function actually declares.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> Tuple[State, Diag]: ...


# Accepted forcing stream forms: iterable/iterator of Forcing or a callable(k)->Forcing
ForcingStream = Union[Iterable[Forcing], Iterator[Forcing], Callable[[int], Forcing]]


def _as_iter(forcing_stream: ForcingStream | None) -> Iterator[Forcing]:
    """Normalize a forcing stream into an iterator. If None, yield {} forever."""
    if forcing_stream is None:
        def gen() -> Iterator[Forcing]:
            while True:
                yield {}
        return gen()

    if callable(forcing_stream):
        k = 0

        def gen() -> Iterator[Forcing]:
            nonlocal k
            while True:
                yield forcing_stream(k)
                k += 1

        return gen()
    if hasattr(forcing_stream, "__iter__"):
        return iter(forcing_stream)  # type: ignore[arg-type]
    raise TypeError("forcing_stream must be Iterable[Forcing] or Callable[[int], Forcing]")


def _normalize_cfg(cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Provide a minimal cfg when None is given."""
    if cfg is None:
        return {"state0": {}, "params": {}}
    return cfg


def _bind_step(step: StepCallable) -> Callable[[State, Forcing, Params, float, XP], Tuple[State, Diag]]:
    """Create a stable calling adapter for a given step function.

    The adapter:
    - Resolves the step's accepted parameter names via inspection
    - Calls with keyword arguments for robustness: dt/state/forcing/params/xp
    - Passes only parameters that the function actually declares
    """
    sig = inspect.signature(step)
    accepted = set(sig.parameters.keys())

    def call(state: State, forcing: Forcing, params: Params, dt: float, xp: XP) -> Tuple[State, Diag]:
        kwargs: dict[str, Any] = {}
        # Pass only what is accepted by the step signature
        if "dt" in accepted:
            kwargs["dt"] = dt
        if "state" in accepted:
            kwargs["state"] = state
        if "forcing" in accepted:
            kwargs["forcing"] = forcing
        if "params" in accepted:
            kwargs["params"] = params
        if "xp" in accepted:
            kwargs["xp"] = xp

        # If the function uses the "full" core order (state, forcing, params, dt, *, xp)
        # this will still work because we pass by keywords.
        out = step(**kwargs)  # type: ignore[misc]
        if not isinstance(out, tuple) or len(out) != 2:
            raise TypeError("Step function must return a tuple (state, diag)")
        new_state, diag = out
        if diag is None:
            diag = {}
        elif not isinstance(diag, dict):
            # Normalize non-dict diags into a dict for hooks/reporting
            diag = {"diag": diag}
        return new_state, diag

    return call


def make_runner(
    step: StepCallable,
    *,
    cfg: Mapping[str, Any] | None = None,
    xp: XP,
    hooks: Tuple[Hook, ...] = (),
    dt: float | None = None,
) -> Callable[[ForcingStream | None, int], Tuple[State, Mapping[str, Any]]]:
    """Create a minimal runner bound to a step, config, backend, and hooks.

    Usage (WSGI-like):
        runner = make_runner(step, cfg=None, xp=numpy, hooks=(timer_hook(...),), dt=None)
        final_state, report = runner(None, 10)

    Args:
        step: A StepCallable implementing the (flexible) step contract.
        cfg: Optional config mapping; init_fn will produce (state0, params).
             If None, a minimal cfg is used: {"state0": {}, "params": {}}.
        xp: Backend array namespace (e.g., numpy, jax.numpy, torch-like).
        hooks: Observational hooks invoked per step; must not mutate state/params.
        dt: Optional explicit timestep; overrides cfg["params"]["time"]["dt"].

    Returns:
        A callable (forcing_stream, n_steps) -> (final_state, report)
    """
    cfg = _normalize_cfg(cfg)
    state0, params = init_fn(cfg, xp=xp)

    # Resolve dt with priority: explicit dt > params["time"]["dt"] > default 1.0
    params_time = params.get("time", {}) if isinstance(params.get("time", {}), Mapping) else {}
    dt_final = float(dt if dt is not None else params_time.get("dt", 1.0))  # type: ignore[arg-type]

    # Bind step once (no per-iteration signature overhead)
    call_step = _bind_step(step)

    def run(forcing_stream: ForcingStream | None, n_steps: int) -> Tuple[State, Mapping[str, Any]]:
        st: State = state0
        report: dict[str, Any] = {"timings": {"per_step_sec": []}, "last_diag": None}
        fiter = _as_iter(forcing_stream)

        for k in range(n_steps):
            forcing = next(fiter, {})
            t0 = perf_counter()
            st, diag = call_step(st, forcing, params, dt_final, xp)
            t1 = perf_counter()

            # Timing into diag + report
            dur = t1 - t0
            (diag.setdefault("timings", {}))["step_sec"] = dur
            report["timings"]["per_step_sec"].append(dur)

            # Invoke hooks (observational only). Support both minimal and richer signatures.
            for hook in hooks:
                try:
                    hook(k, st, diag)  # type: ignore[misc]
                except TypeError:
                    hook(k, st, diag, params=params, xp=xp)  # type: ignore[misc]

            report["last_diag"] = diag

        return st, report

    return run


def run(
    step: StepCallable,
    *,
    n_steps: int,
    xp: XP,
    cfg: Mapping[str, Any] | None = None,
    forcing_stream: ForcingStream | None = None,
    hooks: Tuple[Hook, ...] = (),
    dt: float | None = None,
) -> Tuple[State, Mapping[str, Any]]:
    """One-shot helper: initialize and execute a run with the provided step.

    Examples
        # Minimal (Hello, GCMI)
        final_state, report = run(step, n_steps=5, xp=np)

        # With explicit dt and a simple forcing stream
        final_state, report = run(step, n_steps=10, xp=np, dt=60.0, forcing_stream=lambda k: {})

        # With a non-empty cfg (state0/params) and hooks
        final_state, report = run(step, n_steps=20, xp=np, cfg={"state0": {...}, "params": {"time": {"dt": 300}}}, hooks=(timer_hook(...),))

    Note
        Core contracts remain strict; this Driver API is a convenience layer that:
        - supplies defaults (cfg/forcing_stream/forcing/params),
        - adapts flexible step signatures by parameter name,
        - preserves compatibility with middleware/hooks/tests built on the Core.
    """
    return make_runner(step, cfg=cfg, xp=xp, hooks=hooks, dt=dt)(forcing_stream, n_steps)
