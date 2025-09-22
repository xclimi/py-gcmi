"""GCMI Core API (Python reference)

This module defines the strict, backend‑neutral contracts used by py-gcmi.

Key type aliases (Mappings of string keys to array-like values/metadata)
- State: Dict[str, ArrayLike]
    The prognostic variables that evolve in time and are advanced by the step
    function each iteration (e.g., "T", "q", "u", "v"). State holds the model's
    instantaneous fields. Step functions read State and must return the next
    State (a dict with the same conceptual structure). State values are
    backend-aware arrays (NumPy/JAX/Torch-like), but the container is a plain
    Python dict for ergonomic access, composition, and typing.

- Forcing: Dict[str, ArrayLike]
    External, time‑varying inputs that are not advanced by the model (e.g.,
    prescribed heating/cooling, fixed topography masks when streamed, idealized
    source/sink terms). Forcing is supplied per step by a forcing_stream and
    read by the step function; a step may ignore it.

- Params: Mapping[str, Any]
    Runtime constants/configuration (e.g., grid geometry, physical constants,
    time configuration like {"time":{"dt":...}}, backend capability records,
    ops adapters). Params are conceptually read‑only during a run (treat them
    as constants for reproducibility and clarity). init_fn is responsible for
    constructing Params from a configuration mapping and for recording the
    selected backend namespace under params["backend"]["xp"].

- Diag: Dict[str, Any]
    Per‑step diagnostics and metadata emitted by the step function and/or
    wrappers. Hooks and drivers may consume these entries for logging and
    observability (e.g., timings, budget summaries, middleware metadata).
    By convention, middleware may append metadata under diag["gcmi_mw"] (list)
    for auditability, but this is not strictly required in minimal demos.

Assembly and execution flow
1) Configuration → Initialization:
   A high‑level cfg (environment/config mapping) is passed to init_fn(cfg, xp=...)
   which returns (state0, params). The function injects the backend namespace
   into params["backend"]["xp"] so downstream code can discover the numerical
   backend (NumPy/JAX/Torch) without global state.

2) Stepping contract (StepFn):
   step(state, forcing, params, dt, *, xp) -> (state, diag)
   - Pure and stateless (no hidden globals); must return a new State and a Diag.
   - Middleware composes as (StepFn)->StepFn and may add Diag metadata.

3) Run loop:
   run_fn(init_state, params, forcing_stream, xp=..., n_steps=..., hooks=...)
   - Iterates n_steps; each iteration pulls a Forcing from forcing_stream.
   - Calls step(...) and measures per‑step wall time, attaching
     diag["timings"]["step_sec"].
   - Invokes hooks as hook(k, state, diag, params=?, xp=?). Hooks are strictly
     observational and must not mutate State/Params.

4) Timestep (dt):
   The core run loop reads dt from params["time"]["dt"] if present; otherwise
   defaults to 1.0. Drivers may offer convenience overrides, but Core keeps this
   simple rule for clarity and determinism.

Why dict-based containers?
- Dicts make required keys explicit, enable composable middleware and hooks, and
  keep the public API backend‑neutral. TypedDict/NamedTuple views can be layered
  for developer ergonomics, while runtime objects remain plain dicts for speed
  and simplicity.
"""
from __future__ import annotations

from time import perf_counter
from typing import (Any, Callable, Dict, Iterable, Iterator, Mapping, Protocol,
                    Tuple, Union, cast)


# Protocols for backend-neutral array namespace (numpy/jax.numpy/torch-like)
class ArrayLike(Protocol): ...


class XP(Protocol): ...  # numpy, jax.numpy, torch-like


State = Dict[str, ArrayLike]
Forcing = Dict[str, ArrayLike]
Params = Mapping[str, Any]
Diag = Dict[str, Any]

StepFn = Callable[[State, Forcing, Params, float], Tuple[State, Diag]]


# Module-level step function (can be wrapped by middleware from callers)
# Default "identity physics": pass-through with empty diag.
def step_fn(
    state: State, forcing: Forcing, params: Params, dt: float, *, xp: XP
) -> Tuple[State, Diag]:
    """
    Core step function: advance the model State by one time step.

    Semantics
    - Reads the current State and optional Forcing along with immutable Params
      and a scalar timestep dt; returns the next State and a Diag mapping.
    - Pure function contract: no hidden global state, reproducible from inputs.
    - Backend-neutral: xp is a keyword-only namespace (numpy/jax.numpy/torch-like).

    This default implementation is an identity mapping suited for wiring tests
    and minimal examples. Real models SHOULD replace or wrap this via middleware
    assembly to implement physics/dynamics tendencies.

    Inputs
    - state: Dict[str, ArrayLike]   (prognostic fields to be advanced)
    - forcing: Dict[str, ArrayLike] (external inputs for this step; may be empty)
    - params: Mapping[str, Any]     (runtime constants/config; read-only)
    - dt: float                     (timestep length in seconds or model units)
    - xp: XP                        (array namespace for backend operations)

    Outputs
    - (next_state, diag)
      next_state: Dict[str, ArrayLike] (may alias inputs if updates are in-place
      within backend semantics, but must honor the functional contract)
      diag: Dict[str, Any]             (step-level diagnostics/metadata; middleware
      often appends under diag["gcmi_mw"], and the run loop attaches timings)

    Notes
    - Hooks must treat diag as read-only; only middleware/step compose diag content.
    """
    # Minimal diag structure for downstream hooks/middleware
    diag: Diag = {"gcmi_mw": []}
    return state, diag


def init_fn(cfg: Mapping[str, Any], *, xp: XP) -> Tuple[State, Params]:
    """
    Initialize (state0, params) from a configuration mapping.

    Purpose
    - Normalize a high-level configuration (cfg) into the two core runtime
      containers used by the step/run loop: State and Params.
    - Record the numerical backend namespace under params["backend"]["xp"].

    Expected cfg keys (soft contract; schema validation is external to Core)
    - "state0": optional mapping for initial state arrays, e.g. {"T": xp.array(...)}
    - "params": optional mapping for runtime constants/config; this function
      injects the backend as params["backend"]["xp"] = xp.

    Returns
    - (state0, params)
      state0: Dict[str, ArrayLike]   (prognostic arrays at initial time)
      params: Mapping[str, Any]      (constants/config; includes backend xp)

    Notes
    - This function performs minimal shaping and does not validate schemas; a
      stricter loader/validator (e.g., pydantic) can wrap cfg ahead of this call.
    """
    state0_raw = cast(Mapping[str, Any], cfg.get("state0", {}))
    params_raw = dict(cast(Mapping[str, Any], cfg.get("params", {})))

    # Ensure a backend namespace record exists
    backend = dict(cast(Mapping[str, Any], params_raw.get("backend", {})))
    backend["xp"] = xp
    params_raw["backend"] = backend

    # Shallow copy of state; callers may convert to device arrays in their own init
    state0: State = dict(state0_raw)  # type: ignore[assignment]
    params: Params = params_raw
    return state0, params


def _as_iter(
    forcing_stream: Union[
        Iterable[Forcing], Iterator[Forcing], Callable[[int], Forcing]
    ],
) -> Iterator[Forcing]:
    if callable(forcing_stream):
        k = 0

        def gen() -> Iterator[Forcing]:
            nonlocal k
            while True:
                yield cast(Forcing, forcing_stream(k))
                k += 1

        return gen()
    if hasattr(forcing_stream, "__iter__"):
        return iter(cast(Iterable[Forcing], forcing_stream))
    raise TypeError(
        "forcing_stream must be an Iterable[Forcing] or Callable[[int], Forcing]"
    )


def run_fn(
    init: State,
    params: Params,
    forcing_stream: Union[
        Iterable[Forcing], Iterator[Forcing], Callable[[int], Forcing]
    ],
    *,
    xp: XP,
    n_steps: int,
    hooks: Tuple[Callable[[int, State, Diag], None], ...] = (),
) -> Tuple[State, Mapping[str, Any]]:
    """
    Execute the core run loop for n_steps and invoke observational hooks.

    Behavior
    - Iteratively:
      1) obtain a Forcing from forcing_stream (iterable/iterator/callable(k)),
      2) call step_fn(state, forcing, params, dt, xp=xp),
      3) measure per-step wall time and attach diag["timings"]["step_sec"],
      4) call each hook(k, state, diag, ...) strictly as an observer.

    Inputs
    - init:   initial State (from init_fn or user-constructed)
    - params: runtime constants/config (includes params["backend"]["xp"] = xp)
    - forcing_stream: produces Forcing per step ({} is acceptable if unused)
    - xp: backend namespace (NumPy/JAX/Torch-like)
    - n_steps: number of time steps to execute
    - hooks: tuple of callables observing (k, state, diag). Hooks must not mutate.

    Timestep (dt)
    - Resolved from params["time"]["dt"] if present; otherwise defaults to 1.0.

    Returns
    - (final_state, report)
      final_state: State after n_steps
      report: Mapping with aggregates (e.g., {"timings":{"per_step_sec":[...]},"last_diag":...})

    Notes
    - The run loop uses the module-level step_fn. To use a custom assembled step
      (e.g., with middleware), either rebind gcmi.core.api.step_fn or call your
      own assembled step directly from a custom loop/driver.
    """
    # Ensure step_fn is callable with the expected signature
    step = cast(Callable[[State, Forcing, Params, float], Tuple[State, Diag]], step_fn)

    st: State = init
    report: Dict[str, Any] = {"timings": {"per_step_sec": []}, "last_diag": None}

    # Provide a default dt if not supplied via params; examples can override
    dt = cast(float, cast(Mapping[str, Any], params.get("time", {})).get("dt", 1.0))

    fiter = _as_iter(forcing_stream)

    for k in range(n_steps):
        forcing = next(fiter, {})
        t0 = perf_counter()
        st, diag = step(st, forcing, params, dt, xp=xp)
        t1 = perf_counter()

        # Timing
        dur = t1 - t0
        # Attach per-step timing into diag for hook consumption
        (diag.setdefault("timings", {}))["step_sec"] = dur
        report["timings"]["per_step_sec"].append(dur)

        # Invoke hooks (observational only)
        for hook in hooks:
            # Pass-through extra kwargs for richer hook signatures
            try:
                hook(k, st, diag)  # type: ignore[misc]
            except TypeError:
                # Support hooks that expect named params/xp as described in design
                hook(k, st, diag, params=params, xp=xp)  # type: ignore[misc]

        report["last_diag"] = diag

    return st, report
