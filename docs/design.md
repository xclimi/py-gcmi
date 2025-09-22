# System Design — py-gcmi

Status: Draft
Aligns with: GCMI Core 0.1 (proposed); SpectralOps 0.1 (experimental)
Last Updated: 2025-09-22

1. Architectural Overview

- Philosophy: WSGI-style minimal core plus composable middleware that modify state, and read-only hooks for observability. Favor pure functions and explicit data flow. No implicit globals. Backend neutrality via xp and ops facades (NumPy/JAX/Torch).
- Layers
  - Core: init_fn/step_fn/run_fn – minimal, typed, backend-neutral.
  - Middleware: numerical stabilization, limiting, conservation, methods.
  - Hooks: logging/diagnostics/IO; strictly observational.
  - Ops: grid operators and optional spectral operators.
  - Drivers: default sequential runner; space for parallel variants later.
  - Config: env → cfg → params bridge with validation.

2. Core API (GCMI Core 0.1)

```python
from typing import Protocol, Mapping, Any, Tuple, Callable, Dict

class ArrayLike(Protocol): ...
class XP(Protocol): ...  # numpy, jax.numpy, torch-like

State  = Dict[str, ArrayLike]
Forcing = Dict[str, ArrayLike]
Params = Mapping[str, Any]
Diag   = Dict[str, Any]

StepFn = Callable[[State, Forcing, Params, float], Tuple[State, Diag]]

def init_fn(cfg: Mapping[str, Any], *, xp: XP) -> Tuple[State, Params]: ...
def step_fn(state: State, forcing: Forcing, params: Params, dt: float, *, xp: XP) -> Tuple[State, Diag]: ...
def run_fn(init: State, params: Params, forcing_stream, *,
           xp: XP, n_steps: int,
           hooks: Tuple[Callable[[int, State, Diag], None], ...] = ()
          ) -> Tuple[State, Mapping[str, Any]]: ...
```

- Semantics
  - State/Forcing/Params specify keys and meanings, not concrete tensor types.
  - No hidden global state; reproducibility via Params and State["misc"]["rng"].
  - Versioning: diag["gcmi_version"] and per-middleware metadata for audit.

3. Data Model: Initial Keys and Semantics

- State: {"T", "q", "u", "v", "misc", optional "spec": {"zeta_lm","div_lm", ...}}
- Forcing: {"SW","LW","topo", ...} according to examples.
- Params:
  - "grid": {"dx_min", "metric", "mask", ...}
  - "ops": references to backend implementations (grid/spectral)
  - "energy_budget": {"target_total", ...}
  - "backend": {"xp": xp module or compatible namespace}
- Diag: {"gcmi_mw": [ ... ], "timings": ..., other per-step diagnostics}

4. Middleware Contract

- Definition: mw: (StepFn) -> StepFn, pure and composable, no global state.
- Responsibilities: stabilization (CFL/substeps), limiters, positivity, conservation projection, semi-implicit methods, nudging, substepping, RNG normalization.
- Ordering guidelines:
  - Outer: stability wrappers (CFL/substeps).
  - Middle: local fixups (limiters, hyperdiff, positivity).
  - Tail: global consistency (energy fix, conservation projection).
  - Boundary: sponge/radiation handling.
- Each middleware appends metadata into diag["gcmi_mw"].

5. Hook Contract

- Signature: hook(k: int, state: State, diag: Diag, *, params, xp, ...) -> None.
- Strictly observational: logging, budget tables, IO, timing.
- Must not modify State or Params.

6. Ops Facade (Grid)

- Encapsulates differential operators, stencils, basic filters, and metric handling.
- Backend-neutral call sites; configurable via Params["ops"] and xp.

7. SpectralOps Extension (Experimental)

```python
class SpectralOps(Protocol):
    def sh_forward(self, grid_scalar) -> "spec_array": ...
    def sh_inverse(self, spec_array) -> "grid_scalar": ...
    def uv_from_vort_div(self, zeta_lm, div_lm): ...
    def vort_div_from_uv(self, u, v): ...
    def laplacian_eig(self, l: int) -> float: ...  # -l(l+1)/a^2
    def spectral_filter(self, spec_array, kind="hypervisc", power=4, coef=...): ...
    def dealias(self, spec_array, rule="two_thirds"): ...
    def helmholtz_solve(self, spec_rhs, alpha): ...
```

- State may lazily maintain both grid and spectral forms. Physical processes remain on grid; dynamics via pseudo-spectral middleware.
- Implementation examples: NumPy+shtns, JAX+jax-sht, Torch+custom kernels. Optional and capability-negotiated.

8. Spectral Dynamics Middleware (Pseudo-spectral path)

- with_spectral_dyn_core(step_core, theta=0.5, hypervisc=(4, 1e-4)):
  - spec→grid (recover u,v), compute nonlinear terms on grid, grid→spec, advance (explicit or θ semi-implicit via Helmholtz), spectral hyperviscosity + dealiasing, write back and delegate to step_core, log metadata.
- Physics interplay: physics tendencies on grid; synchronization managed lazily; conservation checks handled by generic middleware.

9. Configuration and Schemas

- load_cfg() produces validated cfg. Env variables are inputs but normalized into cfg; downstream code does not read env directly.
- Typical cfg keys:
  - backend.xp, nsteps, grid (dx, dy, mask), spectral (truncation, radius, dealiasing), middleware parameters (coefficients, limits), IO/hook intervals.

10. Error Handling, Logging, Observability

- No silent failures; explicit exceptions for invalid inputs/configurations.
- logging with structured output in production contexts (JSON).
- Hooks produce CSV/NDJSON for budgets/timings; diag carries per-step metadata.

11. Testing Strategy

- pytest with strict typing and style checks (mypy --strict, ruff, black, isort).
- Baseline tests:
  - Conservation baseline (dry, no forcing): mass/energy/water within epsilon over N steps.
  - Stability baseline under varied dt/resolution: no blow-up; CFL guard effective.
  - Reproducibility baseline (fixed RNG): within tolerances across supported backends where applicable.
  - Spectral roundtrip and Helmholtz tests (if SpectralOps is enabled).
- Coverage monitored via pytest-cov.

12. Performance Considerations

- Minimize Python-level loops; vectorize via xp.
- Batch operations in ops; profile with timing hooks.
- Spectral transforms are global reductions; keep optional and well-factored; provide knobs (power, coef, dealias rule).

13. CI/CD and Tooling

- Pre-commit: ruff, black, isort, mypy --strict, pytest.
- GitHub Actions: enforce AGENTS.md checklist on PRs and main.
- CHANGELOG maintained with Keep a Changelog format.

14. API Stability and Versioning

- py-gcmi __version__ declares supported GCMI Core/SpectralOps versions.
- Backward-compatible additions favored; breaking changes require ADR + semver bump.

15. Example Assembly (Sketch)

```python
from gcmi.core.api import init_fn, run_fn, step_fn
from gcmi.middleware import (
    with_cfl_guard, with_hyperdiff, with_flux_limiter,
    with_positivity, with_energy_fix, with_conservation_projection
)
from gcmi.hooks import energy_budget_hook

xp = numpy  # or jax.numpy / torch
state0, params = init_fn(cfg, xp=xp)

step = step_fn  # core tendencies only
step = with_flux_limiter(step, scheme="mc", vars=("q","T"))
step = with_hyperdiff(step, coeff=2e-3, order=4, vars=("T","u","v"))
step = with_positivity(step, vars=("q",), lower=0.0, conserve="total_q")
step = with_energy_fix(step, budget=("dry_static","latent","kinetic"))
step = with_conservation_projection(step, conserve=("total_mass","moist_energy"))
step = with_cfl_guard(step, cfl_max=0.8, wave_speed_cb=wave_speed_cb)

final_state, report = run_fn(
    state0, params, forcing_stream,
    xp=xp, nsteps=cfg["nsteps"],
    hooks=(lambda k, st, dg: energy_budget_hook(k, st, dg, params=params, xp=xp),)
)
```

16. Ergonomic Dict Handling (Pythonic “destructuring”)

Goal: keep mapping-based APIs (State/Params/Forcing) while writing concise, readable, and type-checkable access patterns.

Core idioms
- operator.itemgetter for tuple-unpack (fast, concise):
  ```python
  from operator import itemgetter as ig
  T, q, u, v = ig("T", "q", "u", "v")(state)
  ```
- Structural pattern matching (Python 3.10+) for readability and capturing “rest”:
  ```python
  match state:
      case {"T": T, "q": q, "u": u, "v": v, **rest}:
          ...
  match params:
      case {"grid": {"dx_min": dx_min, **_}, **_}:
          ...
  ```
- NamedTuple “views” for frequent bundles (adds attribute names):
  ```python
  from typing import NamedTuple, Any
  class UV(NamedTuple):
      u: Any; v: Any
  def uv_view(d: dict) -> UV:
      from operator import itemgetter as ig
      return UV(*ig("u","v")(d))
  u, v = uv_view(state)
  ```

Lightweight utilities (to be placed under gcmi/utils/struct.py)
- take: minimal, fast destructuring for required keys (thin wrapper on itemgetter)
  ```python
  from operator import itemgetter
  from typing import Mapping, Any, Tuple
  def take(d: Mapping[str, Any], *keys: str) -> Tuple[Any, ...]:
      return itemgetter(*keys)(d)
  ```
- require: take but with clearer KeyError message
  ```python
  def require(d: Mapping[str, Any], *keys: str) -> Tuple[Any, ...]:
      try:
          return itemgetter(*keys)(d)
      except KeyError as e:
          miss = e.args[0]
          raise KeyError(f"Missing required key '{miss}'. Expected {keys}. Available {list(d.keys())}") from e
  ```
- take_nested: dotted-path access for nested dicts (cfg/params)
  ```python
  from typing import Any, Mapping, Tuple
  def take_nested(d: Mapping[str, Any], *paths: str) -> Tuple[Any, ...]:
      def get_path(obj: Any, path: str) -> Any:
          cur = obj
          for seg in path.split("."):
              cur = cur[seg]
          return cur
      return tuple(get_path(d, p) for p in paths)
  # Example: dx_min, theta = take_nested(params, "grid.dx_min", "spectral.semi_implicit.theta")
  ```
- split_keys: pick keys and return (picked, rest)
  ```python
  from typing import Mapping
  def split_keys(d: Mapping[str, Any], *keys: str) -> tuple[dict, dict]:
      picked = {k: d[k] for k in keys}
      rest = {k: v for k, v in d.items() if k not in picked}
      return picked, rest
  ```

Style guidance
- Prefer take/require in hot paths for brevity and speed; match-case when clarity is more important (especially when capturing the remainder).
- Use take_nested in config validation/middleware assembly instead of multi-index chains for readability.
- Avoid ad-hoc dotdict wrappers; they hinder static checking and can mask KeyError bugs.
- TypedDicts are acceptable for developer ergonomics when stabilizing commonly used views (e.g., StateTD subset), while keeping runtime objects as plain dicts.

16.5 Declarative Requirements (decorator + early-check middleware)

Motivation
- Components (core, middleware, spectral adapters) often need certain keys/semantics in state/params/forcing (e.g., params.spectral.radius).
- Make these needs explicit, discoverable, and machine-checkable without cluttering the numerical code.

Mechanism
- Declarative requirement spec + decorator (gcmi.utils.requirements):
  - Requirement(where: "state"|"params"|"forcing", path: "a.b.c", required=True, type=..., predicate=..., severity="error"|"warn")
  - @requires(...): attach one or more Requirement to a StepFn or any callable
  - get_requirements(fn): retrieve attached requirements (follows __wrapped__ chain)
  - validate_requirements(...): check containers and return (errors, warnings)
- Early-check middleware (gcmi.middleware.requirements):
  - with_requirements_check(step, max_checks=3, raise_on_error=True, record_warnings=True)
  - Validates attached (+extra) requirements only for the first max_checks calls, then skips for performance.

Example
```python
from gcmi.utils.requirements import Requirement, requires
from gcmi.middleware.requirements import with_requirements_check

@requires(
    Requirement("params", "spectral.radius", type=(int, float), predicate=lambda r: r > 0.0),
    Requirement("state", "misc.seed", type=int, severity="warn"),  # optional warning example
)
def step_core(state, forcing, params, dt, *, xp):
    # physics/dynamics tendencies here...
    return state, {}

# Enforce only in early steps (e.g., first 3); then skip
step = with_requirements_check(step_core, max_checks=3, raise_on_error=True)
```

Behavior and auditing
- On violation during early steps:
  - If raise_on_error=True: raises RequirementError (fail fast).
  - Else: attaches a summary to diag["gcmi_requirements"] for auditing.
- After max_checks invocations, all checks are skipped to remove overhead.
- Requirements are composable across wrappers via __wrapped__ metadata.

Recommended usage
- Middleware and drivers that have structural expectations (e.g., spectral ops) should declare them via @requires.
- Wrap the assembled step with with_requirements_check in examples/drivers to enforce correctness in the first few steps of a run.
- Keep predicates simple and side-effect free; use type and required flags where possible.

17. Open Questions and ADRs

- ADR-000: RNG normalization strategy across backends (algorithm, seeding, reproducibility).
- ADR-001: Default middleware chain ordering and coefficients; recommended presets.
- ADR-002: Spectral normalization conventions and truncation defaults.
- ADR-003: Config schema and validation library choices (pydantic vs. minimal schema).
- ADR-004: Dict ergonomics helpers standardization and their scope of use.

18. Glossary

- GCMI: Gateway Climate Modeling Interface (Core API).
- Middleware: Functions that transform StepFn and modify numerical outcomes.
- Hooks: Observers that never modify State.
- SpectralOps: Optional facade for spherical harmonic operations.
