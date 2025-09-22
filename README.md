# py-gcmi

The Python reference implementation of GCMI, with core APIs, middleware, hooks, and examples.

WSGI-inspired minimal usage
- Like WSGI’s tiny app(environ, start_response) and server loop, GCMI offers a tiny step callable and a minimal driver to run it.
- Goal: make the “hello world” path trivial and elegant to encourage incremental assembly.

Quick start (minimal, WSGI-like)
- Requirements: Python ≥ 3.10
- Clone the repo, then run the minimal example:

```bash
python examples/gcmi_minimal_driver.py
```

You should see:
- CSV lines of per-step timing (k, step_sec) printed by the timing hook
- Final state and an average step time summary

Minimal “hello world” for GCMI
This is the closest analogue to a WSGI hello world.

```python
import sys
import numpy as np
from typing import Any, Dict, Tuple

from gcmi.drivers import run
from gcmi.hooks.timing import timer_hook

def step(state: Dict[str, Any], forcing: Dict[str, Any], params: Dict[str, Any], dt: float, *, xp: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    # Identity physics: pass state through, return minimal diag
    return state, {"gcmi_mw": []}

xp = np
cfg = {
    "state0": {"T": xp.array(300.0)},
    "params": {"time": {"dt": 1.0}},
}
forcing_stream = lambda k: {}  # empty forcing
hook = timer_hook(sink=sys.stdout, fmt="csv")

final_state, report = run(
    step,
    cfg=cfg,
    xp=xp,
    forcing_stream=forcing_stream,
    n_steps=5,
    hooks=(hook,),
)
print("Final state:", {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in final_state.items()})
```

Where the pieces map (WSGI → GCMI)
- app(environ, start_response) → step(state, forcing, params, dt, *, xp)
- make_server(..., app).serve_forever() → run(step, cfg=..., xp=..., forcing_stream=..., n_steps=..., hooks=...)
- WSGI middleware → GCMI middleware (functional wrappers around step); hooks are observational

Repository guide
- Core API: gcmi/core/api.py (init_fn, step_fn, run_fn)
- Minimal driver (WSGI-like): gcmi/drivers/ (StepCallable, make_runner, run)
- Hooks: gcmi/hooks/ (e.g., timer_hook)
- Examples:
  - examples/wsgi_minimal.py (reference WSGI hello)
  - examples/gcmi_minimal.py (uses core.run_fn; shows rebinding step_fn)
  - examples/gcmi_minimal_driver.py (WSGI-like one-shot run helper)
- Design & Plan:
  - docs/design.md (Core contracts, middleware, hooks, ops)
  - docs/plan.md (Milestones M0–M4)

Next steps (after hello world)
- Add a custom step for tiny tendencies (see examples/gcmi_minimal.py for a simple _custom_step)
- Wrap with middleware: CFL guards, positivity, flux limiters, energy fix
- Attach more hooks: energy/water budgets, spectra (if SpectralOps is enabled)
- Move from NumPy to JAX/Torch by swapping xp without changing the step contract

Contributing
- See AGENTS.md for engineering protocols (ruff, black, isort, mypy --strict, pytest, pre-commit)
- Keep changes small and atomic with tests and doc updates
