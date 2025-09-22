"""
A minimal GCMI example using the tiny driver API.

Goal: make the "hello world" for GCMI as elegant as possible.
Here:
  - You write a tiny step(dt, state, forcing, params, *, xp) -> (state, diag)
  - You call run(step, cfg=..., forcing_stream=..., n_steps=..., hooks=..., xp=...)
  - That's it.

This mirrors the WSGI style and avoids rebinding module-level step_fn.
"""

from typing import Any, Dict, Tuple

import numpy as xp  # could be jax.numpy, cupy, etc.

from gcmi.drivers import run


def step(
    dt: float,
    state: Dict[str, Any],
    *,
    xp: Any,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return state, {}


if __name__ == "__main__":
    final_state, report = run(
        step,
        n_steps=5,
        xp=xp
    )
    print("Final state:", final_state)
    print("Report:", report)

