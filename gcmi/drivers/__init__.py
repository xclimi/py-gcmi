"""Drivers entrypoint exposing a WSGI-like minimal runner API for GCMI.

WSGI analogy:
- app(environ, start_response)                    ~ step(state, forcing, params, dt, *, xp)
- make_server(host, port, app).serve_forever()    ~ make_runner(step, cfg, xp).run(forcing, n_steps)

Public API:
- StepCallable: protocol for step callables
- make_runner: bind step+cfg+xp(+hooks) to create a small runner callable
- run: one-shot helper to init and execute a run

Example:
    import numpy as np
    from gcmi.drivers import StepCallable, run

    def step(state, forcing, params, dt, *, xp):
        return state, {"gcmi_mw": []}

    cfg = {"state0": {"T": np.array(300.0)}, "params": {"time": {"dt": 1.0}}}
    final_state, report = run(step, cfg=cfg, xp=np, forcing_stream=lambda k: {}, n_steps=5)
"""

from __future__ import annotations

from .minimal import StepCallable, make_runner, run

__all__ = ["StepCallable", "make_runner", "run"]
