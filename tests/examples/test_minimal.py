from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from gcmi.drivers import run


def test_minimal_hello_world_runs() -> None:
    # Minimal signature: step(dt, state, *, xp)
    def step(dt: float, state: Dict[str, Any], *, xp: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Ensure driver passes through the backend namespace
        assert xp is np
        # Identity step: return state and empty diag
        return state, {}

    final_state, report = run(step, n_steps=3, xp=np)

    assert isinstance(final_state, dict)
    assert "timings" in report and "per_step_sec" in report["timings"]
    timings = report["timings"]["per_step_sec"]
    assert isinstance(timings, list)
    assert len(timings) == 3
    assert "last_diag" in report and "timings" in report["last_diag"]
    assert "step_sec" in report["last_diag"]["timings"]


def test_minimal_with_cfg_state_pass_through() -> None:
    cfg = {"state0": {"T": np.array(300.0)}, "params": {"time": {"dt": 1.0}}}

    def step(dt: float, state: Dict[str, Any], *, xp: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Identity step
        return state, {}

    final_state, report = run(step, n_steps=2, xp=np, cfg=cfg)

    assert "T" in final_state
    # Identity step should preserve initial field
    assert np.allclose(final_state["T"], 300.0)
    assert len(report["timings"]["per_step_sec"]) == 2


def test_back_compat_signature_full_core_order() -> None:
    # Back-compatible signature: step(state, forcing, params, dt, *, xp)
    def step(
        state: Dict[str, Any],
        forcing: Dict[str, Any],
        params: Dict[str, Any],
        dt: float,
        *,
        xp: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Driver should supply empty forcing/params when omitted by caller
        assert isinstance(forcing, dict)
        assert isinstance(params, dict)
        assert xp is np
        return state, {}

    final_state, report = run(step, n_steps=1, xp=np)

    assert isinstance(final_state, dict)
    assert len(report["timings"]["per_step_sec"]) == 1


def test_forcing_stream_and_dt_override() -> None:
    calls: List[int] = []

    def step(dt: float, state: Dict[str, Any], *, xp: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # Check dt override propagates
        assert dt == 2.5
        calls.append(1)
        return state, {}

    final_state, report = run(step, n_steps=4, xp=np, dt=2.5, forcing_stream=lambda k: {})
    assert isinstance(final_state, dict)
    assert len(calls) == 4
    assert len(report["timings"]["per_step_sec"]) == 4


def test_diag_normalization_when_none() -> None:
    def step(dt: float, state: Dict[str, Any], *, xp: Any) -> Tuple[Dict[str, Any], None]:
        # Return None diag; driver should normalize and attach timings
        return state, None  # type: ignore[return-value]

    _, report = run(step, n_steps=1, xp=np)
    assert "last_diag" in report and "timings" in report["last_diag"]
    assert "step_sec" in report["last_diag"]["timings"]
