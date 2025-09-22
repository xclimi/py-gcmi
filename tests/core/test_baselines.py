import copy
from typing import Any, Dict

import gcmi.core.api as core_api
from gcmi.hooks import energy_budget_hook, water_budget_hook
from gcmi.middleware import with_cfl_guard


class XPStub:
    # Minimal 'xp' stub used by tests; compatible with our hooks/ops placeholders
    def sum(self, x):
        try:
            return sum(x)  # works for lists
        except TypeError:
            return x  # scalar

    def zeros_like(self, x):
        try:
            return [0 for _ in x]
        except TypeError:
            return 0


def _forcing_fn(k: int) -> Dict[str, Any]:
    # No forcing for baseline tests
    return {}


def _copy_state(state: Dict[str, Any]) -> Dict[str, Any]:
    # Deep copy lists; values are kept simple in tests
    return copy.deepcopy(state)


def test_conservation_identity_baseline():
    """
    Dry/no forcing: totals remain constant with identity core step.
    """
    xp = XPStub()
    state0 = {
        "T": [1.0, 2.0, 3.0],
        "q": [0.1, 0.2, 0.3],
        "u": [1.0, -1.0, 0.5],
        "v": [0.5, 0.0, -0.5],
        "misc": {"seed": 123},
    }
    params = {"time": {"dt": 1.0}, "grid": {"dx_min": 1.0}}

    # Totals at start
    t0 = float(xp.sum(state0["T"]))
    q0 = float(xp.sum(state0["q"]))
    u0 = float(xp.sum(state0["u"]))
    v0 = float(xp.sum(state0["v"]))

    # Use identity core step (module-level default) with no middleware
    init_state, init_params = state0, params

    # Run
    final_state, report = core_api.run_fn(
        init=init_state,
        params=init_params,
        forcing_stream=_forcing_fn,
        xp=xp,
        n_steps=5,
        hooks=(
            energy_budget_hook(),  # just to exercise hooks path
            water_budget_hook(),
        ),
    )

    # Totals at end should match
    assert float(xp.sum(final_state["T"])) == t0
    assert float(xp.sum(final_state["q"])) == q0
    assert float(xp.sum(final_state["u"])) == u0
    assert float(xp.sum(final_state["v"])) == v0


def test_cfl_guard_substepping_and_metadata():
    """
    CFL guard: when vmax*dt/dx exceeds threshold, uses substepping and records metadata.
    """
    xp = XPStub()
    state0 = {"T": [1.0, 2.0], "misc": {}}
    params = {"time": {"dt": 1.0}, "grid": {"dx_min": 1.0}}

    # Define a trivial step that just echoes diag; allows metadata injection to be observed
    def base_step(state, forcing, params, dt, *, xp=None):
        return state, {"gcmi_mw": []}

    # Wave speed large enough to force substeps for cfl_max=0.5
    def wave_speed_cb(state, params, xp):
        return 10.0

    wrapped = with_cfl_guard(base_step, cfl_max=0.5, wave_speed_cb=wave_speed_cb)

    # Monkey-patch the module-level step_fn for this test
    old = core_api.step_fn
    core_api.step_fn = wrapped  # type: ignore[assignment]
    try:
        final_state, report = core_api.run_fn(
            init=state0,
            params=params,
            forcing_stream=_forcing_fn,
            xp=xp,
            n_steps=1,
            hooks=(),
        )
    finally:
        core_api.step_fn = old  # restore

    # The last diag is in report["last_diag"]
    diag = report["last_diag"] or {}
    mw = diag.get("gcmi_mw", [])
    assert mw, "Expected middleware metadata entries"
    cfl_entries = [m for m in mw if m.get("name") == "cfl_guard"]
    assert cfl_entries, "Expected cfl_guard metadata entry"
    assert cfl_entries[-1]["n_substeps"] >= 2


def test_reproducibility_identity_fixed_rng():
    """
    With identity step and fixed initial state, repeated runs produce identical final states.
    """
    xp = XPStub()
    state0 = {"q": [0.1, 0.2, 0.3], "misc": {"seed": 42}}
    params = {"time": {"dt": 1.0}}

    def run_once():
        st, rep = core_api.run_fn(
            init=_copy_state(state0),
            params=params,
            forcing_stream=_forcing_fn,
            xp=xp,
            n_steps=3,
            hooks=(),
        )
        return st

    st1 = run_once()
    st2 = run_once()

    assert (
        st1 == st2
    ), "Final states should be identical across runs for identity dynamics"
