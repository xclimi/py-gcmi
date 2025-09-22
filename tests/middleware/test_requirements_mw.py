import pytest

from gcmi.middleware.requirements import with_requirements_check
from gcmi.utils.requirements import Requirement, requires


def make_dummy_step(diag_key: str = "ok"):
    def step(state, forcing, params, dt, *, xp=None):
        # Pass-through state; emit minimal diag
        return state, {diag_key: True}

    return step


def test_with_requirements_check_raises_on_error_for_first_calls():
    @requires(
        Requirement(
            "params", "spectral.radius", type=(int, float), predicate=lambda r: r > 0
        )
    )
    def step(state, forcing, params, dt, *, xp=None):
        return state, {}

    wrapped = with_requirements_check(step, max_checks=2, raise_on_error=True)

    # Missing spectral.radius -> should raise on first call
    with pytest.raises(Exception) as ei:
        wrapped(state={}, forcing={}, params={}, dt=1.0, xp=None)
    assert "Requirements not satisfied" in str(ei.value)


def test_with_requirements_check_records_warnings_and_skips_after_max_checks():
    @requires(
        Requirement(
            "params", "spectral.radius", type=(int, float), predicate=lambda r: r > 0
        )
    )
    def step(state, forcing, params, dt, *, xp=None):
        return state, {}

    # Do not raise to allow us to inspect diag; check only first 2 calls
    wrapped = with_requirements_check(
        step, max_checks=2, raise_on_error=False, record_warnings=True
    )

    # Call 1: expect requirements summary with an error entry
    st1, dg1 = wrapped(state={}, forcing={}, params={}, dt=1.0, xp=None)
    assert "gcmi_requirements" in dg1
    assert len(dg1["gcmi_requirements"]) == 1
    assert dg1["gcmi_requirements"][0]["errors"]

    # Call 2: still within max_checks -> another entry
    st2, dg2 = wrapped(state={}, forcing={}, params={}, dt=1.0, xp=None)
    assert "gcmi_requirements" in dg2
    assert (
        len(dg2["gcmi_requirements"]) == 1
    )  # per-call diag only contains its own entry

    # Call 3: beyond max_checks -> no validation/entry
    st3, dg3 = wrapped(state={}, forcing={}, params={}, dt=1.0, xp=None)
    assert "gcmi_requirements" not in dg3


def test_with_requirements_check_passes_when_requirements_met():
    @requires(
        Requirement(
            "params", "spectral.radius", type=(int, float), predicate=lambda r: r > 0
        )
    )
    def step(state, forcing, params, dt, *, xp=None):
        return state, {"inner": True}

    wrapped = with_requirements_check(
        step, max_checks=2, raise_on_error=True, record_warnings=True
    )

    params = {"spectral": {"radius": 6_371_229.0}}
    st, dg = wrapped(state={}, forcing={}, params=params, dt=1.0, xp=None)
    # No errors/warnings recorded when satisfied
    assert "gcmi_requirements" not in dg
    assert dg.get("inner") is True
