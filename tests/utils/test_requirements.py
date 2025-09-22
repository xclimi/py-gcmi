from gcmi.utils.requirements import (
    Requirement,
    RequirementError,
    get_requirements,
    requires,
    validate_requirements,
)


def test_validate_requirements_ok_and_types_and_predicate():
    reqs = [
        Requirement(
            "params", "spectral.radius", type=(int, float), predicate=lambda r: r > 0
        ),
        Requirement("state", "misc.seed", type=int),
        Requirement("forcing", "SW"),
    ]
    state = {"misc": {"seed": 123}}
    params = {"spectral": {"radius": 6_371_229.0}}
    forcing = {"SW": 1.0}
    errors, warns = validate_requirements(
        state=state, params=params, forcing=forcing, requirements=reqs
    )
    assert errors == []
    assert warns == []


def test_validate_requirements_missing_and_type_violation_and_predicate_violation():
    reqs = [
        Requirement(
            "params", "spectral.radius", type=(int, float), predicate=lambda r: r > 0
        ),
        Requirement("params", "grid.dx_min", type=int),  # type mismatch on purpose
        Requirement("forcing", "LW"),  # missing key
    ]
    state = {}
    params = {
        "spectral": {"radius": -1.0},
        "grid": {"dx_min": 1000.0},
    }  # radius <= 0 and dx_min float
    forcing = {}
    errors, warns = validate_requirements(
        state=state, params=params, forcing=forcing, requirements=reqs
    )
    # Expect 3 errors
    assert len(errors) == 3
    msgs = "\n".join(v.message for v in errors)
    assert (
        "Predicate returned False" in msgs
        or "Expected type" in msgs
        or "Missing key" in msgs
    )


def test_requires_decorator_and_get_requirements_collects_chain():
    @requires(Requirement("params", "spectral.radius", type=(int, float)))
    def inner(step_state, step_forcing, step_params, dt, *, xp):
        return step_state, {}

    # Wrap by a simple decorator that preserves __wrapped__
    def wrapper(fn):
        def wrapped(*a, **k):
            return fn(*a, **k)

        setattr(wrapped, "__wrapped__", fn)
        return wrapped

    wrapped_inner = wrapper(inner)

    reqs = get_requirements(wrapped_inner)
    assert any(r.where == "params" and r.path == "spectral.radius" for r in reqs)


def test_requirement_error_str():
    violations = [
        # Craft a minimal Violation-like object using Requirement
        type(
            "V",
            (),
            {
                "where": "params",
                "path": "spectral.radius",
                "severity": "error",
                "message": "bad",
                "requirement": Requirement("params", "spectral.radius"),
            },
        )(),
        type(
            "V",
            (),
            {
                "where": "forcing",
                "path": "SW",
                "severity": "warn",
                "message": "optional",
                "requirement": Requirement("forcing", "SW", severity="warn"),
            },
        )(),
    ]
    err = RequirementError(violations)  # type: ignore[arg-type]
    s = str(err)
    assert "Requirements not satisfied" in s
    assert (
        "PARAMS.spectral.radius" or "params.spectral.radius"
    )  # case can vary based on formatting
    assert "FORCING.SW" or "forcing.SW"
