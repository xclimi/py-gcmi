import pytest

from gcmi.utils.struct import require, split_keys, take, take_nested


def test_take_happy_path():
    state = {"T": 1, "q": 2, "u": 3, "v": 4}
    T, q, u, v = take(state, "T", "q", "u", "v")
    assert (T, q, u, v) == (1, 2, 3, 4)

    # Single key still returns a scalar when unpacked as a 1-tuple
    (T_only,) = take(state, "T")
    assert T_only == 1


def test_take_missing_key_raises_keyerror():
    state = {"T": 1}
    with pytest.raises(KeyError):
        take(state, "T", "q")  # q missing


def test_require_happy_path_and_error_message():
    params = {"grid": {"dx_min": 1000}}
    (dx_min,) = require(params["grid"], "dx_min")
    assert dx_min == 1000

    with pytest.raises(KeyError) as ei:
        require(params["grid"], "dx_min", "dy_min")  # dy_min missing
    msg = str(ei.value)
    assert "Missing required key" in msg
    assert "dy_min" in msg
    assert "dx_min" in msg  # appears in Expected list
    assert "Available keys" in msg


def test_take_nested_happy_path():
    params = {
        "grid": {"dx_min": 1000, "mask": [[1, 0], [0, 1]]},
        "spectral": {"semi_implicit": {"theta": 0.5}},
    }
    dx_min, theta = take_nested(params, "grid.dx_min", "spectral.semi_implicit.theta")
    assert dx_min == 1000
    assert theta == 0.5


def test_take_nested_missing_key_raises_keyerror():
    params = {"grid": {"dx_min": 1000}}
    with pytest.raises(KeyError) as ei:
        take_nested(params, "grid.dy_min")
    assert "dy_min" in str(ei.value)


def test_take_nested_type_error_on_non_mapping():
    params = {"grid": {"dx_min": 1000}}
    # grid.dx_min is an int; asking for another segment should raise TypeError
    with pytest.raises(TypeError) as ei:
        take_nested(params, "grid.dx_min.value")
    assert "non-mapping object" in str(ei.value)


def test_split_keys_basic():
    params = {
        "grid": {"dx_min": 1000},
        "backend": {"xp": "numpy"},
        "energy_budget": {"target_total": 42.0},
    }
    picked, rest = split_keys(params, "grid", "backend", "nonexistent")
    assert picked == {"grid": {"dx_min": 1000}, "backend": {"xp": "numpy"}}
    assert rest == {"energy_budget": {"target_total": 42.0}}
