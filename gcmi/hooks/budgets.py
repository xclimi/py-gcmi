from __future__ import annotations

import json
from typing import IO, Any, Callable, Iterable, Mapping, Optional, Sequence

from gcmi.ops import grid as grid_ops

Hook = Callable[[int, dict[str, Any], dict[str, Any]], None]


def _sum_any(x: Any, *, xp: Any | None) -> float:
    """
    Backend-neutral summation with graceful fallback to Python sum()/float().
    """
    if xp is not None:
        try:
            s = xp.sum(x)  # type: ignore[attr-defined]
            try:
                return float(s)  # type: ignore[arg-type]
            except Exception:
                return s  # type: ignore[return-value]
        except Exception:
            pass
    # Fallbacks
    try:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            return float(sum(x))  # type: ignore[arg-type]
    except Exception:
        pass
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return 0.0


def energy_budget_hook(
    *,
    terms: Sequence[str] = ("dry_static", "latent", "kinetic"),
    # Mapping from energy term name to state keys to sum; placeholders for M1
    term_vars: Mapping[str, Sequence[str]] = {
        "dry_static": ("T",),
        "latent": ("q",),
        "kinetic": ("u", "v"),
    },
    sink: Optional[IO[str]] = None,
    fmt: str = "csv",  # 'csv' or 'ndjson'
) -> Hook:
    """
    Construct a hook that computes simple energy-like totals from state variables.

    For M1 this is a placeholder that sums selected state arrays by term using a backend-neutral
    total; real energy calculations can replace this mapping later.

    The hook records results under diag['budgets']['energy'] and optionally writes to sink.
    """

    def hook(k: int, state: dict[str, Any], diag: dict[str, Any], *_, **kwargs) -> None:
        xp = kwargs.get("xp", None)

        energy: dict[str, float] = {}
        for term in terms:
            total_val = 0.0
            for var in term_vars.get(term, ()):
                if var in state:
                    try:
                        total_val += grid_ops.total(state[var], xp=xp)
                    except Exception:
                        total_val += _sum_any(state[var], xp=xp)
            energy[term] = float(total_val)

        budgets = diag.setdefault("budgets", {})
        budgets.setdefault("energy", {})[k] = energy

        if sink is None:
            return

        if fmt == "csv":
            # One line per step with comma-separated term totals (order per 'terms')
            row = [str(k)] + [str(energy.get(t, 0.0)) for t in terms]
            sink.write(",".join(row) + "\n")
            sink.flush()
        elif fmt == "ndjson":
            rec = {"k": k, "energy": energy}
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
        else:
            raise ValueError(f"Unsupported fmt: {fmt}")

    return hook


def water_budget_hook(
    *,
    var: str = "q",
    sink: Optional[IO[str]] = None,
    fmt: str = "csv",
) -> Hook:
    """
    Construct a hook that computes a simple water budget: total of 'var' (default 'q').

    Records under diag['budgets']['water'] and optionally writes to sink.
    """

    def hook(k: int, state: dict[str, Any], diag: dict[str, Any], *_, **kwargs) -> None:
        xp = kwargs.get("xp", None)
        total_q = 0.0
        if var in state:
            try:
                total_q = grid_ops.total(state[var], xp=xp)
            except Exception:
                total_q = _sum_any(state[var], xp=xp)

        budgets = diag.setdefault("budgets", {})
        budgets.setdefault("water", {})[k] = {var: float(total_q)}

        if sink is None:
            return

        if fmt == "csv":
            sink.write(f"{k},{float(total_q)}\n")
            sink.flush()
        elif fmt == "ndjson":
            sink.write(json.dumps({"k": k, var: float(total_q)}) + "\n")
            sink.flush()
        else:
            raise ValueError(f"Unsupported fmt: {fmt}")

    return hook
