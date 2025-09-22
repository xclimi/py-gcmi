from __future__ import annotations

import json
from typing import IO, Any, Callable, Optional

# Hook signature (observational only): accepts optional params/xp via kwargs
Hook = Callable[[int, dict[str, Any], dict[str, Any]], None]


def timer_hook(
    *,
    sink: Optional[IO[str]] = None,
    fmt: str = "csv",
    include_diag: bool = False,
) -> Hook:
    """
    Create a timing hook that records per-step elapsed wall-clock time.

    The run loop is expected to attach a per-step timing into diag["timings"]["step_sec"].
    This hook reads that value and optionally writes it to a stream.

    Args:
        sink: Optional text file-like (opened in append mode) to write records.
              If None, the hook only ensures a timings entry exists in diag.
        fmt: 'csv' or 'ndjson' when sink is provided.
        include_diag: If True and fmt == 'ndjson', include the full diag in the record.

    Returns:
        A hook callable with signature hook(k, state, diag, ..., params=?, xp=?)
    """

    def hook(k: int, state: dict[str, Any], diag: dict[str, Any], *_, **__) -> None:
        timings = diag.setdefault("timings", {})
        step_sec = timings.get("step_sec", None)

        # If there is no timing attached yet, do nothing beyond ensuring the key exists
        if step_sec is None:
            return

        if sink is None:
            return

        if fmt == "csv":
            # Write header once if file is empty? We avoid file state checks for simplicity.
            sink.write(f"{k},{step_sec}\n")
            sink.flush()
        elif fmt == "ndjson":
            rec: dict[str, Any] = {"k": k, "step_sec": step_sec}
            if include_diag:
                rec["diag"] = diag
            sink.write(json.dumps(rec) + "\n")
            sink.flush()
        else:
            raise ValueError(f"Unsupported fmt: {fmt}")

    return hook
