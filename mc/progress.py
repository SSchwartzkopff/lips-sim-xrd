"""
mc.progress: optional progress reporting for long MC runs.

The progress bar is implemented directly inside `_numba_cycle` via numba's
`objmode`: every `display_every` MC steps the loop drops to Python and
redraws a "[████-----] xx.xx% | Elapsed | ETA" line.  Because it's a single
`cycle()` call, the simulation trajectory is **bit-identical to a silent
run with the same seed**; the only difference is the on-screen display.

This module provides three small helpers:

    cycle_with_progress(MC, steps, n_updates=200)
        Run `steps` MC attempts with a progress bar updated `n_updates`
        times.  Chooses `display_every = steps // n_updates` automatically.

    cycle_timed(MC, steps)
        Single silent `MC.cycle(steps)` with start/end timing only.
        Use when you don't want any display overhead.

    pick_display_every(steps, n_updates=200)
        Just returns the integer `steps // n_updates` (with a minimum of 1).
        Useful if you want to call `MC.cycle(steps, display_every=...)`
        directly.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .engine import MCPhaseSeparation


def pick_display_every(total_steps: int, n_updates: int = 200) -> int:
    """Compute `display_every` such that the bar refreshes ~`n_updates` times."""
    if total_steps <= 0 or n_updates <= 0:
        return 0
    return max(1, total_steps // n_updates)


def cycle_with_progress(MC: MCPhaseSeparation, total_steps: int,
                        n_updates: int = 200,
                        bar_length: int = 40,
                        label: str | None = None) -> float:
    """Run `total_steps` MC attempts with an inline progress bar.

    Parameters
    ----------
    MC : MCPhaseSeparation
    total_steps : int
    n_updates : int
        Target number of progress redraws over the whole run.  The actual
        redraw interval is `total_steps // n_updates`.
    bar_length : int
    label : str | None
        Printed once before the bar starts.

    Returns
    -------
    elapsed : float
        Wall-clock seconds (also reported on the final bar).
    """
    if total_steps <= 0:
        return 0.0
    de = pick_display_every(total_steps, n_updates)
    if label:
        print(f"  -> {label}: {total_steps:,} MC steps (bar update every {de:,})")
    t0 = time.perf_counter()
    MC.cycle(total_steps, display_every=de, bar_length=bar_length)
    return time.perf_counter() - t0


def cycle_timed(MC: MCPhaseSeparation, total_steps: int,
                label: str = "MC") -> float:
    """Single silent `MC.cycle(total_steps)` with start/end timing only."""
    if total_steps <= 0:
        return 0.0
    print(f"{label}: running {total_steps:,} steps …")
    t0 = time.perf_counter()
    MC.cycle(total_steps, display_every=0)
    elapsed = time.perf_counter() - t0
    rate = total_steps / max(elapsed, 1e-9) / 1e6
    print(f"{label}: {total_steps:,} steps done in {elapsed:.1f}s ({rate:.1f} Msteps/s)")
    return elapsed
