"""
mc: Monte-Carlo phase-separation simulation accompanying the paper:

    Investigating Light-Induced Phase Separation in MAPbBr<sub>1.8</sub>I<sub>1.2</sub> Using Time-Resolved X-ray Diffraction and Numerical Simulations
    Sebastian Schwartzkopff, Ivan Zaluzhnyy, Ekaterina Kneschaurek, Dmitry Lapkin, Hans Mauser, Niels Scheffczyk, Paul Zimmermann, Alexander Hinderhofer, Frederik Unger, Fabian Westermeier, Yana Vaynzof, Fabian Paulus and Frank Schreiber
    JOURNAL

Public modules
--------------
    mc.engine      - simulation core (MCPhaseSeparation, SimulationParameters)
    mc.postprocess - q-space conversion (concentration_to_qt, voigt_blur_1d)
    mc.progress    - progress-bar wrappers for long MC runs (optional)
    mc.viz         - matplotlib plotting helpers (optional)

Minimal install
---------------
The `engine` and `postprocess` modules only need:
    numpy, scipy, numba, fast_histogram (optional speedup)
Adding `matplotlib` enables `mc.viz`.

A complete, reproducible example is in `example.ipynb` at the repo root.

Sign conventions
----------------
* J > 0  favours phase separation
* See `mc.engine._numba_cycle` for the full energy expression.

Reproducibility
---------------
Set `SimulationParameters(seed=...)` for deterministic results.  Numba's
inner-loop RNG is seeded separately inside `_numba_cycle` from the same
`seed`, so two runs with identical parameters produce bit-identical qt
arrays.  Tested with numba >= 0.58.
"""

from .engine import MCPhaseSeparation, SimulationParameters
from .postprocess import concentration_to_qt, voigt_blur_1d

__all__ = [
    "MCPhaseSeparation",
    "SimulationParameters",
    "concentration_to_qt",
    "voigt_blur_1d",
]

__version__ = "1.0.0"
