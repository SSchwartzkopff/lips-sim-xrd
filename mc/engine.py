"""
mc.engine: Monte-Carlo phase-separation simulation core.

Public API
----------
    MCPhaseSeparation    : main simulation class
    SimulationParameters : @dataclass holding a full parameter set

Conventions
-----------
* `J` is the magnitude of the mixing-energy coupling, used in the formula
      F = + J * (#like_neighbours) - h_polaron
  i.e. positive J favours phase separation.  This is the sign convention
  used in the paper's parameter table.
* `h = (c_e, c_h)` are the species-specific polaron-field coupling
  constants.  Their effective signs are determined together with
  `electrons_at_bromide` / `holes_at_bromide`.
* `hole_gradient` / `electron_gradient` are decay rates per lattice site
  along the height axis (grid_shape[0]).
* Lattice convention: value 0 = bromide, value 1 = iodide.
"""

import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt
from numba import njit, objmode
from scipy.ndimage import gaussian_filter


# Atom species IDs, fixed at the two-species iodide/bromide model.
BROMIDE = 0
IODIDE = 1


# ─────────────────────────────────────────────────────────────────────────────
# Parameter container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationParameters:
    """Structured container for one simulation's parameters.

    All defaults are the paper-exact values (see the paper's parameter table).
    `SimulationParameters()` with no kwargs reproduces the published run.
    """
    # Lattice
    grid_shape:                  Tuple[int, int] = (706, 500)
    iodide_fraction:             float = 0.4
    # Concentration & polaron length scales
    concentration_sigma:         float = 6.0
    electron_polaron_sigma:      float = 4.0
    hole_polaron_sigma:          float = 3.0
    # Energetics
    J:                           float = 8.0          # positive = phase-separating
    h:                           Tuple[float, float] = (75.0, 20.0)  # (c_e, c_h)
    T:                           float = 3.0
    e_barrier:                   float = 18.0
    # Neighbourhood
    interaction_range:           int  = 2
    allow_diagonal_interactions: bool = True
    # Random seed
    seed:                        int  = 10
    # Three-stage timing & polaron counts (equilibration / light-on / light-off)
    equilibration_hole_num:      int = 100
    equilibration_electron_num:  int = 100
    equilibration_num_steps:     int = 2_500_000_000
    light_on_hole_num:           int = 5_000
    light_on_electron_num:       int = 5_000
    light_on_num_steps:          int = 5_000_000_000
    light_off_hole_num:          int = 100
    light_off_electron_num:      int = 100
    light_off_num_steps:         int = 5_000_000_000
    # Sampling
    polaron_update_every:        int = 25_000
    save_state_every:            int = 10_000_000
    # Gradients
    light_on_hole_gradient:      float = 5e-3
    light_on_electron_gradient:  float = 1e-3
    light_off_hole_gradient:     float = 0.0
    light_off_electron_gradient: float = 0.0
    # Polaron-field generation switches
    normalize_polarons:          bool = False
    max_blur:                    bool = True
    holes_at_bromide:            bool = False     # holes accumulate at iodide
    electrons_at_bromide:        bool = True      # electrons accumulate at bromide

    # ── stage helpers ────────────────────────────────────────────────────────

    _STAGES = ("equilibration", "light_on", "light_off")

    def num_steps_for_stage(self, stage: str) -> int:
        """MC step count configured for the named stage."""
        if stage == "equilibration":
            return self.equilibration_num_steps
        if stage == "light_on":
            return self.light_on_num_steps
        if stage == "light_off":
            return self.light_off_num_steps
        raise ValueError(f"unknown stage {stage!r}; expected one of {self._STAGES}")

    def configure_for_stage(self, MC: "MCPhaseSeparation", stage: str) -> None:
        """Mutate `MC`'s polaron counts and gradients for the named stage.

        Stages: ``'equilibration'``, ``'light_on'``, ``'light_off'``.
        Equilibration uses zero gradients regardless of the dataclass fields.
        """
        if stage == "equilibration":
            MC.hole_num          = self.equilibration_hole_num
            MC.electron_num      = self.equilibration_electron_num
            MC.hole_gradient     = 0.0
            MC.electron_gradient = 0.0
        elif stage == "light_on":
            MC.hole_num          = self.light_on_hole_num
            MC.electron_num      = self.light_on_electron_num
            MC.hole_gradient     = self.light_on_hole_gradient
            MC.electron_gradient = self.light_on_electron_gradient
        elif stage == "light_off":
            MC.hole_num          = self.light_off_hole_num
            MC.electron_num      = self.light_off_electron_num
            MC.hole_gradient     = self.light_off_hole_gradient
            MC.electron_gradient = self.light_off_electron_gradient
        else:
            raise ValueError(f"unknown stage {stage!r}; expected one of {self._STAGES}")


# ─────────────────────────────────────────────────────────────────────────────
# Numba-JIT max-blur helpers  (used by _generate_polaron_fields when max_blur=True)
# ─────────────────────────────────────────────────────────────────────────────

@njit()
def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Centred 2-D Gaussian kernel, peak normalised to 1."""
    x = np.arange(-(size // 2), size // 2 + 1)
    y = x[:, np.newaxis]
    kernel = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    return kernel / kernel.max()


@njit()
def _max_blur_periodic(grid: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Max-aggregated Gaussian blur with periodic boundaries."""
    rows, cols = grid.shape
    k_size = kernel.shape[0]
    pad = k_size // 2
    result = np.zeros_like(grid, dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] > 0:
                for ki in range(k_size):
                    for kj in range(k_size):
                        gi = (i - pad + ki) % rows
                        gj = (j - pad + kj) % cols
                        result[gi, gj] = max(result[gi, gj], kernel[ki, kj])
    return result


@njit()
def _max_blur(grid: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Max-aggregated Gaussian blur with hard (non-periodic) boundaries."""
    rows, cols = grid.shape
    k_size = kernel.shape[0]
    pad = k_size // 2
    result = np.zeros_like(grid, dtype=np.float64)
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] > 0:
                for ki in range(k_size):
                    for kj in range(k_size):
                        gi = i - pad + ki
                        gj = j - pad + kj
                        if 0 <= gi < rows and 0 <= gj < cols:
                            result[gi, gj] = max(result[gi, gj], kernel[ki, kj])
    return result


def _apply_max_gaussian_blur(grid: npt.NDArray[np.float64],
                             sigma: float,
                             periodic: bool = False,
                             kernel_size_mul: int = 6) -> npt.NDArray[np.float64]:
    """Apply a max-aggregated Gaussian blur to a binary point map.

    Each non-zero point in `grid` contributes a Gaussian bump (peak 1); at
    every output pixel the **maximum** contribution is kept rather than the
    sum.  This avoids polarons inflating each other's amplitudes when they
    overlap.
    """
    kernel_size = int(kernel_size_mul * sigma) | 1   # always odd
    kernel = _gaussian_kernel(kernel_size, sigma)
    if periodic:
        return _max_blur_periodic(grid, kernel)
    return _max_blur(grid, kernel)


# ─────────────────────────────────────────────────────────────────────────────
# Polaron-field generator  (called from Python *and* via objmode from Numba)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_polaron_fields(hole_num, electron_num, grid_shape,
                             electron_polaron_sigma, hole_polaron_sigma,
                             concentration_sigma,
                             lattice,
                             hole_gradient=0.0, electron_gradient=0.0,
                             holes_at_bromide=True, electrons_at_bromide=False,
                             normalize=False, max_blur=True):
    """Generate (hole_field, electron_field) on the current lattice.

    The lattice is first smeared into a concentration field `concentration`
    with standard deviation `concentration_sigma`.  Each polaron is then placed
    independently at a site with probability proportional to

         p(i, j)  ∝  [concentration or 1 − concentration]
                     · exp(−|y − h/2| · gradient)

    where the second factor introduces a linear-scale decay along the height
    axis (controlled by `hole_gradient` / `electron_gradient`).  Polaron
    fields are obtained by either max-aggregated or summed Gaussian
    convolution (`max_blur`), optionally clipped to [0, 1] (`normalize`).
    """
    grid_size = lattice.size
    concentration = gaussian_filter(lattice.astype(np.float64),
                                    concentration_sigma, mode='wrap')

    prob_holes     = (1 - concentration) if holes_at_bromide else concentration.copy()
    prob_electrons = (1 - concentration) if electrons_at_bromide else concentration.copy()

    hole_field     = _place_polaron_field(hole_num, prob_holes,
                                          hole_polaron_sigma, hole_gradient,
                                          grid_shape, grid_size,
                                          normalize, max_blur)
    electron_field = _place_polaron_field(electron_num, prob_electrons,
                                          electron_polaron_sigma, electron_gradient,
                                          grid_shape, grid_size,
                                          normalize, max_blur)
    return hole_field, electron_field


def _place_polaron_field(num_polarons, base_probability,
                         polaron_sigma, gradient,
                         grid_shape, grid_size,
                         normalize, max_blur):
    """Sample `num_polarons` sites and convert to a smoothed density field."""
    if num_polarons >= grid_size:
        return np.ones(grid_shape, dtype=np.float64)
    if num_polarons <= 0:
        return np.zeros(grid_shape, dtype=np.float64)

    envelope = np.exp(-np.abs(np.linspace(-grid_shape[0] // 2,
                                          grid_shape[0] // 2,
                                          grid_shape[0])) * gradient)
    probability = base_probability * envelope[:, None]
    probability = np.clip(probability.flatten(), 0, None)
    prob_sum = probability.sum()
    if prob_sum <= 0:
        return np.zeros(grid_shape, dtype=np.float64)

    idx = np.random.choice(grid_size, num_polarons, replace=False,
                           p=probability / prob_sum)
    polaron_grid = np.zeros(grid_shape, dtype=np.float64)
    polaron_grid[np.unravel_index(idx, grid_shape)] = 1.0

    if max_blur:
        field = _apply_max_gaussian_blur(polaron_grid, polaron_sigma,
                                         periodic=True)
    else:
        field = gaussian_filter(polaron_grid, polaron_sigma,
                                mode='wrap') * (2 * np.pi * polaron_sigma ** 2)
    if normalize:
        field = np.clip(field, 0, 1)
    return field


# ─────────────────────────────────────────────────────────────────────────────
# Inline progress bar  (called via `objmode` from inside _numba_cycle)
# ─────────────────────────────────────────────────────────────────────────────

# Module-global handle: tqdm or _AsciiPbar instance.  Not reentrant: only
# one MC cycle can draw a bar at a time within a single Python process.
_active_pbar = None


class _AsciiPbar:
    """Minimal tqdm-substitute with the same `(update, refresh, close)` surface."""

    def __init__(self, total, bar_length=40):
        self.total      = max(1, total)
        self.bar_length = bar_length
        self.n          = 0
        self.start      = time.time()

    def update(self, delta):
        self.n += delta
        self.refresh()

    def refresh(self):
        elapsed  = time.time() - self.start
        progress = min(1.0, self.n / self.total)
        eta      = (elapsed / progress * (1 - progress)) if (0 < progress < 1) else 0.0
        filled   = int(self.bar_length * progress)
        bar_chars = filled * "█" + "-" * (self.bar_length - filled)
        msg = (f"\r[{bar_chars}] {progress*100:6.2f}% | "
               f"Elapsed: {elapsed:7.1f}s | ETA: {eta:7.1f}s")
        try:
            print(msg, end="", flush=True)
        except UnicodeEncodeError:
            print(msg.replace("█", "#"), end="", flush=True)

    def close(self):
        # Newline so subsequent prints don't land on top of the bar
        print()


def _make_pbar(total):
    """Construct a fresh progress bar: tqdm.auto if installed, else ASCII."""
    try:
        from tqdm.auto import tqdm
        # `unit_scale=True` formats 1_000_000 → "1.00M"; `dynamic_ncols=True`
        # lets the bar follow terminal/notebook width.
        return tqdm(total=total, unit=" steps", unit_scale=True,
                    dynamic_ncols=True, leave=True,
                    mininterval=0.1, maxinterval=2.0)
    except ImportError:
        return _AsciiPbar(total)


def _inline_progress(i, num_steps, start_time, bar_length, final=False):
    """Update or finalise the active progress bar.  Called from numba objmode.

    `start_time` and `bar_length` are kept in the signature so _numba_cycle
    doesn't have to be edited; tqdm tracks its own timing and width, while
    `_AsciiPbar` uses `bar_length` for its hand-drawn bar.
    """
    global _active_pbar

    if _active_pbar is None:
        _active_pbar = _make_pbar(num_steps)

    if final:
        # Push counter to exactly num_steps and close
        delta = num_steps - getattr(_active_pbar, "n", 0)
        if delta > 0:
            _active_pbar.update(delta)
        _active_pbar.close()
        _active_pbar = None
    else:
        # Advance to step (i + 1) so the bar reflects work *completed*
        target = i + 1
        delta  = target - getattr(_active_pbar, "n", 0)
        if delta > 0:
            _active_pbar.update(delta)


# ─────────────────────────────────────────────────────────────────────────────
# Inner MC loop  (numba JIT, drops into Python via `objmode` for polaron updates)
# ─────────────────────────────────────────────────────────────────────────────

@njit()
def _numba_cycle(num_steps,
                 lattice, lattice_storage,
                 hole_field, electron_field,
                 hole_storage, electron_storage,
                 index_offset,
                 J, h, e_barrier, T,
                 swap_offsets, interaction_neighbours, distance_scaling,
                 save_state_every, polaron_update_every,
                 hole_num, electron_num, grid_shape,
                 electron_polaron_sigma, hole_polaron_sigma, concentration_sigma,
                 seed,
                 hole_gradient, electron_gradient,
                 normalize_polarons, max_blur,
                 holes_at_bromide, electrons_at_bromide,
                 display_every=0, bar_length=40):
    """Numba inner Metropolis loop.  The energy of a swap is

           F = + J · (Σ like-neighbour count differences)  −  h-polaron term

    so **positive J favours phase separation** (paper convention).

    Progress display
    ----------------
    `display_every > 0` enables an in-loop progress bar: every `display_every`
    MC steps the loop drops into Python (via `objmode`) and calls
    `_inline_progress` to redraw a "[████-----] xx% | Elapsed | ETA" line.
    `display_every = 0` disables the bar entirely (zero overhead).  The
    simulation trajectory is independent of this setting.
    """
    np.random.seed(seed)
    lattice_shape = lattice.shape

    electron_sign = +1 if electrons_at_bromide else -1
    hole_sign     = +1 if holes_at_bromide     else -1

    # Wall-clock start (used only when display_every > 0).  Unconditional
    # objmode keeps the numba SSA/typing consistent; the cost is one
    # `time.time()` call per cycle() invocation, negligible.
    with objmode(start_time='float64'):
        start_time = time.time()

    for i in range(num_steps):
        # ---- regenerate polaron field (drops to Python) ----------------------
        if i % polaron_update_every == 0:
            with objmode(hole_field='float64[:, :]', electron_field='float64[:, :]'):
                hole_field, electron_field = _generate_polaron_fields(
                    hole_num, electron_num, grid_shape,
                    electron_polaron_sigma, hole_polaron_sigma,
                    concentration_sigma, lattice,
                    hole_gradient, electron_gradient,
                    holes_at_bromide, electrons_at_bromide,
                    normalize_polarons, max_blur)

        # ---- snapshot to storage --------------------------------------------
        if i % save_state_every == 0:
            slot = i // save_state_every + 1 + index_offset
            lattice_storage[slot, ...]  = lattice
            hole_storage[slot, ...]     = hole_field
            electron_storage[slot, ...] = electron_field

        # ---- pick a random pair and decide whether to swap -------------------
        site_a = (np.random.randint(0, lattice_shape[0]),
                  np.random.randint(0, lattice_shape[1]))
        dx, dy = swap_offsets[np.random.randint(0, len(swap_offsets))]
        site_b = ((site_a[0] + dx) % lattice_shape[0],
                  (site_a[1] + dy) % lattice_shape[1])

        species_a = lattice[site_a]
        species_b = lattice[site_b]
        if species_a == species_b:
            continue

        energy_original = 0.0
        energy_swapped  = 0.0
        for j in range(len(interaction_neighbours)):
            nx, ny = interaction_neighbours[j]
            nbr_a = ((site_a[0] + nx) % lattice_shape[0],
                     (site_a[1] + ny) % lattice_shape[1])
            nbr_b = ((site_b[0] + nx) % lattice_shape[0],
                     (site_b[1] + ny) % lattice_shape[1])
            if nbr_a != site_b:
                if lattice[nbr_a] == species_a:
                    energy_original += distance_scaling[j]
                    energy_swapped  -= distance_scaling[j]
                else:
                    energy_original -= distance_scaling[j]
                    energy_swapped  += distance_scaling[j]
            if nbr_b != site_a:
                if lattice[nbr_b] == species_b:
                    energy_original += distance_scaling[j]
                    energy_swapped  -= distance_scaling[j]
                else:
                    energy_original -= distance_scaling[j]
                    energy_swapped  += distance_scaling[j]

        # Polaron-field contribution to the swap energy
        if lattice[site_a] == IODIDE:
            h_original = (electron_sign * (electron_field[site_b] - electron_field[site_a]) * h[0]
                          + hole_sign   * (hole_field[site_b]     - hole_field[site_a])     * h[1])
            h_swapped  = (electron_sign * (electron_field[site_a] - electron_field[site_b]) * h[0]
                          + hole_sign   * (hole_field[site_a]     - hole_field[site_b])     * h[1])
        else:  # bromide
            h_original = (electron_sign * (electron_field[site_a] - electron_field[site_b]) * h[0]
                          + hole_sign   * (hole_field[site_a]     - hole_field[site_b])     * h[1])
            h_swapped  = (electron_sign * (electron_field[site_b] - electron_field[site_a]) * h[0]
                          + hole_sign   * (hole_field[site_b]     - hole_field[site_a])     * h[1])

        energy_original = J * energy_original - h_original
        energy_swapped  = J * energy_swapped  - h_swapped
        dF = energy_swapped - energy_original

        if np.random.random() < np.exp(-(dF + e_barrier) / T):
            lattice[site_a], lattice[site_b] = lattice[site_b], lattice[site_a]

        # Progress display (drops to Python; negligible cost at display_every >> 1)
        if display_every > 0 and i % display_every == 0:
            with objmode():
                _inline_progress(i, num_steps, start_time, bar_length, final=False)

    # Final 100% line
    if display_every > 0:
        with objmode():
            _inline_progress(num_steps, num_steps, start_time, bar_length, final=True)

    return (lattice, lattice_storage,
            hole_field, hole_storage,
            electron_field, electron_storage)


# ─────────────────────────────────────────────────────────────────────────────
# High-level simulation class
# ─────────────────────────────────────────────────────────────────────────────

class MCPhaseSeparation:
    """High-level wrapper around `_numba_cycle`.

    Construct with the desired parameters, then call `.cycle(num_steps)` one
    or more times.  Between cycles you can mutate `hole_num`, `electron_num`,
    `hole_gradient`, `electron_gradient`, etc. to model multi-stage protocols
    (e.g. light-on / light-off).

    The accumulated snapshots are exposed as:
        self.lattice_storage   (frames, Y, X)  bool   atom configuration
        self.hole_storage      (frames, Y, X)  f64    hole polaron density
        self.electron_storage  (frames, Y, X)  f64    electron polaron density

    Lattice values: 0 = bromide, 1 = iodide.  Positive J favours phase
    separation (paper convention).
    """

    def __init__(self,
                 # Lattice geometry & composition
                 grid_shape=(706, 500), iodide_fraction=0.4,
                 # Energetics (paper Table 1)
                 J=8.0, h=(75.0, 20.0), T=3.0, e_barrier=18.0,
                 # Neighbourhood (paper: d_ij ≤ 2a, diagonals included)
                 interaction_range=2, allow_diagonal_interactions=True,
                 # Polaron length scales (paper Table 1: σ_h=3, σ_e=4)
                 hole_polaron_sigma=3.0, electron_polaron_sigma=4.0,
                 concentration_sigma=6.0,
                 # Single-cycle defaults match the paper's *equilibration* stage:
                 # low polaron count (N_off), no gradients.  Override these between
                 # cycles, or use `MCPhaseSeparation.from_params(...)` to drive the
                 # full three-stage protocol from a `SimulationParameters`.
                 hole_num=100, electron_num=100,
                 hole_gradient=0.0, electron_gradient=0.0,
                 num_steps=2_500_000_000,
                 save_state_every=10_000_000, polaron_update_every=25_000,
                 # Polaron-field generation switches
                 normalize_polarons=False, max_blur=True,
                 holes_at_bromide=False, electrons_at_bromide=True,
                 # RNG
                 seed=10):
        # Numba's RNG requires an integer seed; if the caller passes None we
        # draw one from OS entropy so behaviour stays reproducible after the
        # constructor returns (`self.original_seed` records the value used).
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0] & 0x7FFFFFFF)
        np.random.seed(seed)
        self.original_seed = seed
        self.current_seed  = seed

        self.grid_shape           = grid_shape
        self.grid_size            = int(np.prod(self.grid_shape))
        self.num_steps            = num_steps
        self.save_state_every     = save_state_every
        self.index_offset         = 0

        self.normalize_polarons   = normalize_polarons
        self.max_blur             = max_blur
        self.holes_at_bromide     = holes_at_bromide
        self.electrons_at_bromide = electrons_at_bromide

        self.hole_gradient        = hole_gradient
        self.electron_gradient    = electron_gradient

        self.num_cycles           = num_steps // save_state_every

        self.polaron_update_every = polaron_update_every

        self.J                    = J
        self.h                    = h
        self.T                    = T
        self.e_barrier            = e_barrier

        self.iodide_fraction      = iodide_fraction
        self.hole_num             = hole_num
        self.electron_num         = electron_num
        self.electron_polaron_sigma = electron_polaron_sigma
        self.hole_polaron_sigma   = hole_polaron_sigma
        self.concentration_sigma  = concentration_sigma

        self.interaction_range    = interaction_range
        self.allow_diagonal_interactions = allow_diagonal_interactions

        # Kawasaki exchange-partner offsets (nearest-neighbour swaps).
        self.swap_offsets = ((0, 1), (-1, 0), (1, 0), (0, -1))

        self._build_interaction_stencil()
        self._compute_distance_scaling()

        self._create_lattice()
        self.lattice_storage = np.zeros((self.num_cycles + 1, *self.grid_shape),
                                        dtype=bool)
        self.lattice_storage[0, :, :] = self.lattice

        self._generate_polarons()
        self.hole_storage = np.zeros((self.num_cycles + 1, *self.grid_shape),
                                     dtype=np.float64)
        self.hole_storage[0, :, :] = self.hole_field
        self.electron_storage = np.zeros((self.num_cycles + 1, *self.grid_shape),
                                         dtype=np.float64)
        self.electron_storage[0, :, :] = self.electron_field

    @classmethod
    def from_params(cls, params: SimulationParameters) -> "MCPhaseSeparation":
        """Construct an MC instance initialised for the equilibration stage.

        Stage-specific fields (polaron counts, gradients, step counts) are
        applied from the equilibration block of `params`.  To advance to a
        later stage, call ``params.configure_for_stage(MC, stage)`` and then
        ``MC.cycle(params.num_steps_for_stage(stage))``.
        """
        return cls(
            grid_shape=params.grid_shape,
            J=params.J, h=params.h, T=params.T, e_barrier=params.e_barrier,
            seed=params.seed,
            iodide_fraction=params.iodide_fraction,
            interaction_range=params.interaction_range,
            allow_diagonal_interactions=params.allow_diagonal_interactions,
            hole_num=params.equilibration_hole_num,
            electron_num=params.equilibration_electron_num,
            electron_polaron_sigma=params.electron_polaron_sigma,
            hole_polaron_sigma=params.hole_polaron_sigma,
            concentration_sigma=params.concentration_sigma,
            num_steps=params.equilibration_num_steps,
            save_state_every=params.save_state_every,
            polaron_update_every=params.polaron_update_every,
            hole_gradient=0.0, electron_gradient=0.0,
            normalize_polarons=params.normalize_polarons,
            max_blur=params.max_blur,
            holes_at_bromide=params.holes_at_bromide,
            electrons_at_bromide=params.electrons_at_bromide,
        )

    # ── neighbourhood & jump-energy weights ──────────────────────────────────

    def _build_interaction_stencil(self):
        """Offsets of the sites that contribute to the swap-energy sum."""
        if self.allow_diagonal_interactions:
            self.interaction_neighbours = tuple(
                (i, j)
                for i in range(-self.interaction_range, self.interaction_range + 1)
                for j in range(-self.interaction_range, self.interaction_range + 1)
                if i != 0 or j != 0
            )
        else:
            base = ((0, 1), (-1, 0), (1, 0), (0, -1))
            self.interaction_neighbours = tuple(
                (n[0] * (i + 1), n[1] * (i + 1))
                for n in base for i in range(self.interaction_range)
            )

    def _compute_distance_scaling(self):
        distances = np.array([1.0 / (dx * dx + dy * dy)
                              for dx, dy in self.interaction_neighbours])
        self.distance_scaling = distances / distances.sum()

    # ── lattice / polaron initialisation ─────────────────────────────────────

    def _create_lattice(self):
        """Create a random initial atom configuration.

        Each site is independently assigned IODIDE with probability
        `iodide_fraction`, otherwise BROMIDE.
        """
        rand = np.random.random(self.grid_shape)
        self.lattice = (rand < self.iodide_fraction).astype(bool)

    def _generate_polarons(self):
        self.hole_field, self.electron_field = _generate_polaron_fields(
            self.hole_num, self.electron_num, self.grid_shape,
            self.electron_polaron_sigma, self.hole_polaron_sigma,
            self.concentration_sigma,
            self.lattice,
            hole_gradient=self.hole_gradient,
            electron_gradient=self.electron_gradient,
            holes_at_bromide=self.holes_at_bromide,
            electrons_at_bromide=self.electrons_at_bromide,
            normalize=self.normalize_polarons,
            max_blur=self.max_blur)

    # ── storage management ──────────────────────────────────────────────────

    def _resize_storage_for_steps(self, new_num_steps):
        """Resize storage to accommodate a new cycle (extending if already populated).

        Increments `current_seed` so the next cycle uses a different RNG stream
        from the previous one.
        """
        self.current_seed += 1
        np.random.seed(self.current_seed)
        self.num_steps  = new_num_steps
        self.num_cycles = new_num_steps // self.save_state_every

        # Decide whether the previous storage was actually filled.
        try:
            already_run = np.sum(self.lattice_storage[1, ...]) != 0
        except Exception:
            already_run = False

        if not already_run:
            # Re-init from scratch
            self.lattice_storage = np.zeros(
                (self.num_cycles + 1, *self.grid_shape), dtype=bool)
            self.lattice_storage[0, :, :] = self.lattice
            self.hole_storage = np.zeros(
                (self.num_cycles + 1, *self.grid_shape), dtype=np.float64)
            self.hole_storage[0, :, :] = self.hole_field
            self.electron_storage = np.zeros(
                (self.num_cycles + 1, *self.grid_shape), dtype=np.float64)
            self.electron_storage[0, :, :] = self.electron_field
        else:
            # Extend
            self.index_offset = self.lattice_storage.shape[0] - 1
            self.lattice_storage = np.concatenate(
                (self.lattice_storage,
                 np.zeros((self.num_cycles, *self.grid_shape), dtype=bool)),
                axis=0)
            self.hole_storage = np.concatenate(
                (self.hole_storage,
                 np.zeros((self.num_cycles, *self.grid_shape), dtype=np.float64)),
                axis=0)
            self.electron_storage = np.concatenate(
                (self.electron_storage,
                 np.zeros((self.num_cycles, *self.grid_shape), dtype=np.float64)),
                axis=0)

    # ── main entry point ────────────────────────────────────────────────────

    def cycle(self, num_steps: int | None = None,
              display_every: int = 0, bar_length: int = 40) -> None:
        """Run `num_steps` Metropolis attempts and accumulate snapshots.

        Parameters
        ----------
        num_steps : int | None
            Number of MC attempts.  None → reuse the previously set
            `self.num_steps`.
        display_every : int
            If > 0, an in-loop progress bar is drawn every `display_every`
            MC steps.  0 disables it.  The simulation trajectory is
            independent of this setting; it controls display only.
        bar_length : int
            Width of the progress bar in characters.
        """
        if num_steps is not None:
            self._resize_storage_for_steps(num_steps)
        (self.lattice, self.lattice_storage,
         self.hole_field, self.hole_storage,
         self.electron_field, self.electron_storage) = _numba_cycle(
            self.num_steps,
            self.lattice, self.lattice_storage,
            self.hole_field, self.electron_field,
            self.hole_storage, self.electron_storage,
            self.index_offset,
            self.J, self.h, self.e_barrier, self.T,
            self.swap_offsets, self.interaction_neighbours, self.distance_scaling,
            self.save_state_every, self.polaron_update_every,
            self.hole_num, self.electron_num, self.grid_shape,
            self.electron_polaron_sigma, self.hole_polaron_sigma,
            self.concentration_sigma,
            self.current_seed,
            self.hole_gradient, self.electron_gradient,
            self.normalize_polarons, self.max_blur,
            self.holes_at_bromide, self.electrons_at_bromide,
            display_every, bar_length)
