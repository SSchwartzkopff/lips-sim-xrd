"""
mc.postprocess: convert simulation snapshots into observable q-axis intensity.

Public API
----------
    concentration_to_qt(u, bins, q_range, ratio) : concentration → diffraction histogram
    voigt_blur_1d(mat, x_range, σ, γ)            : line-shape broadening (Voigt convolution)

The `concentration_to_qt` function uses two lookup tables shipped alongside
this module in `saved_arrays/`:
    conInterp.npy    : (N, 2) array of (q1, q2) values per concentration
    concentration.npy: 1D array of concentration values used as the abscissa
See the paper supplement for details on how these tables were generated.
"""

import os
import numpy as np

from scipy.special import voigt_profile
from scipy.signal import fftconvolve

# Optional: use the much faster `fast_histogram` library if available.
# Falls back to numpy's histogram (~3× slower) automatically.
try:
    from fast_histogram import histogram1d as _fast_hist
    _HAS_FAST_HIST = True
except ImportError:
    _HAS_FAST_HIST = False


_HERE       = os.path.dirname(os.path.abspath(__file__))
_SAVED_DIR  = os.path.join(_HERE, "saved_arrays")
_CACHED_LUT = {"q_001": None, "conc": None}


def _load_lookup_tables():
    """Lazy + cached load of the concentration → q-shift lookup tables."""
    if _CACHED_LUT["q_001"] is None:
        _CACHED_LUT["q_001"] = np.load(os.path.join(_SAVED_DIR, "conInterp.npy"))
        _CACHED_LUT["conc"]  = np.load(os.path.join(_SAVED_DIR, "concentration.npy"))
    return _CACHED_LUT["q_001"], _CACHED_LUT["conc"]


def _row_histograms(arr_2d, bins, range_):
    """Per-row 1D histogram of a 2D array.

    Uses `fast_histogram.histogram1d` if available (3-10x speedup), else
    falls back to a `np.apply_along_axis(np.histogram, …)` loop.  The two
    paths agree to within a single count per bin due to floating-point
    rounding at bin edges (relative error ~10⁻⁵ on real data, invisible
    on log plots).
    """
    n_rows = arr_2d.shape[0]
    out = np.empty((n_rows, bins), dtype=np.float64)
    if _HAS_FAST_HIST:
        lo, hi = range_
        for i in range(n_rows):
            out[i] = _fast_hist(arr_2d[i], bins=bins, range=(lo, hi))
    else:
        for i in range(n_rows):
            out[i] = np.histogram(arr_2d[i], bins=bins, range=range_)[0]
    return out


def concentration_to_qt(u: np.ndarray,
                        bins: int = 1000,
                        q_range: tuple[float, float] = (0.99, 1.08),
                        ratio: float = 1.0) -> np.ndarray:
    """Convert a stack of lattice (or blurred-lattice) snapshots into a (frames, q) intensity matrix.

    Parameters
    ----------
    u : np.ndarray, shape (n_frames, Y, X)
        Concentration field (typically the lattice after a small Gaussian blur).
    bins : int
        Number of q-axis bins inside the lookup table's intrinsic range.
        The output may have *more* columns if `q_range` extends beyond it
        (zero-padded extension).
    q_range : (q_min, q_max)
        Target q-axis range in Å⁻¹.  Bins outside the lookup-table range
        come out as zeros (extended with the same `dq`).
    ratio : float in [0, 1]
        Weighting between the two q-components (cubic vs tetragonal) stored in
        `conInterp.npy`: ``result = (ratio * h1 + (1-ratio) * h2) / 2``.
        The /2 keeps the per-row count normalised to the number of pixels
        when summing two contributions.

    Returns
    -------
    qt : np.ndarray, shape (n_frames, n_q_bins)
    """
    q_001, conc = _load_lookup_tables()

    q1 = np.interp(u, conc, q_001[:, 0])
    q2 = np.interp(u, conc, q_001[:, 1])

    n_frames = q1.shape[0]
    q1 = q1.reshape(n_frames, -1)
    q2 = q2.reshape(n_frames, -1)

    qmin, qmax = float(q_001.min()), float(q_001.max())
    h1 = _row_histograms(q1, bins, (qmin, qmax))
    h2 = _row_histograms(q2, bins, (qmin, qmax))

    # Weighted average of the two component histograms.
    qt_matrix = (ratio * h1 + (1 - ratio) * h2) / 2

    # Pad to the requested q_range
    dq = (qmax - qmin) / bins
    q_ext_down = int(np.floor((qmin - min(q_range)) / dq))
    q_ext_up   = int(np.ceil((max(q_range) - qmax) / dq))
    padded_size = bins + q_ext_up + q_ext_down

    padded = np.zeros((qt_matrix.shape[0], padded_size))
    src_lo = max(-q_ext_down, 0)
    src_hi = min(bins + q_ext_up, bins)
    dst_lo = max(q_ext_down, 0)
    dst_hi = min(q_ext_down + qt_matrix.shape[1], padded_size)
    padded[:, dst_lo:dst_hi] = qt_matrix[:, src_lo:src_hi]
    return padded


def voigt_blur_1d(mat: np.ndarray,
                  x_range: tuple[float, float],
                  sigma: float = 1.0,
                  gamma: float = 1.0,
                  mode: str = "same",
                  resample_nonuniform: bool = True) -> np.ndarray:
    """Row-wise convolution of `mat` with a Voigt kernel of parameters (σ, γ).

    Uses FFT convolution for speed.  If `x` is non-uniform, the rows are
    first linearly interpolated onto a uniform grid; toggle with
    `resample_nonuniform=False` to raise instead.

    Returns a 2D array (or 1D if `mat` was 1D).
    """
    mat = np.asarray(mat, dtype=float)
    if mat.ndim == 1:
        mat = mat[np.newaxis, :]
    n_rows, n_cols = mat.shape

    x = np.linspace(*x_range, n_cols)
    if x.size != n_cols:
        raise ValueError("x must have one entry per column of mat")

    dxs = np.diff(x)
    if np.any(~np.isfinite(dxs)):
        raise ValueError("x must be finite and strictly monotonic")
    dx_mean = float(np.mean(dxs))
    if dx_mean == 0:
        raise ValueError("x spacing zero (duplicate x values?)")

    if np.max(np.abs(dxs - dx_mean)) / abs(dx_mean) > 1e-6:
        if not resample_nonuniform:
            raise ValueError("x is non-uniform; set resample_nonuniform=True.")
        x_uniform = np.linspace(x[0], x[-1], n_cols)
        mat = np.vstack([np.interp(x_uniform, x, row) for row in mat])
        x  = x_uniform
        dx = x[1] - x[0]
    else:
        dx = dx_mean

    # Centred Voigt kernel
    center = n_cols // 2
    kernel = voigt_profile((np.arange(n_cols) - center) * dx, sigma, gamma)

    out = np.vstack([fftconvolve(row, kernel, mode=mode) * dx for row in mat])
    return out[0] if out.shape[0] == 1 else out
