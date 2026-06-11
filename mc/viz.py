"""
mc.viz: lightweight matplotlib helpers for inspecting simulation output.

Public API
----------
    plot_qt(qt, ...)                 : heatmap of the q-vs-time intensity matrix
    plot_lattice_snapshot(MC, frame) : single (lattice | hole | electron) panel
    plot_lattice_grid(MC, frames)    : grid of snapshots across multiple frames
    create_animation(arrays_dict)    : multi-panel HTML5 animation for Jupyter

This module requires only `matplotlib`.  `create_animation` additionally
needs `IPython` (already a dependency of any Jupyter notebook).
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# ─────────────────────────────────────────────────────────────────────────────
# Internal helper
# ─────────────────────────────────────────────────────────────────────────────

def _panel_size(H, W, target_h=5.0, min_w=0.8, max_w=5.0):
    """Return (width_in, height_in) for one image panel, preserving the H:W
    data aspect ratio.  `target_h` sets the panel height; the width is derived
    from the aspect ratio and clamped to [min_w, max_w] so extreme grids (very
    narrow or very wide) stay within a usable figure size."""
    pw = float(np.clip(target_h * W / H, min_w, max_w))
    return pw, float(target_h)


# ─────────────────────────────────────────────────────────────────────────────
# Public functions
# ─────────────────────────────────────────────────────────────────────────────

def plot_qt(qt,
            q_range=(0.98, 1.07),
            extent_max=None,
            vmin=5.0, vmax=None,
            cmap="jet",
            ax=None,
            cbar=True,
            title=None):
    """Plot a `(frames, q_bins)` intensity matrix as a log-norm heatmap.

    Parameters
    ----------
    qt : np.ndarray, shape (n_frames, n_q_bins)
    q_range : (float, float)
        Physical extent of the q-axis (Å⁻¹).
    extent_max : float | None
        Physical extent of the time axis (e.g. total MC steps).  Defaults
        to `qt.shape[0]` if not provided.
    vmin, vmax : float | None
        LogNorm bounds.  `None` for vmax → use the 99th-percentile of the data.
    cmap : str
    ax : matplotlib Axes | None
    cbar : bool
        Whether to draw a colourbar.
    title : str | None

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.2))
    else:
        fig = ax.figure
    if extent_max is None:
        extent_max = qt.shape[0]
    if vmax is None:
        vmax = float(np.percentile(qt, 99))

    # Map both `bad` (NaN / masked) and `under` (< vmin) to the colormap's
    # minimum colour, so log-scale dropouts don't show as white holes.
    _cmap = mpl.colormaps[cmap].copy() if isinstance(cmap, str) else cmap.copy()
    _cmap.set_bad(_cmap(0.0))
    _cmap.set_under(_cmap(0.0))

    im = ax.imshow(qt,
                   extent=[q_range[0], q_range[1], extent_max, 0],
                   cmap=_cmap, norm=LogNorm(vmin=vmin, vmax=vmax),
                   interpolation="none", aspect="auto", origin="upper")
    ax.set_xlabel(r"$q$ (Å$^{-1}$)")
    ax.set_ylabel("MC steps")
    if title:
        ax.set_title(title)
    if cbar:
        fig.colorbar(im, ax=ax, label="Intensity (arb. u.)")
    fig.tight_layout()
    return fig, ax


def plot_lattice_snapshot(MC, frame=-1,
                          figsize=None,
                          lattice_cmap="viridis",
                          pol_cmap="plasma"):
    """Three-panel view of one snapshot: lattice | hole field | electron field.

    `frame=-1` selects the last saved snapshot.  `figsize` is computed
    automatically from the grid's H:W aspect ratio if not supplied, so the
    image is never distorted regardless of grid shape.
    """
    n = MC.lattice_storage.shape[0]
    if frame < 0:
        frame = n + frame
    if not 0 <= frame < n:
        raise IndexError(f"frame {frame} out of range [0, {n})")

    lat = MC.lattice_storage[frame]
    hp  = MC.hole_storage[frame]
    ep  = MC.electron_storage[frame]
    H, W = lat.shape

    if figsize is None:
        pw, ph = _panel_size(H, W, target_h=5.0, min_w=0.8, max_w=5.0)
        figsize = (3 * pw + 2.0, ph)   # 2.0" budget for three colorbars

    fig, axs = plt.subplots(1, 3, figsize=figsize)
    titles = ("Lattice", "Hole polarons", "Electron polarons")
    cmaps  = (lattice_cmap, pol_cmap, pol_cmap)
    for ax, data, t, c in zip(axs, (lat, hp, ep), titles, cmaps):
        im = ax.imshow(data, cmap=c, interpolation="none", aspect="equal")
        ax.set_title(t)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, axs


def plot_lattice_grid(MC, frames=None,
                      n_cols=4,
                      figsize=None,
                      channel="lattice",
                      cmap=None):
    """Grid of snapshots from one channel ('lattice' | 'holes' | 'electrons').

    `figsize` is computed automatically from the grid's H:W aspect ratio if
    not supplied.
    """
    channel_map = {
        "lattice":   (MC.lattice_storage,  "viridis"),
        "holes":     (MC.hole_storage,     "plasma"),
        "electrons": (MC.electron_storage, "plasma"),
    }
    if channel not in channel_map:
        raise ValueError(f"channel must be one of {list(channel_map)}, got {channel!r}")
    stack, default_cmap = channel_map[channel]
    cmap = cmap or default_cmap

    n_frames_total = stack.shape[0]
    if frames is None:
        frames = np.linspace(0, n_frames_total - 1, 12, dtype=int)

    n = len(frames)
    n_rows = (n + n_cols - 1) // n_cols
    H, W = stack.shape[1], stack.shape[2]

    if figsize is None:
        pw, ph = _panel_size(H, W, target_h=2.5, min_w=0.6, max_w=3.0)
        figsize = (n_cols * pw + 0.5, n_rows * ph + 0.8)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    for k, f in enumerate(frames):
        ax = axes[k // n_cols][k % n_cols]
        ax.imshow(stack[f], cmap=cmap, interpolation="none", aspect="equal")
        ax.set_title(f"frame {f}")
        ax.set_xticks([]); ax.set_yticks([])
    for k in range(n, n_rows * n_cols):
        axes[k // n_cols][k % n_cols].axis("off")
    fig.suptitle(f"{channel} (every {n_frames_total // max(n, 1)}th frame)")
    fig.tight_layout()
    return fig, axes


def create_animation(arrays_dict, max_frames=100, fps=10,
                     figsize=None, cmaps=None, title=None,
                     vmin=None, vmax=None,
                     step_per_frame=None):
    """Create a multi-panel HTML5 animation suitable for Jupyter display.

    Renders one panel per array, side by side, with synchronised playback.
    Returns an `IPython.display.HTML` object; assign it to the last
    expression of a cell (or pass it to `display()`) to embed in a notebook.

    Parameters
    ----------
    arrays_dict : dict[str, np.ndarray]
        Mapping ``{panel_title: array(frames, Y, X)}``.  All arrays must
        share the same first (time) dimension.
    max_frames : int
        Maximum number of frames in the output.  Inputs with more frames
        are downsampled at evenly spaced indices.
    fps : int
        Playback rate in frames per second.
    figsize : tuple | None
        Matplotlib figure size.  Default is derived from the grid's H:W
        aspect ratio so the lattice is never distorted.
    cmaps : list[str] | str | None
        Per-panel colormaps.  Default: `viridis` for the first panel
        (lattice), `plasma` for the rest (polaron densities).
    title : str | None
        Static part of the figure title.  The dynamic per-frame counter
        is appended to it in the suptitle (so it never overlaps a panel).
    vmin, vmax : list | scalar | None
        Per-panel colour-scale limits.  None → use each array's min/max.
    step_per_frame : int | None
        If given (typically `MC.save_state_every`), the per-frame counter
        in the suptitle shows the *MC step number* of each frame, e.g.
        "step 20,000,000".  None → falls back to "frame K".

    Returns
    -------
    IPython.display.HTML
        Embed in a notebook cell as the last expression to display.
    """
    from matplotlib.animation import FuncAnimation
    from IPython.display import HTML

    panels = list(arrays_dict.items())
    n_panels = len(panels)
    if n_panels == 0:
        raise ValueError("arrays_dict is empty")

    n_frames_in = panels[0][1].shape[0]
    if any(arr.shape[0] != n_frames_in for _, arr in panels):
        raise ValueError("All arrays must share the same first (time) dimension")

    # Downsample to max_frames
    if max_frames is not None and n_frames_in > max_frames:
        idx = np.linspace(0, n_frames_in - 1, max_frames, dtype=int)
        panels = [(name, arr[idx]) for name, arr in panels]
        n_frames = max_frames
    else:
        idx = np.arange(n_frames_in)
        n_frames = n_frames_in

    if cmaps is None:
        cmaps = ['viridis'] + ['plasma'] * (n_panels - 1)
    if isinstance(cmaps, str):
        cmaps = [cmaps] * n_panels

    if figsize is None:
        H, W = panels[0][1].shape[1], panels[0][1].shape[2]
        pw, ph = _panel_size(H, W, target_h=4.0, min_w=1.5, max_w=5.0)
        figsize = (n_panels * pw + 0.6 * n_panels, ph + 0.5)

    # Per-panel colour bounds
    def _bounds(b, default_fn):
        if b is None:
            return [default_fn(arr) for _, arr in panels]
        if np.isscalar(b):
            return [b] * n_panels
        return list(b)
    vmins = _bounds(vmin, lambda a: float(a.min()))
    vmaxs = _bounds(vmax, lambda a: float(a.max()))

    fig, axs = plt.subplots(1, n_panels, figsize=figsize, constrained_layout=True)
    if n_panels == 1:
        axs = [axs]

    ims = []
    for ax, (name, arr), cmap, lo, hi in zip(axs, panels, cmaps, vmins, vmaxs):
        im = ax.imshow(arr[0], cmap=cmap, interpolation='none',
                       vmin=lo, vmax=hi, aspect='equal')
        ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ims.append(im)

    # Dynamic suptitle: "title - step 1,234,567" (or "frame K/N" when
    # step_per_frame is not provided).  Suptitle has its own area above
    # the panels so it never overlaps the data.
    def _frame_label(k):
        if step_per_frame is not None:
            return f"step {int(idx[k]) * int(step_per_frame):,}"
        return f"frame {int(idx[k])} / {n_frames_in - 1}"

    title_parts = [title] if title else []
    suptitle = fig.suptitle(" - ".join(title_parts + [_frame_label(0)]))

    def update(k):
        for im, (_, arr) in zip(ims, panels):
            im.set_array(arr[k])
        suptitle.set_text(" - ".join(title_parts + [_frame_label(k)]))
        return [*ims, suptitle]

    # blit=False because the suptitle isn't a child of any axes; with blit
    # enabled matplotlib won't re-render it on each frame.
    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=1000 / fps, blit=False)
    html = anim.to_jshtml()
    plt.close(fig)        # prevent the still-frame from also appearing
    return HTML(html)
