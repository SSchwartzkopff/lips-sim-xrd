# Monte-Carlo phase-separation simulation

Companion code for:

> **Investigating Light-Induced Phase Separation in MAPbBr<sub>1.8</sub>I<sub>1.2</sub> Using Time-Resolved X-ray Diffraction and Numerical Simulations**
> *Sebastian Schwartzkopff, Ivan Zaluzhnyy, Ekaterina Kneschaurek, Dmitry Lapkin, Hans Mauser, Niels Scheffczyk, Paul Zimmermann, Alexander Hinderhofer, Frederik Unger, Fabian Westermeier, Yana Vaynzof, Fabian Paulus and Frank Schreiber*
> *JOURNAL*, [YEAR]. DOI: [PAPER DOI]

This repository contains the Monte-Carlo simulation used to produce the
phase-separation results in the paper, together with the post-processing
pipeline that converts lattice snapshots into the diffraction-like *q(t)*
intensity maps shown in the figures.

---

## Overview

The model places iodide and bromide ions on a 2-D lattice and evolves their
positions via a Kawasaki-exchange Metropolis algorithm.  Charge carriers
(hole and electron polarons) are regenerated periodically from a
concentration-weighted probability distribution and couple to the ionic
configuration through a local field energy.  The simulation reproduces the
three experimental stages (dark equilibration, illumination, and recovery)
by adjusting polaron counts and gradient parameters between cycles.

Key source files:

| File | Contents |
|---|---|
| `mc/engine.py` | Metropolis loop (Numba JIT), polaron-field generator, `MCPhaseSeparation` class |
| `mc/postprocess.py` | `concentration_to_qt` (concentration → *q*-space histogram), `voigt_blur_1d` |
| `mc/viz.py` | Matplotlib helpers for lattice snapshots, *q(t)* maps, and animations |
| `mc/progress.py` | Progress-bar wrappers for long runs |
| `mc/saved_arrays/` | Concentration → *q* lookup tables (`conInterp.npy`, `concentration.npy`); see paper supplement for derivation |
| `example.ipynb` | Self-contained notebook; read this first |
| `_build_notebooks.py` | Developer helper that regenerates `example.ipynb` from source |

---

## Requirements

Python ≥ 3.10 with the packages listed in `requirements.txt`:

```
numpy, scipy, numba, h5py, matplotlib, tqdm
fast_histogram   # optional; ~3x speedup in concentration_to_qt
```

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate mc-sim
```

### pip

```bash
pip install -r requirements.txt
```

---

## Quickstart

Open `example.ipynb` in Jupyter and run all cells.  
The `mul` parameter at the top of the *Simulation parameters* cell controls
the trade-off between run time and grid size:

| `mul` | Grid      | Total MC steps | Approx. run time      |
|-------|-----------|----------------|-----------------------|
| 0.2   | 706 × 100 | 2.5 × 10⁹      | ~ 1-2 h (single core) |
| 1.0   | 706 × 500 | 1.25 × 10¹⁰    | ~ 13 h (single core)  |

`mul = 1.0` reproduces the exact published simulation.  
`mul = 0.2` reproduces the same physics at reduced cost and is suitable for
verifying the installation and exploring the parameter space.

All physics parameters (`J`, `h`, `T`, `e_barrier`, polaron sigmas,
gradients) are identical for every value of `mul`.

---

## Reproducing the paper figures

Run the notebook with `mul = 1.0`, including the optional HDF5 cell at the end.
The output file (`example_output.h5`) contains:

- `/qt`: the *q(t)* intensity matrix (frames × q-bins)
- `/raw/lattice`, `/raw/holes`, `/raw/electrons`: full snapshot stacks

The *q(t)* map produced by `concentration_to_qt` + `voigt_blur_1d` corresponds
directly to the diffraction time series shown in Figure 2. of the paper.

---

## Citation

If you use this code, please cite **both** the paper and the code:

**Paper:**
```
[AUTHOR 1], [AUTHOR 2], … "[PAPER TITLE]."
[JOURNAL] [VOLUME], [PAGES] ([YEAR]).
https://doi.org/[PAPER DOI]
```

**Code:**
```
Sebastian Schwartzkopff. Monte-Carlo phase-separation simulation ([VERSION]).
Zenodo. https://doi.org/[ZENODO DOI]
```

A `CITATION.cff` file is not yet included; it will be added once the Zenodo
DOI is assigned.

---

## License

MIT (LICENSE).
