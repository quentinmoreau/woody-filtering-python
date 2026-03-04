# woody-filtering-python

A Python implementation of the Woody filter for EEG/MEG single-trial temporal alignment, including a stochastic variant for improved convergence and robustness.

---

## Background

The Woody filter is an iterative cross-correlation method for aligning single-trial to the grand average by estimating individual trial latency jitter (Woody, 1967). 

This repository provides:
- A clean, standalone Python implementation of the classic Woody filter
- A stochastic variant with random initialisation to avoid local cross-correlation maxima


## Installation

No package installation required. Simply clone the repo and import the module:
```bash
git clone https://github.com/quentinmoreau/woody-filtering-python.git
```

**Dependencies:**
```bash
pip install numpy scipy
```

## Attribution

This Python implementation was inspired by the `woody.m` function from the **WFDB Toolbox for MATLAB and Octave** (v0.10.0), originally developed by Ikaro Silva, Benjamin Moody, and George Moody, and distributed under the GNU General Public License v3.

> Silva, I., Moody, B., & Moody, G. (2021). *Waveform Database Software Package (WFDB) for MATLAB and Octave* (version 0.10.0). PhysioNet. https://doi.org/10.13026/6zcz-e163

> Silva I, Moody G. An Open-source Toolbox for Analysing and Processing PhysioNet Databases in MATLAB and Octave. *Journal of Open Research Software*. 2014;2(1):e27. https://doi.org/10.5334/jors.bi

> Goldberger A, et al. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. *Circulation*. 2000;101(23):e215–e220.

Original algorithm:
> Woody, C. D. (1967). Characterization of an adaptive filter for the analysis of variable latency neuroelectric signals. *Medical and Biological Engineering*, 5(6), 539–553.

## License

This project is licensed under the **GNU General Public License v3.0** (GPL-3.0), in keeping with the license of the original WFDB MATLAB toolbox from which it was derived.
