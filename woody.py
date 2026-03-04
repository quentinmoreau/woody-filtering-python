"""
woody_filter.py
---------------
Python implementation of the Woody filter for temporal alignment of EEG/MEG
single-trial epochs with temporal jitter, including a stochastic variant with
random initialisation to escape local minima.

Inspired by the `woody.m` function from the WFDB Toolbox for MATLAB and Octave
(Silva, Moody & Moody, 2021; PhysioNet, https://doi.org/10.13026/6zcz-e163),
distributed under the GNU General Public License v3.

Authors: Quentin Moreau, James Bonaiuto
Date: 2026-01-19

Original algorithm:
    Woody, C. D. (1967). Characterization of an adaptive filter for the
    analysis of variable latency neuroelectric signals.
    Medical and Biological Engineering, 5(6), 539-553.

License: GNU General Public License v3.0
"""

import numpy as np
from scipy.signal import correlate
from scipy.stats import pearsonr


def woody(x, tol=0.1, max_it=100, max_shift=None):
    """
    Iterative Woody filter for aligning single-trial epochs with temporal jitter.

    At each iteration, every trial is cross-correlated with the current template
    to estimate its latency offset. Trials are then shifted accordingly and the
    template is updated as the mean of the aligned trials. Iteration stops when
    the change in mean Pearson correlation between the aligned trials and the
    template falls below ``tol``.

    Cross-correlations are computed using an unbiased normalisation
    (divided by ``n_timepoints - |lag|``), which corrects for the reduced
    overlap at large lags and prevents spurious peaks near the boundaries.

    Parameters
    ----------
    x : np.ndarray, shape (n_timepoints, n_trials)
        Signal matrix. Each column is one trial.
    tol : float, optional
        Convergence tolerance: iteration stops when
        ``|mean_corr_current - mean_corr_previous| < tol``. Default: 0.1.
    max_it : int, optional
        Maximum number of iterations. Default: 100.
    max_shift : int or None, optional
        Maximum lag (in samples) to consider during cross-correlation peak
        search. Restricting this range is strongly recommended at low SNR to
        avoid spurious alignments at the edges of the cross-correlation
        function. If None, the full range is searched. Default: None.

    Returns
    -------
    template : np.ndarray, shape (n_timepoints,)
        Final latency-aligned average waveform.
    est_lags : np.ndarray, shape (n_trials,)
        Estimated lag for each trial in samples. A positive lag means the trial
        was shifted to the right (i.e., the trial peak was delayed relative to
        the template).
    p : float
        Mean Pearson correlation between each aligned trial and the final
        template, used as the convergence criterion.

    """
    n_timepoints, n_trials = x.shape
    template = np.mean(x, axis=1)

    # Unbiased normalisation weights for cross-correlation
    lags = np.arange(-(n_timepoints - 1), n_timepoints)
    norm = n_timepoints - np.abs(lags)

    # Centre index of the full cross-correlation output (lag = 0)
    ref = len(correlate(template, template, mode="full")) // 2

    est_lags = np.zeros(n_trials, dtype=int)
    p = 0.0
    conv = True
    iteration_idx = 0

    while conv and (iteration_idx < max_it):

        aligned_trials = np.zeros((n_timepoints, n_trials))

        for i in range(n_trials):
            trial = x[:, i]

            # Unbiased cross-correlation between template and trial
            cross_corr = correlate(template, trial, mode="full") / norm

            # Optionally restrict the search range to ±max_shift samples
            if max_shift is not None:
                search_start = max(0, ref - max_shift)
                search_end = min(len(cross_corr), ref + max_shift + 1)
                ind = np.argmax(cross_corr[search_start:search_end]) + search_start
            else:
                ind = np.argmax(cross_corr)

            lag = ref - ind
            est_lags[i] = lag

            # Shift trial to align with template
            if lag > 0:
                aligned_trials[: n_timepoints - lag, i] = trial[lag:]
            elif lag < 0:
                lag_abs = abs(lag)
                aligned_trials[lag_abs:, i] = trial[: n_timepoints - lag_abs]
            else:
                aligned_trials[:, i] = trial

        # Update template as the mean of aligned trials
        template = np.mean(aligned_trials, axis=1)

        # Convergence check: mean Pearson r between aligned trials and template
        p_old = p
        corrs = np.array(
            [pearsonr(aligned_trials[:, i], template)[0] for i in range(n_trials)]
        )
        p = float(np.mean(corrs))

        if abs(p - p_old) < tol:
            conv = False

        iteration_idx += 1

    return template, est_lags, p


def woody_stochastic(
    x,
    n_runs=1000,
    max_lag_init=50,
    tol=0.1,
    max_it=100,
    max_shift=None,
    verbose=False,
    random_state=None,
):
    """
    Stochastic Woody filter with random initialisation.

    Runs the Woody filter ``n_runs`` times, each starting from a different
    random per-trial lag offset drawn uniformly from
    ``[-max_lag_init, +max_lag_init]``. The run that yields the highest
    mean Pearson correlation with the final template is returned.

    Random initialisation helps escape local cross-correlation maxima that
    can trap the deterministic algorithm, particularly at low SNR or when
    the true jitter distribution is wide.

    Parameters
    ----------
    x : np.ndarray, shape (n_timepoints, n_trials)
        Signal matrix. Each column is one trial.
    n_runs : int, optional
        Number of random initialisations. Default: 1000.
    max_lag_init : int, optional
        Half-range of the uniform distribution used to draw initial lag
        offsets (in samples). Default: 50.
    tol : float, optional
        Convergence tolerance passed to :func:`woody`. Default: 0.1.
    max_it : int, optional
        Maximum iterations per run passed to :func:`woody`. Default: 100.
    max_shift : int or None, optional
        Maximum lag search window passed to :func:`woody`. Default: None.
    verbose : bool, optional
        Print best correlation every 100 runs. Default: False.
    random_state : int or None, optional
        Seed for the random number generator for reproducibility. Default: None.

    Returns
    -------
    best_avg : np.ndarray, shape (n_timepoints,)
        Aligned average waveform from the best run.
    best_lags : np.ndarray, shape (n_trials,)
        Combined lag estimates (initialisation + Woody refinement) for the
        best run.
    best_corr : float
        Mean Pearson correlation of the best run.

    Notes
    -----
    The total lag returned as ``best_lags`` is the sum of the random
    initialisation offset and the lag estimated by the deterministic Woody
    filter on the pre-shifted data. This represents the full shift applied
    to each trial relative to the original input ``x``.

    """
    n_timepoints, n_trials = x.shape
    rng = np.random.default_rng(random_state)

    best_corr = -np.inf
    best_avg = None
    best_lags = None

    for run in range(n_runs):

        # Random initial shifts drawn uniformly from [-max_lag_init, +max_lag_init]
        init_lags = rng.integers(-max_lag_init, max_lag_init + 1, size=n_trials)

        # Apply initial shift to all trials before passing to woody()
        x_shifted = np.zeros_like(x)
        for i in range(n_trials):
            lag = init_lags[i]
            if lag > 0:
                x_shifted[: n_timepoints - lag, i] = x[lag:, i]
            elif lag < 0:
                x_shifted[-lag:, i] = x[: n_timepoints + lag, i]
            else:
                x_shifted[:, i] = x[:, i]

        try:
            aligned_avg, detected_lags, p = woody(
                x_shifted, tol=tol, max_it=max_it, max_shift=max_shift
            )

            # Total lag = initialisation offset + Woody refinement
            total_lags = init_lags + detected_lags

            if p > best_corr:
                best_corr = p
                best_avg = aligned_avg.copy()
                best_lags = total_lags.copy()

                if verbose and run % 100 == 0:
                    print(f"Run {run}/{n_runs}: new best r = {best_corr:.4f}")

        except Exception as e:
            if verbose:
                print(f"Run {run} failed: {e}")
            continue

    return best_avg, best_lags, best_corr
