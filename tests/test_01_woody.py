import numpy as np
import pytest

from woody import woody, woody_stochastic, _shift_with_zeros


def _make_gaussian_trials(
    true_jitters,
    n_times=751,
    sfreq=500,
    tmin=-0.5,
    tmax=1.0,
    peak_time=0.2,
    amplitude=1.0,
    width=0.04,
    amplitude_jitter_sd=0.0,
    noise_sd=0.0,
    random_state=0,
):
    """
    Generate single-component Gaussian trials with optional amplitude jitter and noise.

    Returns
    -------
    trials : np.ndarray, shape (n_trials, n_times)
    """
    rng = np.random.default_rng(random_state)
    times = np.linspace(tmin, tmax, n_times)
    n_trials = len(true_jitters)

    trials = np.zeros((n_trials, n_times), dtype=float)
    for i, lag in enumerate(true_jitters):
        amp = amplitude * rng.lognormal(mean=0.0, sigma=amplitude_jitter_sd)
        waveform = amp * np.exp(
            -0.5 * ((times - (peak_time + lag / sfreq)) / width) ** 2
        )
        if noise_sd > 0:
            waveform = waveform + rng.normal(0.0, noise_sd, size=n_times)
        trials[i] = waveform

    return trials


def _make_ern_pe_trials(
    true_jitters,
    n_times=751,
    sfreq=500,
    tmin=-0.5,
    tmax=1.0,
    amplitude_jitter_sd=0.2,
    noise_sd=0.0,
    random_state=0,
):
    """
    Generate ERN + Pe style trials similar to the user's example.
    """
    rng = np.random.default_rng(random_state)
    times = np.linspace(tmin, tmax, n_times)
    n_trials = len(true_jitters)

    ern_peak_time, ern_amplitude, ern_width = 0.15, -8.0, 0.05
    pe_peak_time, pe_amplitude, pe_width = 0.30, 10.0, 0.07

    trials = np.zeros((n_trials, n_times), dtype=float)

    for i, lag in enumerate(true_jitters):
        ern_amp = ern_amplitude * rng.lognormal(0.0, amplitude_jitter_sd)
        pe_amp = pe_amplitude * rng.lognormal(0.0, amplitude_jitter_sd)

        ern = ern_amp * np.exp(
            -0.5 * ((times - (ern_peak_time + lag / sfreq)) / ern_width) ** 2
        )
        pe = pe_amp * np.exp(
            -0.5 * ((times - (pe_peak_time + lag / sfreq)) / pe_width) ** 2
        )

        trial = ern + pe
        if noise_sd > 0:
            trial = trial + rng.normal(0.0, noise_sd, size=n_times)

        trials[i] = trial

    return trials


def _relative_lags(lags):
    lags = np.asarray(lags)
    return lags - lags[0]


def test_shift_with_zeros_zero_lag_returns_same_signal():
    trial = np.array([1, 2, 3, 4, 5])
    shifted = _shift_with_zeros(trial, 0)
    np.testing.assert_array_equal(shifted, trial)


def test_shift_with_zeros_positive_lag_shifts_left():
    trial = np.array([1, 2, 3, 4, 5])
    shifted = _shift_with_zeros(trial, 2)
    expected = np.array([3, 4, 5, 0, 0])
    np.testing.assert_array_equal(shifted, expected)


def test_shift_with_zeros_negative_lag_shifts_right():
    trial = np.array([1, 2, 3, 4, 5])
    shifted = _shift_with_zeros(trial, -2)
    expected = np.array([0, 0, 1, 2, 3])
    np.testing.assert_array_equal(shifted, expected)


def test_shift_with_zeros_raises_for_non_1d_input():
    with pytest.raises(ValueError, match="trial must be 1D"):
        _shift_with_zeros(np.zeros((3, 3)), 1)


def test_woody_returns_expected_shapes():
    true_jitters = [0, 5, 10, 15]
    trials = _make_gaussian_trials(true_jitters, random_state=1)

    template, est_lags, p = woody(trials.T)

    assert template.shape == (trials.shape[1],)
    assert est_lags.shape == (trials.shape[0],)
    assert isinstance(p, float)


def test_woody_no_jitter_recovers_equal_lags():
    true_jitters = [0, 0, 0, 0, 0]
    trials = _make_gaussian_trials(
        true_jitters,
        amplitude_jitter_sd=0.0,
        noise_sd=0.0,
        random_state=2,
    )

    _, est_lags, p = woody(trials.T)

    assert np.all(est_lags == est_lags[0])
    assert p > 0.99


def test_woody_recovers_relative_lags_for_simple_gaussian_trials():
    true_jitters = [30, 40, 10, 0, 50]
    trials = _make_gaussian_trials(
        true_jitters,
        amplitude_jitter_sd=0.0,
        noise_sd=0.0,
        width=0.03,
        random_state=3,
    )

    _, est_lags, p = woody(trials.T)

    est_rel = _relative_lags(est_lags)
    true_rel = _relative_lags(true_jitters)

    # Relative lag recovery should be very good in the simple, noiseless case.
    mae = np.mean(np.abs(est_rel - true_rel))
    assert mae <= 2.0
    assert p > 0.95


def test_woody_respects_max_shift_constraint():
    true_jitters = [0, 5, 10, 15]
    trials = _make_gaussian_trials(
        true_jitters,
        amplitude_jitter_sd=0.0,
        noise_sd=0.0,
        random_state=4,
    )

    _, est_lags, _ = woody(trials.T, max_shift=6)

    assert np.all(est_lags >= -6)
    assert np.all(est_lags <= 6)


def test_woody_stochastic_is_reproducible_with_fixed_random_state():
    true_jitters = [30, 40, 10, 0, 50]
    trials = _make_ern_pe_trials(
        true_jitters,
        amplitude_jitter_sd=0.2,
        noise_sd=0.0,
        random_state=5,
    )

    out1 = woody_stochastic(
        trials.T,
        n_runs=200,
        max_lag_init=10,
        random_state=123,
    )
    out2 = woody_stochastic(
        trials.T,
        n_runs=200,
        max_lag_init=10,
        random_state=123,
    )

    template1, lags1, p1 = out1
    template2, lags2, p2 = out2

    np.testing.assert_allclose(template1, template2)
    np.testing.assert_array_equal(lags1, lags2)
    assert p1 == p2


def test_woody_stochastic_returns_expected_shapes():
    true_jitters = [0, 10, 20, 5]
    trials = _make_gaussian_trials(true_jitters, random_state=6)

    template, est_lags, best_p = woody_stochastic(
        trials.T,
        n_runs=50,
        max_lag_init=5,
        random_state=42,
    )

    assert template.shape == (trials.shape[1],)
    assert est_lags.shape == (trials.shape[0],)
    assert isinstance(best_p, float)


def test_woody_stochastic_not_worse_than_deterministic_on_simple_case():
    true_jitters = [30, 40, 10, 0, 50]
    trials = _make_gaussian_trials(
        true_jitters,
        amplitude_jitter_sd=0.05,
        noise_sd=0.0,
        width=0.03,
        random_state=7,
    )

    _, est_lags_det, p_det = woody(trials.T)
    _, est_lags_sto, p_sto = woody_stochastic(
        trials.T,
        n_runs=500,
        max_lag_init=10,
        random_state=99,
    )

    det_mae = np.mean(
        np.abs(_relative_lags(est_lags_det) - _relative_lags(true_jitters))
    )
    sto_mae = np.mean(
        np.abs(_relative_lags(est_lags_sto) - _relative_lags(true_jitters))
    )

    assert p_sto >= p_det
    assert (sto_mae <= 1.0) and (det_mae <= 1.0)


def test_woody_runs_on_user_like_ern_pe_example():
    true_jitters = [30, 40, 10, 0, 50]
    trials = _make_ern_pe_trials(
        true_jitters,
        amplitude_jitter_sd=0.2,
        noise_sd=0.0,
        random_state=8,
    )

    template, est_lags, p = woody(trials.T)

    assert template.shape == (trials.shape[1],)
    assert est_lags.shape == (len(true_jitters),)
    assert np.isfinite(p)

    est_rel = _relative_lags(est_lags)
    true_rel = _relative_lags(true_jitters)
    mae = np.mean(np.abs(est_rel - true_rel))

    # This is intentionally looser than the single-Gaussian case because
    # multi-component waveforms can produce local maxima in correlation.
    assert mae < 5.0


def test_woody_stochastic_runs_on_user_like_ern_pe_example():
    true_jitters = [30, 40, 10, 0, 50]
    trials = _make_ern_pe_trials(
        true_jitters,
        amplitude_jitter_sd=0.2,
        noise_sd=0.0,
        random_state=9,
    )

    template, est_lags, best_p = woody_stochastic(
        trials.T,
        n_runs=500,
        max_lag_init=10,
        random_state=321,
    )

    assert template.shape == (trials.shape[1],)
    assert est_lags.shape == (len(true_jitters),)
    assert np.isfinite(best_p)

    est_rel = _relative_lags(est_lags)
    true_rel = _relative_lags(true_jitters)
    mae = np.mean(np.abs(est_rel - true_rel))

    assert mae < 5.0