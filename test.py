import numpy as np
import scipy as sp
from woody_filter import woody, woody_stochastic

if __name__ == '__main__':
    # Parameters
    sfreq = 500
    tmin, tmax = -0.5, 1.0
    n_times = int((tmax - tmin) * sfreq) + 1
    times = np.linspace(tmin, tmax, n_times)
    amplitude_jitter_sd = 0.2  # 20% CV

    # ERN + Pe parameters
    ern_peak_time, ern_amplitude, ern_width = 0.15, -8, 0.05
    pe_peak_time, pe_amplitude, pe_width = 0.3, 10, 0.07

    n_trials = 5

    trials = np.zeros((n_trials, n_times))
    true_jitters = [30, 40, 10, 0, 50]
    #true_jitters = [0, 0, 0, 0, 0]

    for trial in range(n_trials):
        ern_amp = ern_amplitude * np.random.lognormal(0, amplitude_jitter_sd)
        pe_amp = pe_amplitude * np.random.lognormal(0, amplitude_jitter_sd)

        ern_trial = ern_amp * np.exp(-0.5 * ((times - (ern_peak_time + true_jitters[trial]/sfreq)) / ern_width) ** 2)
        pe_trial = pe_amp * np.exp(-0.5 * ((times - (pe_peak_time + true_jitters[trial]/sfreq)) / pe_width) ** 2)

        trials[trial, :] = ern_trial + pe_trial

    template, est_lags, p = woody(trials.T)

    print(true_jitters)
    print(est_lags)
    print(p)
    print(np.mean(np.abs(true_jitters-est_lags)))

    template, est_lags, best_p = woody_stochastic(trials.T, max_lag_init=10, n_runs=10000)
    print(true_jitters)
    print(est_lags)
    print(best_p)
    print(np.mean(np.abs(true_jitters - est_lags)))
