import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch

# Statistical api
def mean_firing_rate(spike_train):
    """

    Parameters
    ----------
    spike_train： numpy.ndarray
                2 dimensional ndarray, the first axis is time dimension

    Returns
    -------

    """
    acitvate_idx = np.where(spike_train.mean(axis=0) > 0.001)[0]
    if len(acitvate_idx) > 10:
        return np.mean(spike_train[:, acitvate_idx])
    else:
        return 0.


def instantaneous_rate(spike_train, bin_width=5):
    """
    Parameters
    ----------
    spike_train: numpy.ndarray
                2 dimensional ndarray, the first axis is time dimension
    bin_width: window width to count spikes

    Returns
    -------

    """
    spike_rate = spike_train.mean(axis=1)
    out = uniform_filter1d(spike_rate.astype(np.float32), size=bin_width)
    return out


def spike_psd(spike_train, sampling_rate=1000, subtract_mean=False):
    """

    spike spectrum of instantaneous fire rate

    """
    st = instantaneous_rate(spike_train, bin_width=5)
    if subtract_mean:
        data = st - np.mean(st)
    else:
        data = st
    N_signal = data.size
    fourier_transform = np.abs(np.fft.rfft(data))
    abs_fourier_transform = np.abs(fourier_transform)
    power_spectrum = np.square(abs_fourier_transform)
    frequency = np.linspace(0, sampling_rate / 2, len(power_spectrum))
    return frequency, power_spectrum


def pearson_cc(spike_train, pairs=20):
    """
    compute pearson correlation of instantaneous_rate of parir neurons
    Parameters
    ----------
    spike_train
    pairs: num pairs to compute correlation

    Returns
    -------

    """
    valid_idx = np.where(spike_train.mean(axis=0) > 0.001)[0]
    # print(valid_idx)
    if len(valid_idx) < 10:
        return 0.
    correlation = []
    for _ in range(pairs):
        a, b = np.random.choice(valid_idx, size=(2,), replace=False)
        x1 = uniform_filter1d(spike_train[:, a].astype(np.float32), size=50, )
        x2 = uniform_filter1d(spike_train[:, b].astype(np.float32), size=50, )
        correlation.append(np.corrcoef(x1, x2)[0, 1])
    correlation = np.array(correlation)
    return np.mean(correlation)


def correlation_coefficent(spike_train):
    ins_rate = instantaneous_rate(spike_train, bin_width=5)
    return np.std(ins_rate) / np.mean(ins_rate)


def coefficient_of_variation(spike_train):
    activate_idx = (spike_train.sum(0) > 5).nonzero()[0]
    if len(activate_idx) < 10:
        return np.nan
    else:
        cvs = []
        for i in activate_idx:
            out = spike_train[:, i].nonzero()[0]
            fire_interval = out[1:] - out[:-1]
            cvs.append(fire_interval.std() / fire_interval.mean())
        return np.array(cvs).mean()


def spike_spectrum(spike):
    ins_fr = instantaneous_rate(spike)
    freqs, psd = welch(ins_fr, 1.,return_onesided=True, scaling='density')
    return freqs, psd
