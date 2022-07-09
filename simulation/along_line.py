# -*- coding: utf-8 -*- 
# @Time : 2022/6/22 15:34 
# @Author : lepold
# @File : along_line.py

import os
import glob

import numpy as np
from scipy.io import savemat
from analysis.avalanches import *
from analysis.spike_statistics import *


def total_avalanches(spike, threshold=1):
    aval_size_total = []
    aval_dur_total = []
    for i in range(40):
        seed = np.random.randint(low=0, high=500, size=(100), dtype=np.int32)
        _, aval_size, aval_dur, _ = compute_avalanche(spike[:, seed], threshold)
        aval_size_total.append(aval_size)
        aval_dur_total.append(aval_dur)
    aval_size_total = np.concatenate(aval_size_total, axis=0)
    aval_dur_total = np.concatenate(aval_dur_total, axis=0)
    return aval_size_total, aval_dur_total


data_dir = r"/public/home/ssct004t/project/zenglb/CriticalNN/data/along_critical"
total = 437
mean_fr = np.empty(total)
pcc = np.empty(total)
cc = np.empty(total)
ks_size = np.empty(total)
ks_dur = np.empty(total)
exponent_size = np.empty(total)
exponent_dur = np.empty(total)
cv = np.empty(total)
peak_power = np.empty(total)
valid_idx = []
for i in range(total):
    print(i)
    path = os.path.join(data_dir, f"spike_{i}.npy")
    log = np.load(path)
    log = log.reshape((-1, log.shape[-1]))[800:, ]
    part_log = log[-3000:, :]
    if part_log.max() < 1:
        continue
    valid_idx.append(i)
    mean_fr[i] = mean_firing_rate(part_log)
    cv[i] = coefficient_of_variation(part_log)
    pcc[i] = pearson_cc(part_log, pairs=200)
    cc[i] = correlation_coefficent(part_log)
    ins_fr = instantaneous_rate(log)
    freqs, psd = welch(ins_fr, 1000, return_onesided=True, scaling='density', nperseg=1024, noverlap=800)
    id = np.argmax(psd[:50])
    peak_power[i] = freqs[id]
    # freq, psd = spike_spectrum(log)
    aval_size, aval_dur = total_avalanches(log)
    alpha, D = fit_powerlaw(aval_size)
    exponent_size[i] = alpha
    ks_size[i] = D
    alpha, D = fit_powerlaw(aval_dur)
    exponent_dur[i] = alpha
    ks_dur[i] = D
valid_idx = np.array(valid_idx)
np.savez(os.path.join(data_dir, "along_line.npz"), mean_fr=mean_fr,
         pcc=pcc, cc=cc, ks_size=ks_size, ks_dur=ks_dur, exponent_size=exponent_size, exponent_dur=exponent_dur, cv=cv, valid_idx=valid_idx, peak_power=peak_power)


print("Done!!")