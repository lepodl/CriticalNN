import numpy as np
import os
from analysis.avalanches import *
from analysis.spike_statistics import *

results_dir = "/public/home/ssct004t/project/zenglb/CriticalNN/data/grid_search"
ampa_contribution = np.linspace(0.5, 1, num=50, endpoint=True)
gabaA_contribution = np.linspace(0., 0.5, num=50, endpoint=True)
contribution = np.stack(np.meshgrid(ampa_contribution, gabaA_contribution, indexing='ij'), axis=-1).reshape((-1, 2))
total = contribution.shape[0]
mean_fr = np.empty((50, 50)).reshape(-1)
pcc = np.empty((50, 50)).reshape(-1)
cc = np.empty((50, 50)).reshape(-1)
ks = np.empty((50, 50)).reshape(-1)
for i in range(total):
    path = os.path.join(results_dir, f"log_{i}.npy")
    log = np.load(path)
    mean_fr[i] = mean_firing_rate(log)
    pcc[i] = pearson_cc(log)
    cc[i] = correlation_coefficent(log)
    _, aval_size, _, _ = compute_avalanche(log)
    _, D = fit_powerlaw(aval_size)
    ks[i] = D
mean_fr = mean_fr.reshape((50, 50))
pcc = pcc.reshape((50, 50))
cc = cc.reshape((50, 50))
ks = ks.reshape((50, 50))
np.savez(os.path.join("/public/home/ssct004t/project/zenglb/CriticalNN/data"), "grid_search.npz", mean_fr=mean_fr, pcc=pcc, cc=cc, ks=ks)
print("Done!!")

