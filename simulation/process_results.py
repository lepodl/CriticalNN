import os
from analysis.avalanches import *
from analysis.spike_statistics import *

results_dir = "/public/home/ssct004t/project/zenglb/CriticalNN/data/grid_search"
ampa_contribution = np.linspace(0.5, 1, num=100, endpoint=True)
gabaA_contribution = np.linspace(0., 0.5, num=100, endpoint=True)
contribution = np.stack(np.meshgrid(ampa_contribution, gabaA_contribution, indexing='ij'), axis=-1).reshape((-1, 2))
total = contribution.shape[0]
mean_fr = np.empty((100, 100)).reshape(-1)
pcc = np.empty((100, 100)).reshape(-1)
cc = np.empty((100, 100)).reshape(-1)
ks = np.empty((100, 100)).reshape(-1)
exponent = np.empty((100, 100)).reshape(-1)
cv = np.empty((100, 100)).reshape(-1)
for i in range(total):
    path = os.path.join(results_dir, f"log_{i}.npy")
    log = np.load(path)
    mean_fr[i] = mean_firing_rate(log)
    cv[i] = coefficient_of_variation(log)
    pcc[i] = pearson_cc(log)
    cc[i] = correlation_coefficent(log)
    _, aval_size, _, _ = compute_avalanche(log)
    alpha, D = fit_powerlaw(aval_size)
    exponent[i] = alpha
    ks[i] = D
mean_fr = mean_fr.reshape((100, 100))
pcc = pcc.reshape((100, 100))
cc = cc.reshape((100, 100))
ks = ks.reshape((100, 100))
exponent = exponent.reshape((100, 100))
cv = cv.reshape((100, 100))
np.savez(os.path.join("/public/home/ssct004t/project/zenglb/CriticalNN/data", "grid_search.npz"), mean_fr=mean_fr,
         pcc=pcc, cc=cc, ks=ks, exponent=exponent, cv=cv, ampa_contribution=ampa_contribution,
         gabaA_contribution=gabaA_contribution)
print("Done!!")
