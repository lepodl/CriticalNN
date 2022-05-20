import os
import glob
from scipy.io import savemat
from analysis.avalanches import *
from analysis.spike_statistics import *

def total_avalanches(spike, threshold=1):
    aval_size_total = []
    aval_dur_total = []
    for i in range(80):
        seed = np.random.randint(low=0, high=1000, size=(100), dtype=np.int32)
        _, aval_size, aval_dur, _ = compute_avalanche(spike[:, seed], threshold)
        aval_size_total.append(aval_size)
        aval_dur_total.append(aval_dur)
    aval_size_total = np.concatenate(aval_size_total, axis=0)
    aval_dur_total = np.concatenate(aval_dur_total, axis=0)
    return aval_size_total, aval_dur_total


def grid_search_process():
    results_dir = "/public/home/ssct004t/project/zenglb/CriticalNN/data/grid_search_d500_specific_range"
    ampa_contribution = np.linspace(0.4, 1., num=100, endpoint=True)
    gabaA_contribution = np.linspace(0.2, 0.8, num=100, endpoint=True)
    contribution = np.stack(np.meshgrid(ampa_contribution, gabaA_contribution, indexing='ij'), axis=-1).reshape((-1, 2))
    total = contribution.shape[0]
    mean_fr = np.empty((100,100)).reshape(-1)
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
        pcc[i] = pearson_cc(log, pairs=100)
        cc[i] = correlation_coefficent(log)
        if mean_fr[i]<=0.002:
            alpha=np.nan
            D = np.nan
        else:
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
    np.savez(os.path.join("/public/home/ssct004t/project/zenglb/CriticalNN/data", "grid_search_d500_specific_range2.npz"), mean_fr=mean_fr,
             pcc=pcc, cc=cc, ks=ks, exponent=exponent, cv=cv, ampa_contribution=ampa_contribution,
             gabaA_contribution=gabaA_contribution)
    savemat(os.path.join("/public/home/ssct004t/project/zenglb/CriticalNN/data", "grid_search_d500_specific_range2.mat"),
            mdict={"mean_fr": mean_fr, "pcc": pcc, "cc": cc, "cv": cv, 'ampa_contribution':ampa_contribution, 'gabaA_contribution':gabaA_contribution})  # "ks": ks, "exponent": exponent,
    print("Done!!")


def size_influence():
    data_dir = r"/public/home/ssct004t/project/zenglb/CriticalNN/data/multi_size_result/from_critical"
    total = 50
    mean_fr = np.empty(total)
    pcc = np.empty(total)
    cc = np.empty(total)
    ks = np.empty(total)
    exponent = np.empty(total)
    cv = np.empty(total)
    for i in range(total):
        path = os.path.join(data_dir, f"size_{i}", "spike.npy")
        log = np.load(path)
        log = log.reshape((-1, log.shape[-1]))
        part_log = log[-2000:, :]
        mean_fr[i] = mean_firing_rate(part_log)
        cv[i] = coefficient_of_variation(part_log)
        pcc[i] = pearson_cc(part_log, pairs=200)
        cc[i] = correlation_coefficent(part_log)
        aval_size, _ = total_avalanches(log)
        alpha, D = fit_powerlaw(aval_size)
        exponent[i] = alpha
        ks[i] = D
    np.savez(os.path.join(data_dir, "size_influence_from_critical_new.npz"), mean_fr=mean_fr,
             pcc=pcc, cc=cc, ks=ks, exponent=exponent, cv=cv)
    print("Done!!")


def big_block_simulation():
    path_all = glob.glob("/public/home/ssct004t/project/zenglb/CriticalNN/data/100m_scale_block/*/spike.npy")
    total = len(path_all)
    mean_fr = np.zeros((100, 100))
    pcc = np.zeros((100, 100))
    cc = np.zeros((100, 100))
    ks = np.zeros((100, 100))
    exponent = np.zeros((100, 100))
    cv = np.zeros((100, 100))
    sample_idx = np.arange(5, 100, 10, dtype=np.int8)
    sample_idx = np.stack(np.meshgrid(sample_idx, sample_idx, indexing="ij"), axis=-1).reshape((-1, 2))
    for path in path_all:
        idx = path.split('/')[-2][7:]
        idx = int(idx)
        a, b = sample_idx[idx]
        log = np.load(path)
        log = log.reshape((-1, log.shape[-1]))
        part_log = log[-2000:, :]
        mean_fr[a, b] = mean_firing_rate(part_log)
        cv[a, b] = coefficient_of_variation(part_log)
        pcc[a, b] = pearson_cc(part_log)
        cc[a, b] = correlation_coefficent(part_log)
        aval_size, _ = total_avalanches(log)
        alpha, D = fit_powerlaw(aval_size)
        exponent[a, b] = alpha
        ks[a, b] = D
    np.savez(os.path.join("/public/home/ssct004t/project/zenglb/CriticalNN/data/100m_scale_block/", "big_block_simulation.npz"), mean_fr=mean_fr,
             pcc=pcc, cc=cc, ks=ks, exponent=exponent, cv=cv)
    savemat(os.path.join("/public/home/ssct004t/project/zenglb/CriticalNN/data/100m_scale_block/", "big_block_simulation.mat"), mdict={"mean_fr":mean_fr, "pcc":pcc, "cc":cc, "ks":ks, "exponent":exponent, "cv":cv})



if __name__ == '__main__':
    # grid_search_process()
    size_influence()
    # big_block_simulation()