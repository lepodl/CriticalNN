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


def grid_search_process():
    results_dir = "/public/home/ssct004t/project/zenglb/CriticalNN/data/grid_search_d100_new_new"

    # ampa_contribution = np.linspace(0., 1., num=num, endpoint=True)
    # gabaA_contribution = np.linspace(0., 1., num=num, endpoint=True)
    # contribution = np.stack(np.meshgrid(ampa_contribution, gabaA_contribution, indexing='ij'), axis=-1).reshape((-1, 2))
    # total = contribution.shape[0]
    num = 80
    ampa = np.linspace(0.022 / 2, 0.022 / 1, num)
    gabaA = np.linspace(0., 0.5 / 2, num)
    total = num * num
    mean_fr = np.empty((num,num)).reshape(-1)
    pcc = np.empty((num, num)).reshape(-1)
    cc = np.empty((num, num)).reshape(-1)
    ks = np.empty((num, num)).reshape(-1)
    exponent = np.empty((num, num)).reshape(-1)
    cv = np.empty((num, num)).reshape(-1)
    for i in range(total):
        path = os.path.join(results_dir, f"log_{i}.npy")
        log = np.load(path)
        mean_fr[i] = mean_firing_rate(log)
        cv[i] = coefficient_of_variation(log)

        if mean_fr[i]<=0.002:
            alpha=np.nan
            D = np.nan
            pcc[i] = 0.
            cc[i] = 0.
        else:
            pcc[i] = pearson_cc(log, pairs=200)
            cc[i] = correlation_coefficent(log)
            _, aval_size, _, _ = compute_avalanche(log)
            alpha, D = fit_powerlaw(aval_size)
        exponent[i] = alpha
        ks[i] = D
    mean_fr = mean_fr.reshape((num, num))
    pcc = pcc.reshape((num, num))
    cc = cc.reshape((num, num))
    ks = ks.reshape((num, num))
    exponent = exponent.reshape((num, num))
    cv = cv.reshape((num, num))
    np.savez(os.path.join("/public/home/ssct004t/project/zenglb/CriticalNN/data", "grid_search_d500_new_new.npz"), mean_fr=mean_fr,
             pcc=pcc, cc=cc, ks=ks, exponent=exponent, cv=cv, ampa=ampa,
             gabaA=gabaA)
    # savemat(os.path.join("/public/home/ssct004t/project/zenglb/CriticalNN/data", "grid_search_d500_specific_range_0to1_npoint_50.mat"),
    #         mdict={"mean_fr": mean_fr, "pcc": pcc, "cc": cc, "cv": cv, 'ampa_contribution':ampa_contribution, 'gabaA_contribution':gabaA_contribution})  # "ks": ks, "exponent": exponent,
    print("Done!!")


def size_influence():
    data_dir = r"/public/home/ssct004t/project/zenglb/CriticalNN/data/multi_size_results/from_critical"
    total = 50
    mean_fr = np.empty(total)
    pcc = np.empty(total)
    cc = np.empty(total)
    ks = np.empty(total)
    exponent = np.empty(total)
    cv = np.empty(total)
    for i in range(total):
        print(i)
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


def cuda_grid_search_process():
    results_dir = "/public/home/ssct004t/project/zenglb/CriticalNN/data/grid_search_on_region_d300_1e7"
    num = 5
    total = 25
    ampa_contribution = np.linspace(0., 1., num=50, endpoint=True)
    gabaA_contribution = np.linspace(0., 1., num=50, endpoint=True)
    sample_idx = np.arange(5, 50, 5, dtype=np.int8)
    ampa_contribution = ampa_contribution[sample_idx]
    gabaA_contribution = gabaA_contribution[sample_idx]
    mean_fr = np.empty((25, 90))
    pcc = np.empty((25, 90))
    for i in range(total):
        path = os.path.join(os.path.join(results_dir, f"grid_{i}"), f"spike.npy")
        log = np.load(path)
        log = log.reshape((-1, log.shape[-1]))
        log = log[-5000:]
        for j in range(90):
            sub_log = log[:, j*300:(j+1)*300]
            mean_fr[i, j] = mean_firing_rate(sub_log)
            pcc[i, j] = pearson_cc(sub_log, pairs=200)

    np.savez(os.path.join("/public/home/ssct004t/project/zenglb/CriticalNN/data/", "grid_search_on_region_d300_1e7.npz"), mean_fr=mean_fr,
             pcc=pcc, ampa_contribution=ampa_contribution,
             gabaA_contribution=gabaA_contribution)
    # savemat(os.path.join("/public/home/ssct004t/project/zenglb/CriticalNN/data/", "grid_search_on_region_d300_1e7.mat"),
    #         mdict={"mean_fr": mean_fr, "pcc": pcc, 'ampa_contribution':ampa_contribution, 'gabaA_contribution':gabaA_contribution})  # "ks": ks, "exponent": exponent,
    print("Done!!")



if __name__ == '__main__':
    grid_search_process()
    # size_influence()
    # big_block_simulation()
    # cuda_grid_search_process()