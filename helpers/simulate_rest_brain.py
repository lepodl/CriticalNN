# -*- coding: utf-8 -*- 
# @Time : 2022/3/27 14:42 
# @Author : lepold
# @File : simulate_brain.py

import os
import time
import numpy as np
import torch
from dtb.python.dist_blockwrapper_pytorch import BlockWrapper as block_gpu
from brain_block.bold_model_pytorch import BOLD
import pandas as pd
import argparse
import h5py
import prettytable as pt
import matplotlib.pylab as plt
from scipy.io import loadmat

v_th = -50


def torch_2_numpy(u, is_cuda=True):
    assert isinstance(u, torch.Tensor)
    if is_cuda:
        return u.cpu().numpy()
    else:
        return u.numpy()


def load_if_exist(func, *args, **kwargs):
    path = os.path.join(*args)
    if os.path.exists(path + ".npy"):
        out = np.load(path + ".npy")
    else:
        out = func(**kwargs)
        np.save(path, out)
    return out


def norm(data, dim):
    return (data - np.mean(data, axis=dim, keepdims=True)) / np.std(data, axis=dim, keepdims=True)


def get_bold_signal(bold_path, rest=True):
    if not rest:
        bold_y = np.load(bold_path)["task_bold"]
        bold_y = bold_y.T
    else:
        bold_y = np.load(bold_path)["rest_bold"]
        bold_y = bold_y.T
    bold_y = norm(bold_y, dim=0)
    # bold_y = 0.02 + 0.03 * (bold_y - bold_y.min()) / (bold_y.max() - bold_y.min())
    return bold_y


def np_move_avg(a, n=10, mode="valid"):
    if a.ndim > 1:
        tmp = []
        for i in range(a.shape[1]):
            tmp.append(np.convolve(a[:, i], np.ones((n,)) * 1000 / n, mode=mode))
        tmp = np.stack(tmp, axis=1)
    else:
        tmp = np.convolve(a, np.ones((n,)) * 1000 / n, mode=mode)
    return tmp


def sample_in_cortical_and_subcortical(aal_region, neurons_per_population_base, num_sample_voxel_per_region=1,
                                       num_neurons_per_voxel=300):
    base = neurons_per_population_base
    subcortical = np.array([37, 38, 41, 42, 71, 72, 73, 74, 75, 76, 77, 78], dtype=np.int64) - 1  # region index from 0
    subblk_base = [0]
    tmp = 0
    for i in range(len(aal_region)):
        if aal_region[i] in subcortical:
            subblk_base.append(tmp + 2)
            tmp = tmp + 2
        else:
            subblk_base.append(tmp + 8)
            tmp = tmp + 8
    subblk_base = np.array(subblk_base)
    uni_region = np.unique(aal_region)
    num_sample_neurons = len(uni_region) * num_neurons_per_voxel * num_sample_voxel_per_region
    sample_idx = np.empty([num_sample_neurons, 4], dtype=np.int64)
    # the (, 0): neuron idx; (, 1): voxel idx, (,2): subblk(population) idx, (, 3) voxel idx belong to which brain region
    s1, s2 = int(0.8 * num_neurons_per_voxel), int(0.2 * num_neurons_per_voxel)
    lcm_gm = np.array([
        33.8 * 78, 33.8 * 22,
        34.9 * 80, 34.9 * 20,
        7.6 * 82, 7.6 * 18,
        22.1 * 83, 22.1 * 17], dtype=np.float64)  # ignore the L1 neurons
    lcm_gm /= lcm_gm.sum()
    cn = (lcm_gm * num_neurons_per_voxel).astype(np.int)
    c1, c2, c3, c4, c5, c6, c7, c8 = cn
    c8 += num_neurons_per_voxel - np.sum(cn)
    count_voxel = 0
    for i in uni_region:
        print("sampling for region: ", i)
        choices = np.random.choice(np.where(aal_region == i)[0], num_sample_voxel_per_region)
        for choice in choices:
            if i in subcortical:
                sample1 = np.random.choice(
                    np.arange(start=base[subblk_base[choice]], stop=base[subblk_base[choice] + 1], step=1), s1,
                    replace=False)
                sample2 = np.random.choice(
                    np.arange(start=base[subblk_base[choice] + 1], stop=base[subblk_base[choice] + 2], step=1), s2,
                    replace=False)
                sample = np.concatenate([sample1, sample2])
                sub_blk = np.concatenate(
                    [np.ones_like(sample1) * subblk_base[choice], np.ones_like(sample2) * (subblk_base[choice] + 1)])[:,
                          None]
                # print("sample_shape", sample.shape)
                sample = np.stack(np.meshgrid(sample, np.array([choice])), axis=-1).squeeze()
                sample = np.concatenate([sample, sub_blk, np.ones((num_neurons_per_voxel, 1)) * i], axis=-1)
                sample_idx[num_neurons_per_voxel * count_voxel:num_neurons_per_voxel * (count_voxel + 1), :] = sample
            else:
                sample1 = np.random.choice(
                    np.arange(start=base[subblk_base[choice]], stop=base[subblk_base[choice] + 1], step=1), c1,
                    replace=False)
                sample2 = np.random.choice(
                    np.arange(start=base[subblk_base[choice] + 1], stop=base[subblk_base[choice] + 2], step=1), c2,
                    replace=False)
                sample3 = np.random.choice(
                    np.arange(start=base[subblk_base[choice] + 2], stop=base[subblk_base[choice] + 3], step=1), c3,
                    replace=False)
                sample4 = np.random.choice(
                    np.arange(start=base[subblk_base[choice] + 3], stop=base[subblk_base[choice] + 4], step=1), c4,
                    replace=False)
                sample5 = np.random.choice(
                    np.arange(start=base[subblk_base[choice] + 4], stop=base[subblk_base[choice] + 5], step=1), c5,
                    replace=False)
                sample6 = np.random.choice(
                    np.arange(start=base[subblk_base[choice] + 5], stop=base[subblk_base[choice] + 6], step=1), c6,
                    replace=False)
                sample7 = np.random.choice(
                    np.arange(start=base[subblk_base[choice] + 6], stop=base[subblk_base[choice] + 7], step=1), c7,
                    replace=False)
                sample8 = np.random.choice(
                    np.arange(start=base[subblk_base[choice] + 7], stop=base[subblk_base[choice] + 8], step=1), c8,
                    replace=False)
                sample = np.concatenate([sample1, sample2, sample3, sample4, sample5, sample6, sample7, sample8])
                # print("sample_shape", sample.shape)
                sample = np.stack(np.meshgrid(sample, np.array([choice])), axis=-1).squeeze()
                sub_blk = np.concatenate(
                    [np.ones(l) * (subblk_base[choice] + k) for k, l in enumerate([c1, c2, c3, c4, c5, c6, c7, c8])])[:,
                          None]
                sample = np.concatenate([sample, sub_blk, np.ones((num_neurons_per_voxel, 1)) * i], axis=-1)
                sample_idx[num_neurons_per_voxel * count_voxel:num_neurons_per_voxel * (count_voxel + 1), :] = sample
            count_voxel += 1
    return sample_idx.astype(np.int64)

def sample_in_voxel_splitEI(aal_region, neurons_per_population_base, num_sample_voxel_per_region=1,
                                       num_neurons_per_voxel=300):
    base = neurons_per_population_base
    subblk_base = np.arange(len(aal_region)) * 2
    uni_region = np.unique(aal_region)
    num_sample_neurons = len(uni_region) * num_neurons_per_voxel * num_sample_voxel_per_region
    sample_idx = np.empty([num_sample_neurons, 4], dtype=np.int64)
    # the (, 0): neuron idx; (, 1): voxel idx, (,2): subblk(population) idx, (, 3) voxel idx belong to which brain region
    s1, s2 = int(0.8 * num_neurons_per_voxel), int(0.2 * num_neurons_per_voxel)
    count_voxel = 0
    for i in uni_region:
        print("sampling for region: ", i)
        choices = np.random.choice(np.where(aal_region == i)[0], num_sample_voxel_per_region)
        for choice in choices:
            sample1 = np.random.choice(
                np.arange(start=base[subblk_base[choice]], stop=base[subblk_base[choice] + 1], step=1), s1,
                replace=False)
            sample2 = np.random.choice(
                np.arange(start=base[subblk_base[choice] + 1], stop=base[subblk_base[choice] + 2], step=1), s2,
                replace=False)
            sample = np.concatenate([sample1, sample2])
            sub_blk = np.concatenate(
                [np.ones_like(sample1) * subblk_base[choice], np.ones_like(sample2) * (subblk_base[choice] + 1)])[:,
                      None]
            # print("sample_shape", sample.shape)
            sample = np.stack(np.meshgrid(sample, np.array([choice])), axis=-1).squeeze()
            sample = np.concatenate([sample, sub_blk, np.ones((num_neurons_per_voxel, 1)) * i], axis=-1)
            sample_idx[num_neurons_per_voxel * count_voxel:num_neurons_per_voxel * (count_voxel + 1), :] = sample
        count_voxel += 1

    return sample_idx.astype(np.int64)


def simulate_hetero_model_after_assimilation_for_rest_state(block_path, ip, noise_rate, route_path=None,
                                                            step=400, number=800, update_hyper_parameter=False,
                                                            write_path=None, **kwargs):
    """

    Parameters
    ----------
    update_hyper_parameter: bool
        whether update hyper parameter, if True, for example, in simulation after da.
    ip : str
        server ip
    step : int
        total steps to simulate
    number : int
        iteration number in one step
    write_path: str
        which dir to write
    route_path: str
        route_path for large scale accelerating simulation speed
    noise_rate: float
        noise rate in block model.
    block_path: str
        connection block setting path
    kwargs : key
        other keyword argument contain: {re_parameter_ind, hp]
    """
    start_time = time.time()
    theme = kwargs.get("theme", "whole_brain_voxel_splitEI_before_assimilation")

    os.makedirs(write_path, exist_ok=True)

    # file = h5py.File(
    #     '/public/home/ssct004t/project/yeleijun/spiking_nn_for_brain_simulation/data/jianfeng_normal/A1_1_DTI_voxel_structure_data_jianfeng.mat',
    #     'r')
    # aal_region = file['dti_AAL'][:]
    # aal_region = aal_region[0].astype(np.int32)
    aal_region = np.arange(90, dtype=np.int8)

    def conditional_info_helper(region=(42, 43, 90, 91), excited=True):
        region = np.array(region)
        if not excited:
            raise NotImplementedError
        voxels_belong_to_region = np.isin(aal_region, region).nonzero()[0]
        # they are all cortical voxels, so have 8 populations.
        populations_index_belong_to_region = [np.array([0, 2, 4, 6]) + 8 * i for i in voxels_belong_to_region]
        populations_index_belong_to_region = np.concatenate(populations_index_belong_to_region)
        populations_name_belong_to_region = populations[populations_index_belong_to_region]

        neurons_idx_belong_to_region = []
        for idx in populations_index_belong_to_region:
            neurons_idx_belong_to_region.append(
                np.arange(neurons_per_population_base[idx], neurons_per_population_base[idx + 1]))
        neurons_idx_belong_to_region = np.concatenate(neurons_idx_belong_to_region)
        return populations_name_belong_to_region.astype(np.int64), neurons_idx_belong_to_region.astype(np.int64)

    if route_path == "None":
        route_path = None
    block_model = block_gpu(ip, block_path, 0.1,
                            route_path=route_path,
                            force_rebase=False, cortical_size=1)
    # for voxel structure(splitEi) else laminar structure 10
    cortical_size = 2
    populations = block_model.subblk_id.cpu().numpy()  # such as [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, ...]
    print("\npopulations:")
    total_populations = int(block_model.total_subblks)
    total_neurons = int(block_model.total_neurons)
    neurons_per_population = block_model.neurons_per_subblk.cpu().numpy()
    neurons_per_population_base = np.add.accumulate(neurons_per_population)
    neurons_per_population_base = np.insert(neurons_per_population_base, 0, 0)
    num_voxel = populations.max() // cortical_size + 1
    assert num_voxel == len(aal_region)

    neurons_per_voxel, _ = np.histogram(populations, weights=neurons_per_population, bins=num_voxel,
                                        range=(0, num_voxel * cortical_size))
    tb = pt.PrettyTable()
    tb.field_names = ["Index", "Property", "Value", "Property-", "Value-"]
    tb.add_row([1, "name", theme, "ensembles", 1])
    tb.add_row([2, "neurons", total_neurons, "voxels", num_voxel])
    tb.add_row([3, "toal_populations", total_populations, "noise_rate", noise_rate])
    tb.add_row([4, "step", step, "number", number])
    print(tb)

    # Update noise rate to setting noise rate
    population_info = np.stack(np.meshgrid(populations, np.array([0], dtype=np.int64), indexing="ij"),
                               axis=-1).reshape((-1, 2))
    population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
    alpha = torch.ones(total_populations, device="cuda:0") * noise_rate * 1e8
    beta = torch.ones(total_populations, device="cuda:0") * 1e8
    block_model.gamma_property_by_subblk(population_info, alpha, beta, debug=False)
    bold1 = BOLD(epsilon=200, tao_s=0.8, tao_f=0.4, tao_0=1, alpha=0.2, E_0=0.8, V_0=0.02)

    for idx in np.array([10, 11, 12, 13]):
        population_info = np.stack(np.meshgrid(populations, idx, indexing="ij"),
                                   axis=-1).reshape((-1, 2))
        population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
        gamma = torch.ones(total_populations, device="cuda:0") * 5.
        block_model.gamma_property_by_subblk(population_info, gamma, gamma, debug=False)

    def _update(e_rate, i_rate):
        para_base = torch.tensor([0.02675, 0.004, 0.5034, 0.02775], dtype=torch.float32)
        population_info = np.stack(np.meshgrid(populations, np.array([10]), indexing="ij"),
                                   axis=-1).reshape((-1, 2))
        population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
        ampa = para_base[0] * e_rate
        ampa = torch.repeat_interleave(ampa, 2)
        block_model.mul_property_by_subblk(population_info, ampa.reshape(-1))

        population_info = np.stack(np.meshgrid(populations, np.array([11]), indexing="ij"),
                                   axis=-1).reshape((-1, 2))
        population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
        ampa = para_base[1] * (1 - e_rate)
        ampa = torch.repeat_interleave(ampa, 2)
        block_model.mul_property_by_subblk(population_info, ampa.reshape(-1))

        population_info = np.stack(np.meshgrid(populations, np.array([12]), indexing="ij"),
                                   axis=-1).reshape((-1, 2))
        population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
        ampa = para_base[2] * i_rate
        ampa = torch.repeat_interleave(ampa, 2)
        block_model.mul_property_by_subblk(population_info, ampa.reshape(-1))

        population_info = np.stack(np.meshgrid(populations, np.array([13]), indexing="ij"),
                                   axis=-1).reshape((-1, 2))
        population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
        ampa = para_base[3] * (1 - i_rate)
        ampa = torch.repeat_interleave(ampa, 2)
        block_model.mul_property_by_subblk(population_info, ampa.reshape(-1))

    if update_hyper_parameter:
        print("update_hyper_parameter: True")
        hp_path = kwargs["reset_hp_info"]
        hp_total = np.load(os.path.join(hp_path, "hp.npy"))
        # new method da a plande
        hp_total = hp_total[-1, :, :]
        e_rate = torch.from_numpy(hp_total[:, 0].astype(np.float32)).cuda()
        i_rate = torch.from_numpy(hp_total[:, 1].astype(np.float32)).cuda()
        _update(e_rate, i_rate)
        # tt = hp_total.shape[0]
        # hp_total = hp_total.reshape((tt, -1))
        # hp_total = np.load(os.path.join(hp_initial, "hp_new.npy"))
        # re_parameter_ind = kwargs.get("re_parameter_ind", 10)
        # assert hp_total.shape[1] == total_populations
        # hp_total = torch.from_numpy(hp_total.astype(np.float32)).cuda()
        # block_model.mul_property_by_subblk(population_info, hp_total[0, :])
        # hp_total = hp_total[1:, ]
    else:
        print("update_hyper_parameter: False")
        hp_total = None

    simulate_start = time.time()
    sample_idx = load_if_exist(sample_in_voxel_splitEI, os.path.join(write_path, "sample_idx"),
                               aal_region=aal_region,
                               neurons_per_population_base=neurons_per_population_base, num_sample_voxel_per_region=1,
                               num_neurons_per_voxel=300)

    sample_number = sample_idx.shape[0]
    print("sample_num:", sample_number)
    assert sample_idx[:, 0].max() < total_neurons
    sample_idx = torch.from_numpy(sample_idx).cuda()[:, 0]
    block_model.set_samples(sample_idx)

    load_if_exist(lambda: block_model.neurons_per_subblk.cpu().numpy(), os.path.join(write_path, "blk_size"))

    re_parameter_time = time.time()
    print(f"\nre_parameter have Done, Cost time {re_parameter_time - start_time:.2f}")

    for j in range(5):
        t13 = time.time()
        temp_fre = []
        for return_info in block_model.run(8000, freqs=True, vmean=True, sample_for_show=True):
            Freqs, vmean, spike, vi = return_info
            temp_fre.append(Freqs)
        temp_fre = torch.stack(temp_fre, dim=0)
        t14 = time.time()
        print(
            f"{j}th step number 800, mean fre: {torch.mean(torch.mean(temp_fre / block_model.neurons_per_subblk.float() * 1000, dim=0)):.1f}, cost time {t14 - t13:.1f}")

    print("\n\n\n initialize over!!!")
    step = step - 1
    bolds_out = np.zeros([step, num_voxel], dtype=np.float32)

    for ii in range((step - 1) // 50 + 1):
        nj = min(step - ii * 50, 50)
        FFreqs = np.zeros([nj, number, total_populations], dtype=np.uint32)
        Vmean = np.zeros([nj, number, total_populations], dtype=np.float32)
        Spike = np.zeros([nj, number, sample_number], dtype=np.uint8)
        Vi = np.zeros([nj, number, sample_number], dtype=np.float32)

        for j in range(nj):
            i = ii * 50 + j
            t13 = time.time()
            temp_fre = []
            temp_spike = []
            temp_vi = []
            temp_vmean = []
            bold_out1 = None
            if update_hyper_parameter:
                # block_model.mul_property_by_subblk(population_info, hp_total[i, :])
                pass
            for return_info in block_model.run(number * 10, freqs=True, vmean=True, sample_for_show=True):
                Freqs, vmean, spike, vi = return_info
                spike &= (torch.abs(vi - v_th) / 50 < 1e-5)
                temp_fre.append(Freqs)
                temp_vmean.append(vmean)
                temp_spike.append(spike)
                temp_vi.append(vi)
            temp_fre = torch.stack(temp_fre, dim=0)
            temp_vmean = torch.stack(temp_vmean, dim=0)
            temp_spike = torch.stack(temp_spike, dim=0)
            temp_vi = torch.stack(temp_vi, dim=0)
            temp_fre = temp_fre.reshape((number, 10, -1))
            temp_vmean = temp_vmean.reshape((number, 10, -1))
            temp_spike = temp_spike.reshape((number, 10, -1))
            temp_vi = temp_vi.reshape((number, 10, -1))
            temp_fre = temp_fre.sum(axis=1)
            freqs = temp_fre.cpu().numpy()
            for idxx in range(number):
                ffreqs = freqs[idxx]
                act, _ = np.histogram(populations, weights=ffreqs, bins=num_voxel, range=(0, num_voxel * cortical_size))
                act = (act / neurons_per_voxel).reshape(-1)
                act = torch.from_numpy(act).cuda()
                bold_out1 = bold1.run(torch.max(act, torch.tensor([1e-05]).type_as(act)))
            bolds_out[i, :] = torch_2_numpy(bold_out1)
            temp_vmean = temp_vmean.mean(axis=1)
            temp_spike = temp_spike.sum(axis=1)
            temp_vi = temp_vi.mean(axis=1)
            t14 = time.time()
            FFreqs[j] = torch_2_numpy(temp_fre)
            Vmean[j] = torch_2_numpy(temp_vmean)
            Spike[j] = torch_2_numpy(temp_spike)
            Vi[j] = torch_2_numpy(temp_vi)
            print(
                f"{i}th step, median fre: {torch.median(torch.mean(temp_fre / block_model.neurons_per_subblk.float() * 1000, dim=0)):.1f}, cost time {t14 - t13:.1f}")
        np.save(os.path.join(write_path, "spike_sim_{}.npy".format(ii)), Spike)
        np.save(os.path.join(write_path, "vi_sim_{}.npy".format(ii)), Vi)
        np.save(os.path.join(write_path, "freqs_sim_{}.npy".format(ii)), FFreqs)
        np.save(os.path.join(write_path, "vmean_sim_{}.npy".format(ii)), Vmean)
    np.save(os.path.join(write_path, "Bold_sim.npy"), bolds_out)

    print(f"\n simulation have Done, Cost time {time.time() - simulate_start:.2f} ")
    print(f"\n Totally have Done, Cost time {time.time() - start_time:.2f} ")
    block_model.shutdown()


def draw(log_path, freqs, block_size, sample_idx, write_path, name_path, bold_path, real_bold_path, vmean_path, vsample):
    log = np.load(log_path, )
    Freqs = np.load(freqs)
    vmean = np.load(vmean_path)
    vsample = np.load(vsample)
    if len(log.shape) > 2:
        log = log.reshape([-1, log.shape[-1]])
        vsample = vsample.reshape([-1, vsample.shape[-1]])
        Freqs = Freqs.reshape([-1, Freqs.shape[-1]])
        vmean = vmean.reshape((-1, vmean.shape[-1]))
    os.makedirs(write_path, exist_ok=True)
    block_size = np.load(block_size)
    name = loadmat(name_path)['AAL']
    bold_simulation = np.load(bold_path)
    # bold_simulation = norm(bold_simulation, dim=0)
    bold_simulation = bold_simulation[30:, ]
    voxels = bold_simulation.shape[1]
    # real_bold = get_bold_signal(real_bold_path, rest=True)[30:, :voxels]
    real_bold = None

    # the (, 0): neuron idx; (, 1): voxel idx, (,2): subblk(population) idx, (, 3) voxel idx belong to which brain region
    property = np.load(sample_idx)
    unique_voxel = np.unique(property[:, 1])

    def run_voxel(i):
        print("draw voxel: ", i)
        idx = np.where(property[:, 1] == i)[0]
        subpopu_idx = np.unique(property[idx, 2])
        sub_log = log[:, idx]
        sub_vsample = vsample[:, idx]
        region = property[idx[0], 3]
        if region < 90:
            sub_name = name[region // 2][0][0] + '-' + ['L', 'R'][region % 2]
        elif region==91:
            sub_name = 'LGN-L'
        else:
            sub_name = 'LGN-R'
        if real_bold is None:
            sub_real_bold = None
        else:
            sub_real_bold = real_bold[:, i]
        sub_sim_bold = bold_simulation[:, i]
        sub_vmean = vmean[:, subpopu_idx]

        _, split = np.unique(property[idx, 2], return_counts=True)
        split = np.add.accumulate(split)
        split = np.insert(split, 0, 0)
        index = np.unique(property[idx, 2])
        fire_rate = Freqs[:, index]
        return write_path, block_size, i, sub_log, split, sub_name, fire_rate, index, sub_real_bold, sub_sim_bold, sub_vsample, sub_vmean

    n_nlocks = [process_block(*run_voxel(i)) for i in unique_voxel]

    # TODO(luckyzlb15@163.com): table is not completed.
    table = pd.DataFrame({'Name': [b[1] for b in n_nlocks],
                          'Size': [b[0] for b in n_nlocks],
                          'Neuron Sample': ['\includegraphics[scale=0.275]{log_%i.png}' % (i,) for i in
                                            range(len(n_nlocks))],
                          'Fire Rate(Hz)': ['\includegraphics[scale=0.275]{fr_%i.png}' % (i,) for i in
                                            range(len(n_nlocks))],
                          'Fre statistics(Hz)': ['\includegraphics[scale=0.275]{statis_%i.png}' % (i,) for i in
                                                 range(len(n_nlocks))]
                          })
    column_format = '|l|c|r|c|c|c|'

    table = table.sort_values(by=['Size'], ascending=False)

    with open(os.path.join(write_path, 'chart.tex'), 'w') as f:
        f.write("""
                    \\documentclass[varwidth=25cm]{standalone}
                    \\usepackage{graphicx}
                    \\usepackage{longtable,booktabs}
                    \\usepackage{multirow}
                    \\usepackage{multicol}
                    \\begin{document}
                """)
        f.write(table.to_latex(bold_rows=True, longtable=True, multirow=True, multicolumn=True, escape=False,
                               column_format=column_format))

        f.write("""
                    \\end{document}
                """)

    print('-')


def process_block(write_path, real_block, block_i, log, split, name, fire_rate, subblk_index, bold_real=None,
                  bold_sim=None, sub_vsample=None, sub_vmean=None, time=1200, slice_window=800, stride=200):
    block_size = log.shape[-1]
    real_block_size = real_block[subblk_index]
    names = ['L2/3', 'L4', 'L5', 'L6']

    activate_idx = (log.sum(0) > 10).nonzero()[0]

    frequence = log.sum() * 1000 / log.shape[0] / activate_idx.shape[0]
    frequence_map = torch.from_numpy(log.astype(np.float32)).transpose(0, 1).unsqueeze(1)
    frequence_map = 1000 / slice_window * torch.conv1d(frequence_map, torch.ones([1, 1, slice_window]),
                                                       stride=stride).squeeze().transpose(0, 1).numpy()
    fig_fre = plt.figure(figsize=(4, 4), dpi=500)
    fig_fre.gca().hist(frequence_map.reshape([-1]), 100, density=True)
    fig_fre.gca().set_yscale('log')
    fig_fre.savefig(os.path.join(write_path, "statis_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    plt.close(fig_fre)

    fire_rate_ = (fire_rate[-time:, ::2] + fire_rate[-time:, 1::2]) / (real_block_size[::2] + real_block_size[1::2])
    fire_rate_ = np_move_avg(fire_rate_, n=10, mode="same")
    fig_frequence = plt.figure(figsize=(4, 4), dpi=500)
    ax1 = fig_frequence.add_subplot(1, 1, 1)
    ax1.grid(False)
    ax1.set_xlabel('time(ms)')
    ax1.set_ylabel('Instantaneous fr(hz)')
    for i in range(fire_rate_.shape[1]):
        ax1.plot(fire_rate_[:, i], label=names[i])
    ax1.legend(loc='best')
    fig_frequence.savefig(os.path.join(write_path, "fr_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    plt.close(fig_frequence)

    activate_idx = (log.sum(0) > 0).nonzero()[0]  # log.shape=[100*800, 300]
    cvs = []
    for i in activate_idx:
        out = log[:, i].nonzero()[0]
        if out.shape[0] >= 3:
            fire_interval = out[1:] - out[:-1]
            cvs.append(fire_interval.std() / fire_interval.mean())

    cv = np.array(cvs)
    fig_cv = plt.figure(figsize=(4, 4), dpi=500)
    fig_cv.gca().hist(cv, 100, range=(0, 2), density=True)
    fig_cv.savefig(os.path.join(write_path, "cv_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    plt.close(fig_cv)

    fig = plt.figure(figsize=(4, 4), dpi=500)
    axes = fig.add_subplot(1, 1, 1)
    color = dict(boxes='DarkGreen', whiskers='DarkOrange', medians='DarkBlue', caps='Gray')
    fire_rate_ = np_move_avg(fire_rate[-2000:, ], n=10, mode="valid")
    if len(split) > 3:
        df = pd.DataFrame(fire_rate_, columns=['2/3E', '2/3I', '4E', '4I', '5E', '5I', '6E', '6I'])
    else:
        df = pd.DataFrame(fire_rate_, columns=['E', 'I'])
    df.plot.box(vert=False, showfliers=False, widths=0.2, color=color, ax=axes)
    fig.savefig(os.path.join(write_path, "frpopu_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    valid_idx = np.where(log.mean(axis=0)>0.001)[0]
    instanous_fr = log[-2000:-800, valid_idx].mean(axis=1)
    instanous_fr = np_move_avg(instanous_fr, 10, mode="valid")
    length = len(instanous_fr)
    fig = plt.figure(figsize=(8, 4), dpi=500)
    ax1 = fig.add_subplot(1, 1, 1, frameon=False)
    ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax1.grid(False)
    ax1.set_xlabel('time(ms)')
    axes = fig.add_subplot(2, 1, 1)
    if len(split) > 3:
        sub_vmean = sub_vmean[-2000:-800, :] * np.array(
            [0.24355972, 0.05152225, 0.25995317, 0.07025761, 0.11709602, 0.03512881, 0.18501171, 0.03747072])
        sub_vmean = sub_vmean.sum(axis=-1)
        for t in range(8):
            x, y = log[-2000:-800, split[t]:split[t + 1]].nonzero()
            if t % 2 == 0:
                axes.scatter(x, y + split[t], marker=",", s=0.1, color="blue")
            else:
                axes.scatter(x, y + split[t], marker=",", s=0.1, color="red")
        names = ['L2/3', 'L4', 'L5', 'L6']
        names_loc = split[:-1][::2]
    else:
        sub_vmean = sub_vmean[-2000:-800, :] * np.array([0.8, 0.2])
        sub_vmean = sub_vmean.sum(axis=-1)
        for t in range(2):
            x, y = log[-2000:-800, split[t]:split[t + 1]].nonzero()

            if t % 2 == 0:
                axes.scatter(x, y + split[t], marker=",", s=0.1, color="blue")
            else:
                axes.scatter(x, y + split[t], marker=",", s=0.1, color="red")
        names = ["E", "I"]
        names_loc = split[:-1]
    axes.set_title("fre of spiking neurons: %.2f" % instanous_fr.mean())
    axes.set_xlim((0, length))
    axes.set_ylim((0, block_size))
    plt.yticks(names_loc, names)
    axes.invert_yaxis()
    axes.set_aspect(aspect=1)
    axes = fig.add_subplot(2, 1, 2)
    axes.plot(instanous_fr, c="black")
    fig.savefig(os.path.join(write_path, "log_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    fig_vi = plt.figure(figsize=(8, 4), dpi=500)
    ax1 = fig_vi.add_subplot(1, 1, 1, frameon=False)
    ax1.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    ax1.grid(False)
    ax1.set_xlabel('time(ms)')
    axes = fig_vi.add_subplot(2, 1, 1)
    sub_vsample = sub_vsample[-2000:-800, :]
    axes.imshow(sub_vsample.T, vmin=-65, vmax=-50, cmap='jet', origin="lower")
    axes = fig_vi.add_subplot(2, 1, 2)
    axes.plot(sub_vmean, c="r")
    fig_vi.savefig(os.path.join(write_path, "vi_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    plt.close(fig_vi)

    fig_bold = plt.figure(figsize=(4, 4), dpi=500)
    if bold_real is not None:
        fig_bold.gca().plot(np.arange(len(bold_real)), bold_real, "r-", label="real")
    fig_bold.gca().plot(np.arange(len(bold_sim)), bold_sim, "b-", label="sim")
    fig_bold.gca().set_ylim((0., 0.08))
    fig_bold.gca().legend(loc="best")
    fig_bold.gca().set_xlabel('time')
    fig_bold.gca().set_ylabel('bold')
    fig_bold.savefig(os.path.join(write_path, "bold_{}.png".format(block_i)), bbox_inches='tight', pad_inches=0)
    plt.close(fig_bold)

    return real_block_size.sum(), name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--block_path', type=str,
                        default="test/laminar_structure_whole_brain_include_subcortical/200m_structure/n40d100/single")
    parser.add_argument('--route_path', type=str,
                        default="/public/home/ssct004t/project/spiking_nn_for_brain_simulation/route_dense_10k_22703.json")
    parser.add_argument('--ip', type=str, default='11.5.4.2:50051')
    parser.add_argument("--noise_rate", type=str, default="0.01")
    parser.add_argument("--update_hyper_parameter", type=bool, default=False)
    parser.add_argument("--step", type=int, default=400)
    parser.add_argument("--number", type=int, default=800)
    parser.add_argument("--write_path", type=str, default="./")
    parser.add_argument("--reset_hp_info", type=str, default="./")
    parser.add_argument("--re_parameter_ind", type=int, default=10)

    parser.add_argument("--log_name", type=str, default="spike_sim_0.npy")
    parser.add_argument("--freqs_name", type=str, default="freqs_sim_0.npy")
    parser.add_argument("--block_size_name", type=str, default="blk_size.npy")
    parser.add_argument("--sample_idx_name", type=str, default="sample_idx.npy")
    parser.add_argument("--name_path", type=str, default="aal_names.mat")
    parser.add_argument("--theme", type=str, default="1b_v1")
    parser.add_argument("--extern_stimulus", type=bool, default=False)
    # bold_path, real_bold_path, vmean_path
    parser.add_argument("--sim_bold_path", type=str, default="Bold_sim.npy")
    parser.add_argument("--real_bold_path", type=str, default="bold_simulation.npy")
    parser.add_argument("--vmean_path", type=str, default="vmean_sim_0.npy")
    parser.add_argument("--vsample_path", type=str, default="vi_sim_0.npy")

    FLAGS, unparsed = parser.parse_known_args()
    write_file_path = os.path.join(FLAGS.write_path, FLAGS.theme, "result_file")
    simulate_hetero_model_after_assimilation_for_rest_state(FLAGS.block_path,
                                                            FLAGS.ip,
                                                            float(FLAGS.noise_rate),
                                                            route_path=FLAGS.route_path,
                                                            step=FLAGS.step,
                                                            number=FLAGS.number,
                                                            update_hyper_parameter=FLAGS.update_hyper_parameter,
                                                            write_path=write_file_path,
                                                            reset_hp_info=FLAGS.reset_hp_info,
                                                            re_parameter_ind=FLAGS.re_parameter_ind,
                                                            extern_stimulus=FLAGS.extern_stimulus,
                                                            theme=FLAGS.theme)
    write_fig_path = os.path.join(FLAGS.write_path, FLAGS.theme, "fig")
    log_path = os.path.join(write_file_path, FLAGS.log_name)
    freqs = os.path.join(write_file_path, FLAGS.freqs_name)
    block_size = os.path.join(write_file_path, FLAGS.block_size_name)
    sample_idx = os.path.join(write_file_path, FLAGS.sample_idx_name)
    simulation_bold = os.path.join(write_file_path, FLAGS.sim_bold_path)
    vmean = os.path.join(write_file_path, FLAGS.vmean_path)
    vsample = os.path.join(write_file_path, FLAGS.vsample_path)
    draw(log_path, freqs, block_size, sample_idx, write_fig_path, FLAGS.name_path,  simulation_bold, FLAGS.real_bold_path, vmean, vsample)
