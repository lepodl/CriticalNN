import argparse
import os

import numpy as np
import torch
from cuda0606.python.dist_blockwrapper_pytorch import BlockWrapper as block_gpu


def load_if_exist(func, *args, **kwargs):
    path = os.path.join(*args)
    if os.path.exists(path + ".npy"):
        out = np.load(path + ".npy")
    else:
        out = func(**kwargs)
        np.save(path, out)
    return out


def torch_2_numpy(u, is_cuda=True):
    assert isinstance(u, torch.Tensor)
    if is_cuda:
        return u.cpu().numpy()
    else:
        return u.numpy()


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


def run_simulation(ip, block_path, write_path, idx=0):
    # defaulat critical parameter
    # 0.0115, 0.0020, 0.2517, 0.0111 subcritical parameter for d100
    # ampa_contribution = np.linspace(0.5, 1., num=50, endpoint=True)
    # gabaA_contribution = np.linspace(0., 0.5, num=50, endpoint=True)
    # contribution = np.stack(np.meshgrid(ampa_contribution, gabaA_contribution, indexing='ij'), axis=-1).reshape(
    #     (-1, 2))
    # ampa_contribution = contribution[:, 0]
    # gabaA_contribution = contribution[:, 1]
    # ampa = ampa_contribution / 102
    # nmda = (1 - ampa_contribution) / 500
    # gabaA = gabaA_contribution / 6
    # gabaB = (1 - gabaA_contribution) / 108
    # para = np.stack([ampa, nmda, gabaA, gabaB], axis=1)
    # sample_idx = np.arange(5, 50, 10, dtype=np.int8)
    # sample_idx = np.stack(np.meshgrid(sample_idx, sample_idx, indexing="ij"), axis=-1).reshape((-1, 2))
    # a, b = sample_idx[idx]
    # specific_gui = para[a * 50 + b]
    # critical_param = np.load('vertical_line_param.npy')

    specific_gui = np.array([0.01921519, 0.00055696, 0.11708861, 0.02127286])
    os.makedirs(write_path, exist_ok=True)
    v_th = -50
    aal_region = np.array([0])
    block_model = block_gpu(ip, block_path, 0.1,
                            route_path=None,
                            force_rebase=False, cortical_size=1)

    total_neurons = int(block_model.total_neurons)
    neurons_per_population = block_model.neurons_per_subblk.cpu().numpy()
    neurons_per_population_base = np.add.accumulate(neurons_per_population)
    neurons_per_population_base = np.insert(neurons_per_population_base, 0, 0)
    populations = block_model.subblk_id.cpu().numpy()
    total_populations = int(block_model.total_subblks)

    # update noise rate
    population_info = np.stack(np.meshgrid(populations, np.array([0], dtype=np.int64), indexing="ij"),
                               axis=-1).reshape((-1, 2))
    population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
    alpha = torch.ones(total_populations, device="cuda:0") * 0.0003 * 1e8
    beta = torch.ones(total_populations, device="cuda:0") * 1e8
    block_model.gamma_property_by_subblk(population_info, alpha, beta, debug=False)

    def _update_gui(gui=(0.0115, 0.0020, 0.2517, 0.0111)):
        for i, idx in enumerate(np.arange(10, 14)):
            population_info = np.stack(np.meshgrid(populations, idx, indexing="ij"),
                                       axis=-1).reshape((-1, 2))
            population_info = torch.from_numpy(population_info.astype(np.int64)).cuda()
            alpha = torch.ones(total_populations, device="cuda:0") * gui[i] * 1e8
            beta = torch.ones(total_populations, device="cuda:0") * 1e8
            block_model.gamma_property_by_subblk(population_info, alpha, beta, debug=False)
    # _update_gui(gui=specific_gui)

    sample_idx = load_if_exist(sample_in_voxel_splitEI, os.path.join(write_path, "sample_idx"),
                               aal_region=aal_region,
                               neurons_per_population_base=neurons_per_population_base, num_sample_voxel_per_region=1,
                               num_neurons_per_voxel=500)

    sample_number = sample_idx.shape[0]
    print("sample_num:", sample_number)
    assert sample_idx[:, 0].max() < total_neurons
    sample_idx = torch.from_numpy(sample_idx).cuda()[:, 0]
    block_model.set_samples(sample_idx)

    def _index2param(param1, param2, num_grids, index):
        a1, b1, c1 = param1
        a2, b2, c2 = param2
        x, y = num_grids
        ampa = np.linspace(c1 / a1 / 2, c1 / a1, x)
        nmda = (c1 - ampa * a1) / b1
        gabaA = np.linspace(0., c2 / a2 / 2, y)
        gabaB = (c2 - gabaA * a2) / b2
        param = np.stack([ampa[index[:, 0]], nmda[index[:, 0]], gabaA[index[:, 1]], gabaB[index[:, 1]]], axis=0)
        return param

    center = (54, 35)
    # update parameter for each channel and each neuron
    for radius in np.arange(1, 15, 1):
        para_info = np.zeros((total_neurons, 5), dtype=np.int64)
        para_info[:, 0] = np.arange(total_neurons, dtype=np.int64)
        para_info[:, 1:] = np.tile(np.array([10, 11, 12, 13]).astype(np.int64), (total_neurons, 1))
        para_info = torch.from_numpy(para_info).cuda()
        grid_index = np.random.normal(loc=center, scale=(radius, radius), size=(total_neurons, 2))
        grid_index = grid_index.astype(np.int64)
        grid_index = np.where(grid_index>79, 79, grid_index)
        grid_index = np.where(grid_index<0, 0, grid_index)
        param = _index2param((1, 5, 0.022), (1, 18, 0.5), (80, 80), grid_index)
        param = torch.from_numpy(param.astype(np.float32)).cuda()
        for i, idx in enumerate(np.arange(10, 14)):
            block_model.update_property(para_info[:, [0, i + 1]], param[i])

        Spike = np.zeros((10, 800, sample_number), dtype=np.uint8)
        _ = block_model.run(16000, freqs=True, vmean=False, sample_for_show=False)
        for j in range(10):
            temp_spike = []
            for return_info in block_model.run(8000, freqs=False, vmean=False, sample_for_show=True):
                spike, vi = return_info
                spike &= (torch.abs(vi - v_th) / 50 < 1e-6)
                temp_spike.append(spike)
            temp_spike = torch.stack(temp_spike, dim=0)
            temp_spike = temp_spike.reshape((800, 10, -1))
            temp_spike = temp_spike.sum(axis=1)
            Spike[j] = torch_2_numpy(temp_spike)
        np.save(os.path.join(write_path, f"spike_radius_{radius}.npy"), Spike)
    block_model.shutdown()
    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CUDA Data siulation")
    parser.add_argument("--ip", type=str, default="11.5.4.2:50051")
    parser.add_argument("--block_path", type=str, default="blokc_path/single")
    parser.add_argument("--write_path", type=str, default="write_path")
    parser.add_argument("--idx", type=str, default="0")
    args = parser.parse_args()
    run_simulation(args.ip, args.block_path, args.write_path, int(args.idx))