import argparse
import os

import numpy as np
import torch
from mpi4py import MPI

from brain_block.block import block
from brain_block.random_initialize import connect_for_block


def run_simulation(block_path, res_path):
    os.makedirs(res_path, exist_ok=True)
    # for d100 \math
    #       34 * ampa + 250 * nmda = 1
    #       2 * gabaA + 36 * gabaB = 1
    # and for d500 \math
    #       172 * ampa + 600 * nmda = 1
    #       10 * gabaA + 180 * gabaB = 1
    # and for d300 \math
    #       102 * ampa + 500 * nmda = 1
    #       6 * gabaA + 108 * gabaB = 1

    # ampa_contribution = np.linspace(0., 1., num=50, endpoint=True)
    # gabaA_contribution = np.linspace(0., 1., num=50, endpoint=True)
    # contribution = np.stack(np.meshgrid(ampa_contribution, gabaA_contribution, indexing='ij'), axis=-1).reshape(
    #     (-1, 2))
    # ampa_contribution = contribution[:, 0]
    # gabaA_contribution = contribution[:, 1]
    # ampa = ampa_contribution / 102
    # nmda = (1 - ampa_contribution) / 500
    # gabaA = gabaA_contribution / 6
    # gabaB = (1 - gabaA_contribution) / 108
    # para = np.stack([ampa, nmda, gabaA, gabaB], axis=1)

    # new for d100
    # 92 * ampa + 1856 * nmda = 1.8
    # 24 * gabaA + 420 * gabaB = 1.1

    x = y = 80
    ampa = np.linspace(0.022 / 2, 0.022 / 1, x)
    gabaA = np.linspace(0., 0.5 / 2, y)
    param = np.stack(np.meshgrid(ampa, gabaA, indexing="ij"), axis=-1).reshape((-1, 2))
    nmda = (0.022 - param[:, 0] * 1) / 5
    gabaB = (0.5 - param[:, 1] * 1) / 18
    para = np.stack([param[:, 0] / 1.2, nmda / 1.1, param[:, 1] / 1.35, gabaB / 1.35], axis=1)
    para = para.astype(np.float32)
    total = para.shape[0]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    for i in range(rank, total, size):
        property, w_uij = connect_for_block(os.path.join(block_path, 'd500'))
        property[:, (10, 11, 12, 13)] = torch.from_numpy(para[i])
        property = property.cuda()
        w_uij = w_uij.cuda()
        B = block(
            node_property=property,
            w_uij=w_uij,
            delta_t=0.1,
        )

        # run
        log = []
        for time in range(40000):
            B.run(noise_rate=0.0005, isolated=False)
            if time >= 10000:
                log.append(B.active.data.cpu().numpy())
        log = np.array(log, dtype=np.uint8)[:, 1400:1600]
        log = log.reshape((-1, 10, 200))
        log = log.sum(axis=1)
        np.save(os.path.join(res_path, f'log_{i}.npy'), log)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument("--write_path", type=str, default="Data")
    parser.add_argument("--block_path", type=str, default="Data")
    args = parser.parse_args()
    run_simulation(args.block_path, args.write_path)