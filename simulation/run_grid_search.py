import os
import numpy as np
import torch
from mpi4py import MPI
import argparse
from brain_block.random_initialize import connect_for_block
from brain_block.block import block


def run_simulation(block_path, res_path):
    os.makedirs(res_path, exist_ok=True)
    # \math:
    #       40 * ampa + 200 * nmda = 1
    #       2 * gabaA + 50 * gabaB = 1
    ampa_contribution = np.linspace(0.5, 1, num=50, endpoint=True)
    gabaA_contribution = np.linspace(0., 0.5, num=50, endpoint=True)
    contribution = np.stack(np.meshgrid(ampa_contribution, gabaA_contribution, indexing='ij'), axis=-1).reshape((-1, 2))
    ampa_contribution = contribution[:, 0]
    gabaA_contribution = contribution[:, 1]
    ampa = ampa_contribution / 40
    nmda = (1 - ampa_contribution) / 200
    gabaA = gabaA_contribution / 2
    gabaB = (1 - gabaA_contribution) / 50
    para =np.stack([ampa, nmda, gabaA, gabaB], axis=1)
    para = para.astype(np.float32)
    total = para.shape[0]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    for i in range(rank, total, size):
        property, w_uij = connect_for_block(os.path.join(block_path, 'single'))
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
        for time in range(30000):
            B.run(noise_rate=0.0003, isolated=False)
            if time >= 10000:
                log.append(B.active.data.cpu().numpy())
        log = np.array(log, dtype=np.uint8)[:, 1400:1650]
        log = log.reshape((-1, 10, 250))
        log = log.sum(axis=1)
        np.save(os.path.join(res_path, f'log_{i}.npy'), log)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run simulation")
    parser.add_argument("--write_path", type=str, default="Data")
    parser.add_argument("--block_path", type=str, default="Data")
    args = parser.parse_args()
    run_simulation(args.block_path, args.write_path)