import argparse
import os

import numpy as np
import torch
from mpi4py import MPI

from brain_block.random_initialize import connect_for_multi_sparse_block, merge_dti_distributation_block


def make_small_block(write_path, initial_parameter=(0.0115, 0.0020, 0.2517, 0.0111)):
    prob = torch.tensor([[1.]])
    tau_ui = (8, 40, 10, 50)
    if os.path.exists(os.path.join(write_path, 'single', 'block_0.npz')):
        print("remove")
        os.remove(os.path.join(write_path, 'single', 'block_0.npz'))
    connect_for_multi_sparse_block(prob, {'g_Li': 0.03,
                                          'g_ui': initial_parameter,
                                          "V_reset": -65,
                                          'tao_ui': tau_ui},
                                   E_number=int(1.6e3), I_number=int(4e2), degree=100, init_min=0,
                                   init_max=1, perfix=write_path)
    print("Done")


def make_multi_size_block(write_path, initial_parameter=(0.020042676767676766, 0.00043232323232323246, 0.16424060606060603, 0.01869621212121212)):
    size_list = np.logspace(3, 8, num=50, endpoint=True)
    total = size_list.shape[0]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    for i in range(rank, total, size):
        current_dir = os.path.join(write_path, f"size_{i}")
        os.makedirs(current_dir, exist_ok=True)
        if size_list[i] <= int(5e6):
            blocks = 4
        else:
            blocks = 4 * int(size_list[i] / 2e7 + 1)
        blocks = np.minimum(blocks, 20)
        block_size = np.array([0.8, 0.2])
        conn_prob = np.array([[0.8, 0.2],
                              [0.8, 0.2]])
        degree = np.array([100, 100], dtype=np.uint16)
        kwords = [{"V_th": -50,
                   "V_reset": -65,
                   'g_Li': 0.03,
                   'g_ui': initial_parameter,
                   'tao_ui': (8, 40, 10, 50),
                   "E_number": int(max(b * size_list[i], 0)) if i % 2 == 0 else 0,
                   "I_number": int(max(b * size_list[i], 0)) if i % 2 == 1 else 0, }
                  for i, b in enumerate(block_size)]
        conn = connect_for_multi_sparse_block(conn_prob, kwords,
                                              degree=degree,
                                              init_min=0,
                                              init_max=1,
                                              perfix=None,
                                              dtype=['single'],
                                              split_EI=True)
        merge_dti_distributation_block(conn, current_dir,
                                       MPI_rank=None,
                                       number=blocks,
                                       dtype=["single"],
                                       debug_block_path=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate block")
    parser.add_argument("--write_path", type=str, default="Data")
    args = parser.parse_args()
    # make_small_block(args.write_path)
    make_multi_size_block(args.write_path)
