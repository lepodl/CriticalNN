import argparse
import os

import numpy as np
import torch
import sparse
from mpi4py import MPI
from scipy.io import loadmat

from brain_block.random_initialize import connect_for_multi_sparse_block, merge_dti_distributation_block


def add_laminar_cortex_model(conn_prob, gm):
    canonical_voxel = True
    if not canonical_voxel:
        lcm_connect_prob = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 0, 3554, 804, 881, 45, 431, 0, 136, 0, 1020],
                                     [0, 0, 1778, 532, 456, 29, 217, 0, 69, 0, 396],
                                     [0, 0, 417, 84, 1070, 690, 79, 93, 1686, 0, 1489],
                                     [0, 0, 168, 41, 628, 538, 36, 0, 1028, 0, 790],
                                     [0, 0, 2550, 176, 765, 99, 621, 596, 363, 7, 1591],
                                     [0, 0, 1357, 76, 380, 32, 375, 403, 129, 0, 214],
                                     [0, 0, 643, 46, 549, 196, 327, 126, 925, 597, 2609],
                                     [0, 0, 80, 8, 92, 3, 159, 11, 76, 499, 1794]], dtype=np.float64
                                    )

        lcm_gm = np.array([0, 0,
                           33.8 * 78, 33.8 * 22,
                           34.9 * 80, 34.9 * 20,
                           7.6 * 82, 7.6 * 18,
                           22.1 * 83, 22.1 * 17], dtype=np.float64)  # ignore the L1 neurons
    else:
        # 4:1 setting
        lcm_connect_prob = np.array([[0.55, 0.2, 0.25],
                                     [0.55, 0.2, 0.25]], dtype=np.float64)
        lcm_gm = np.array([0.8, 0.2], dtype=np.float64)

    lcm_gm /= lcm_gm.sum()

    syna_nums_in_lcm = lcm_connect_prob.sum(1) * lcm_gm
    lcm_degree_scale = syna_nums_in_lcm / syna_nums_in_lcm.sum() / lcm_gm
    lcm_degree_scale = np.where(np.isnan(lcm_degree_scale), 0, lcm_degree_scale)
    lcm_connect_prob /= lcm_connect_prob.sum(axis=1, keepdims=True)

    if conn_prob.shape[0] == 1:
        conn_prob[:, :] = 1
    else:
        conn_prob[np.diag_indices(conn_prob.shape[0])] = 0
        conn_prob = conn_prob / conn_prob.sum(axis=1, keepdims=True)

    out_gm = (gm[:, None] * lcm_gm[None, :]).reshape([-1])
    out_degree_scale = np.broadcast_to(lcm_degree_scale[None, :], [gm.shape[0], lcm_gm.shape[0]]).reshape([-1])
    conn_prob = sparse.COO(conn_prob)
    # only e5 is allowed to output.
    corrds1 = np.empty([4, conn_prob.coords.shape[1] * lcm_connect_prob.shape[0]], dtype=np.int64)
    if not canonical_voxel:
        corrds1[3, :] = 6
    else:
        corrds1[3, :] = 0
    corrds1[(0, 2), :] = np.broadcast_to(conn_prob.coords[:, :, None],
                                         [2, conn_prob.coords.shape[1], lcm_connect_prob.shape[0]]).reshape([2, -1])
    corrds1[(1), :] = np.broadcast_to(np.arange(lcm_connect_prob.shape[0], dtype=np.int64)[None, :],
                                      [conn_prob.coords.shape[1], lcm_connect_prob.shape[0]]).reshape([1, -1])

    data1 = (conn_prob.data[:, None] * lcm_connect_prob[:, -1]).reshape([-1])

    lcm_connect_prob_inner = sparse.COO(lcm_connect_prob[:, :-1])
    corrds2 = np.empty([4, conn_prob.shape[0] * lcm_connect_prob_inner.data.shape[0]], dtype=np.int64)
    corrds2[0, :] = np.broadcast_to(np.arange(conn_prob.shape[0], dtype=np.int64)[:, None],
                                    [conn_prob.shape[0], lcm_connect_prob_inner.data.shape[0]]).reshape([-1])
    corrds2[2, :] = corrds2[0, :]
    corrds2[(1, 3), :] = np.broadcast_to(lcm_connect_prob_inner.coords[:, None, :],
                                         [2, conn_prob.shape[0], lcm_connect_prob_inner.coords.shape[1]]).reshape(
        [2, -1])
    data2 = np.broadcast_to(lcm_connect_prob_inner.data[None, :],
                            [conn_prob.shape[0], lcm_connect_prob_inner.coords.shape[1]]).reshape([-1])

    out_conn_prob = sparse.COO(coords=np.concatenate([corrds1, corrds2], axis=1),
                               data=np.concatenate([data1, data2], axis=0),
                               shape=[conn_prob.shape[0], lcm_connect_prob.shape[0], conn_prob.shape[1],
                                      lcm_connect_prob.shape[1] - 1])

    out_conn_prob = out_conn_prob.reshape((conn_prob.shape[0] * lcm_connect_prob.shape[0],
                                           conn_prob.shape[1] * (lcm_connect_prob.shape[1] - 1)))
    if conn_prob.shape[0] == 1:
        out_conn_prob = out_conn_prob / out_conn_prob.sum(axis=1, keepdims=True)
    return out_conn_prob, out_gm, out_degree_scale


def make_small_block(write_path, initial_parameter=(0.00495148, 0.0009899, 0.08417509, 0.00458287)):
    prob = torch.tensor([[1.]])
    tau_ui = (8, 40, 10, 50)
    if os.path.exists(os.path.join(write_path, 'single', 'block_0.npz')):
        print("remove")
        os.remove(os.path.join(write_path, 'single', 'block_0.npz'))
    connect_for_multi_sparse_block(prob, {'g_Li': 0.03,
                                          'g_ui': initial_parameter,
                                          "V_reset": -65,
                                          'tao_ui': tau_ui},
                                   E_number=int(1.6e3), I_number=int(4e2), degree=300, init_min=0,
                                   init_max=1, perfix=write_path)
    print("Done")


def make_multi_size_block(write_path, initial_parameter=(0.00408739, 0.00059394, 0.03636364, 0.00353535)):
    size_list = np.logspace(3, 8, num=50, endpoint=True)
    total = size_list.shape[0]
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    for i in range(rank, total, size):
        current_dir = os.path.join(write_path, f"size_{i}")
        os.makedirs(current_dir, exist_ok=True)
        if size_list[i] <= int(1e6):
            blocks = 4
        else:
            blocks = 4 * int(size_list[i] / 4e6 + 1)
        blocks = np.minimum(blocks, 100)
        block_size = np.array([0.8, 0.2])
        conn_prob = np.array([[0.8, 0.2],
                              [0.8, 0.2]])
        degree = np.array([500, 500], dtype=np.uint16)
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
                                       number=int(blocks),
                                       dtype=["single"],
                                       debug_block_path=None)


def make_degree_distribution_block(write_path, initial_parameter=(0.00364106, 0.00074747, 0.02777778, 0.00401235),
                                   size=int(5e3)):
    current_dir = os.path.join(write_path, "sigma_200")
    conn_prob = np.array([[0.8, 0.2],
                          [0.8, 0.2]])
    block_size = np.array([0.8, 0.2])
    degree = np.array([500, 500], dtype=np.uint16)
    kwords = [{"V_th": -50,
               "V_reset": -65,
               'g_Li': 0.03,
               'g_ui': initial_parameter,
               'tao_ui': (8, 40, 10, 50),
               "E_number": int(max(b * size, 0)) if i % 2 == 0 else 0,
               "I_number": int(max(b * size, 0)) if i % 2 == 1 else 0, }
              for i, b in enumerate(block_size)]
    conn = connect_for_multi_sparse_block(conn_prob, kwords,
                                          degree=degree,
                                          init_min=1,
                                          init_max=1,
                                          perfix=None,
                                          dtype=['single'],
                                          split_EI=True, degree_distribution="normal", sigma=200, minum_degree=10)
    merge_dti_distributation_block(conn, current_dir,
                                   MPI_rank=None,
                                   number=1,
                                   dtype=["single"],
                                   debug_block_path=None)


def make_dti_network(write_path):
    os.makedirs(write_path, exist_ok=True)
    file = loadmat('../data/DTI_T1_92ROI.mat')
    block_size = file['t1_roi'][:90, 0]
    block_size /= block_size.sum(0)
    dti = np.float32(file['weight_dti'])[:90, :90]
    dti[np.diag_indices_from(dti)] = 0.
    gui = np.array((0.0017797816801139058,
                    0.0011564625850340139,
                    0.02040816326530612,
                    0.004421768707482994))

    scale = int(1e7)
    minmum_neurons_for_block = 1000

    prob = dti.copy()
    # prob[np.diag_indices_from(prob)] = (prob.sum(1) * inner_component) / (1 - inner_component)
    # prob /= np.sum(prob, axis=1, keepdims=True)
    conn_prob, block_size, degree_scale = add_laminar_cortex_model(prob, block_size)

    kwords = [{"V_th": -50,
               "V_reset": -65,
               'g_Li': 0.03,
               'g_ui': gui,
               'tao_ui': (8, 40, 10, 50),
               "E_number": int(max(b * scale, minmum_neurons_for_block)) if i % 2 == 0 else 0,
               "I_number": int(max(b * scale, minmum_neurons_for_block)) if i % 2 == 1 else 0, }
              for i, b in enumerate(block_size)]
    conn = connect_for_multi_sparse_block(conn_prob, kwords,
                                          degree=300,
                                          init_min=0,
                                          init_max=1,
                                          perfix=None,
                                          dtype=['single'],
                                          split_EI=True)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for i in range(rank, 12, size):
        merge_dti_distributation_block(conn, write_path,
                                       MPI_rank=i,
                                       number=12,
                                       dtype=["single"],
                                       debug_block_path=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate block")
    parser.add_argument("--write_path", type=str, default="../data/degree_distribution_d500")
    args = parser.parse_args()
    # make_small_block(args.write_path)
    # make_multi_size_block(args.write_path)
    # make_degree_distribution_block(args.write_path)
    make_dti_network(args.write_path)