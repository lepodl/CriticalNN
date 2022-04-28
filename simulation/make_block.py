import os
import torch
import argparse
from brain_block.random_initialize import connect_for_multi_sparse_block


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
                                   E_number=int(1.6e3), I_number=int(4e2), degree=400, init_min=0,
                                   init_max=1, perfix=write_path)
    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Data Assimilation")
    parser.add_argument("--write_path", type=str, default="Data")
    args = parser.parse_args()
    make_small_block(args.write_path)
