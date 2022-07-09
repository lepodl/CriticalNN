import argparse
import os
import numpy as np
import glob
import matplotlib.pyplot as plt


def main(dir, index=(1,), mode='single'):
    if mode=="single":
        for idx in index:
            file_path = os.path.join(dir, f"log_{idx}.npy")
            log = np.load(file_path)
            log = log.reshape((-1, log.shape[-1]))
            fig = plt.figure(figsize=(8, 3), dpi=300)
            fig.gca().scatter(*log[-2000:, ].nonzero(), marker=".", s=0.5)
            fig.gca().set_xlabel("time(ms)")
            fig.gca().set_ylabel("neuronal index")
            fig.savefig(f"raster_{idx}.png")
            plt.close()
    else:
        names = glob.glob(os.path.join(dir, "*.npy"))
        for name in names:
            log = np.load(name)
            log = log.reshape((-1, log.shape[-1]))
            fig = plt.figure(figsize=(8, 3), dpi=300)
            fig.gca().scatter(*log[-2000:, ].nonzero(), marker=".", s=0.5)
            fig.gca().set_xlabel("time(ms)")
            fig.gca().set_ylabel("neuronal index")
            fig.savefig(f"raster_{name[-12:-4]}.png")
    print("done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot")
    parser.add_argument("-d", "--dir", type=str, default="../data/grid_search")
    parser.add_argument("-i", "--index", type=int, nargs='+', help="index")
    parser.add_argument("-m", "--mode", type=str, default="single")
    args = parser.parse_args()
    main(args.dir, args.index, args.mode)
