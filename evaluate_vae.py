import argparse

import vae
import dill
import os
import scipy.sparse
import numpy as np


def main(args):
    data = scipy.sparse.load_npz(r"C:\Users\11818\Desktop\misc\CSR_test.npz").A
    data = vae.Dataset(data, batch_size=args.batch_size)
    with open(os.path.join(args.path, "model.pkl"), 'rb') as f:
        model = dill.load(f)
    Z_mean, Z_sd = model.evaluate(data, tensors=['z_mean', 'z_sd'])
    Z_mean = np.concatenate(Z_mean, axis=0)
    Z_sd = np.concatenate(Z_sd, axis=0)
    np.save(os.path.join(args.path, "mean.npy"), Z_mean)
    np.save(os.path.join(args.path, "sd.npy"), Z_sd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Evaluation')

    parser.add_argument("--path", type=str)
    parser.add_argument("--batch_size", default=1000, type=int)

    args = parser.parse_args()
    main(args)
