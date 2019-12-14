import argparse

import vae
import dill
import os
import scipy.sparse
import numpy as np


def main(args):
    args.path = "/newNAS/Workspaces/DRLGroup/xiangyuliu/Computer-Network/log/501000/epochs=100 batch_size=1000 n_samples=10 lr=0.001/run4/"
    with open(os.path.join(args.path, "model.pkl"), 'rb') as f:
        model = dill.load(f)
    print("begin to evaluate")

    data = scipy.sparse.load_npz("/newNAS/Workspaces/DRLGroup/xiangyuliu/CSR_no_blacklist.1.1.npz").A
    data = vae.Dataset(data, batch_size=args.batch_size)
    Z_mean, Z_sd = model.evaluate(data, tensors=['z_mean', 'z_sd'])
    Z_mean = np.concatenate(Z_mean, axis=0).reshape(shape=(-1, Z_mean[0].shape[0]))
    Z_sd = np.concatenate(Z_sd, axis=0).reshape(shape=(-1, Z_sd[0].shape[0]))
    np.save(os.path.join(args.path, "mean.npy"), Z_mean)
    np.save(os.path.join(args.path, "sd.npy"), Z_sd)

    data_blacklist = scipy.sparse.load_npz("/newNAS/Workspaces/DRLGroup/xiangyuliu/CSR_blacklist.1.0.npz").A
    data_blacklist = vae.Dataset(data_blacklist, batch_size=args.batch_size)
    Z_mean, Z_sd = model.evaluate(data_blacklist, tensors=['z_mean', 'z_sd'])
    Z_mean = np.concatenate(Z_mean, axis=0).reshape(shape=(-1, Z_mean[0].shape[0]))
    Z_sd = np.concatenate(Z_sd, axis=0).reshape(shape=(-1, Z_sd[0].shape[0]))
    np.save(os.path.join(args.path, "black_mean.npy"), Z_mean)
    np.save(os.path.join(args.path, "black_sd.npy"), Z_sd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Evaluation')

    parser.add_argument("--path", type=str)
    parser.add_argument("--hidden_units", default=1000, type=int)
    parser.add_argument("--batch_size", default=1000, type=int)

    args = parser.parse_args()
    main(args)
