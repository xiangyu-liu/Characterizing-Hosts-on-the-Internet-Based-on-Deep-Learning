import argparse

import vae
import dill
import os
import scipy.sparse
import numpy as np


def main(args):
    args.path = "/newNAS/Workspaces/DRLGroup/xiangyuliu/Computer-Network/log/50-1000-True-True/epochs=1000 batch_size=1000 n_samples=10 lr=0.001/run1"
    with open(os.path.join(args.path, "model.pkl"), 'rb') as f:
        model = dill.load(f)
    print("begin to evaluate")

    data = scipy.sparse.load_npz("/newNAS/Workspaces/DRLGroup/xiangyuliu/delete_about_headers_80_metadata.npz").A
    data = vae.Dataset(data, batch_size=args.batch_size)
    Z_mean, Z_sd = model.evaluate(data, tensors=['z_mean', 'z_sd'])
    Z_mean = np.concatenate(Z_mean, axis=0)
    Z_sd = np.concatenate(Z_sd, axis=0)
    np.save(os.path.join(args.path, "new_mean.npy"), Z_mean)
    np.save(os.path.join(args.path, "new_sd.npy"), Z_sd)

    # data_blacklist = scipy.sparse.load_npz("/newNAS/Workspaces/DRLGroup/xiangyuliu/delete_headers_about_metadata_80.npz").A
    # data_blacklist = vae.Dataset(data_blacklist, batch_size=args.batch_size)
    # Z_mean, Z_sd = model.evaluate(data_blacklist, tensors=['z_mean', 'z_sd'])
    # Z_mean = np.concatenate(Z_mean, axis=0)
    # Z_sd = np.concatenate(Z_sd, axis=0)
    # np.save(os.path.join(args.path, "headers_about_metadata_mean.npy"), Z_mean)
    # np.save(os.path.join(args.path, "headers_about_metadata_sd.npy"), Z_sd)
    #
    # data_blacklist = scipy.sparse.load_npz("/newNAS/Workspaces/DRLGroup/xiangyuliu/delete_headers_80_only80.npz").A
    # data_blacklist = vae.Dataset(data_blacklist, batch_size=args.batch_size)
    # Z_mean, Z_sd = model.evaluate(data_blacklist, tensors=['z_mean', 'z_sd'])
    # Z_mean = np.concatenate(Z_mean, axis=0)
    # Z_sd = np.concatenate(Z_sd, axis=0)
    # np.save(os.path.join(args.path, "only80_mean.npy"), Z_mean)
    # np.save(os.path.join(args.path, "only80_sd.npy"), Z_sd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Evaluation')

    parser.add_argument("--path", type=str)
    parser.add_argument("--hidden_units", default=1000, type=int)
    parser.add_argument("--batch_size", default=1000, type=int)

    args = parser.parse_args()
    main(args)
