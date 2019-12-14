import dill
from pathlib import Path
import os
import vae
import tensorflow as tf
import numpy as np
import argparse
import scipy.sparse


def main(args):
    model_dir = Path('./log') / (str(args.n_latent) + "-" + str(args.hidden_units)) / "epochs={} batch_size={} n_samples={} lr={}".format(args.epochs,
                                                                                                         args.batch_size,
                                                                                                         args.n_samples,
                                                                                                         args.lr)
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    log_dir = model_dir / curr_run
    os.makedirs(log_dir)
    print("making directory", str(log_dir))

    data = scipy.sparse.load_npz("/newNAS/Workspaces/DRLGroup/xiangyuliu/CSR_no_blacklist.1.1.npz").A
    data_blacklist = scipy.sparse.load_npz("/newNAS/Workspaces/DRLGroup/xiangyuliu/CSR_blacklist.1.0.npz").A
    data = np.concatenate([data, data_blacklist], axis=0)
    print(data.shape)
    validation = np.random.choice(data.shape[0], size=1000)
    train = [i for i in range(data.shape[0]) if not (i in validation)]
    train_data = data[train]
    validation_data = data[validation]
    train_data = vae.Dataset(train_data, batch_size=args.batch_size)
    validation_data = vae.Dataset(validation_data, batch_size=args.batch_size)

    model = vae.VAE(
        n_inputs=data.shape[1],
        n_latent=args.n_latent,
        n_encoder=[args.hidden_units, args.hidden_units],
        n_decoder=[args.hidden_units, args.hidden_units],
        visible_type='binary',
        nonlinearity=tf.nn.relu,
        weight_normalization=args.not_weight_normalization,
        importance_weighting=args.importance_weighting,
        optimizer=args.optimizer,
        learning_rate=args.lr,
        model_dir=str(log_dir)
    )

    with open(log_dir/"model.pkl", 'wb') as f:
        dill.dump(model, f)
    print("begin to fit")

    model.fit(
        train_data,
        validation_data=validation_data,
        epochs=args.epochs,
        shuffle=args.not_shuffle,
        summary_steps=args.summary_steps,
        init_feed_dict={'batch_size': args.batch_size},
        batch_size=args.batch_size,
        n_samples=args.n_samples
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VAE Training')

    parser.add_argument("--n_latent", default=50, type=int)
    parser.add_argument("--hidden_units", default=1000, type=int)
    parser.add_argument("--importance_weighting", default=False, action="store_true")
    parser.add_argument("--not_weight_normalization", default=True, action="store_false")


    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--n_samples", default=10, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--not_shuffle", default=True, action="store_false")
    parser.add_argument("--summary_steps", default=100, type=int)

    args = parser.parse_args()
    main(args)
