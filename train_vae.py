import dill
import vae
import tensorflow as tf
import numpy as np
import argparse
import scipy

def main(args):
    data = scipy.sparse.load_npz(r"C:\Users\11818\Desktop\misc\CSR_test.npz").A
    train_data = vae.Dataset(data, batch_size=args.batch_size)
    model = vae.VAE(
        n_inputs=data.shape[1],
        n_latent=args.n_latent,
        n_encoder=[1000, 1000],
        n_decoder=[1000, 1000],
        visible_type='binary',
        nonlinearity=tf.nn.relu,
        weight_normalization=args.not_weight_normalization,
        importance_weighting=args.importance_weighting,
        optimizer=args.optimizer,
        learning_rate=args.lr,
        model_dir='vae'
    )

    with open('vae/model.pkl', 'wb') as f:
        dill.dump(model, f)

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

    parser.add_argument("--n_latent", default=2, type=int)
    parser.add_argument("--importance_weighting", default=False, action="store_true")
    parser.add_argument("--not_weight_normalization", default=True, action="store_false")

    parser.add_argument("--batch_size", default=1000, type=int)
    parser.add_argument("--n_samples", default=10, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--not_shuffle", default=True, action="store_false")
    parser.add_argument("--summary_steps", default=100, type=int)

    args = parser.parse_args()
    main(args)