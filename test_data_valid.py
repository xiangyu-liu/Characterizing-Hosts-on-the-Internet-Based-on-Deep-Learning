import pickle
import dill
import numpy as np
from scipy import sparse
import json
from matplotlib import pyplot as plt


def main():
    black_data = sparse.load_npz(r"C:\Users\11818\Desktop\misc\data_5.0\data_blacklist_5.0.npz").A
    white_data = sparse.load_npz(r"C:\Users\11818\Desktop\misc\data_5.0\data_no_black_5.0.npz").A
    with open(r"C:\Users\11818\Desktop\RL\Code\vae\data_no_black_5.1.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    plt.figure()
    for i, feature in enumerate(vectorizer.feature_names_):
        if "server" in feature:
            print("{} {}".format(i, feature))
    data = np.concatenate([black_data, white_data], axis=0)
    for i in np.arange(start=1539, stop=1548):
        print("{} {} {}".format(vectorizer.feature_names_[i], np.sum(black_data[:, i]), np.sum(white_data[:, i])))
    print("{} {}".format(np.sum(black_data[:, 1539:1548]), np.sum(white_data[:, 1539:1548])))
    ax1 = plt.subplot(121)
    plt.hist(np.sum(black_data, axis=1), bins=100)
    ax2 = plt.subplot(122)
    plt.hist(np.sum(white_data, axis=1), bins=100)
    plt.show()
    google_embedding = vectorizer.transform(
        [json.load(open(r"C:\Users\11818\Desktop\RL\Code\vae\test_data\google.json"))])
    baidu_embedding = vectorizer.transform(
        [json.load(open(r"C:\Users\11818\Desktop\RL\Code\vae\test_data\baidu.json"))])
    github_embedding = vectorizer.transform(
        [json.load(open(r"C:\Users\11818\Desktop\RL\Code\vae\test_data\github.json"))])

    print(google_embedding.sum(), baidu_embedding.sum(), github_embedding.sum())
    np.save("google.npy", google_embedding)
    np.save("baidu.npy", baidu_embedding)
    np.save("github.npy", github_embedding)


if __name__ == '__main__':
    main()
