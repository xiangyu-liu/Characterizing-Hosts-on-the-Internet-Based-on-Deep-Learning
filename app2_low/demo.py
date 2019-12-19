import dill
import scipy
import scipy.sparse as sp
import scipy.io
import xgboost
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import json
import pickle

import vae


def demo2_low(content_dict):
    params = {
        'eta': 0.1,
        'max_depth': 6,
        'booster': 'gbtree',
        'silent': 1,
        'objective': 'multi:softmax',
        'min_child_weight': 3,
        'num_class': 15,
        'nthread': 4
    }
    token_dict = ["10", "15", "16", "apache", "centos", "httpapi", "iis", "lighttpd", "microsoft", "nginx", "openssl",
                  "php", "server", "ubuntu", "webs"]
    model = xgboost.Booster({'nthread': 4})  # 初始化模型
    model.load_model(r"C:\Users\11818\Desktop\RL\Code\vae\app2_low\model_8003")  # 加载已存储的模型

    X_test = []
    Y_test = []
    with open(r"C:\Users\11818\Desktop\RL\Code\vae\app1_demo\vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    X_test = vectorizer.transform(content_dict)
    # -----------------------

    # 下面是对输入的数据进行处理与demo展示无关，mask 80 端口headers server，medadata
    X = X_test
    X = np.array(X.todense())
    # print(X.shape)
    mask = np.ones((5987,))
    unwanted = [4550, 4551, 4552, 4553, 4554, 4555, 4556, 4557, 4558, 4559, 4560, 4561, 4562, 4563, 4564]
    for col in unwanted:
        X[:, col] = False
    # print(X.shape)
    X = sp.csr_matrix(X)
    # print(X.shape)

    # 针对model8001，需要修改line21加载存储的模型
    # unwanted=[4550, 4551,	4552,	4553,	4554,	4555,	4556,	4557,	4558,	4559,	4560,	4561,	4562,	4563,	4564,   17,	18,	19,	20,	285,	286,	287,	288,	601,	602,	603,	604,	605,	606,	607,	1557,	1558,	1559,	1560,	1561,	1562,	1563,	1564,	1565,	1566,	1567,	1568,	1569,	1570,	1571,	1572,	1573,	1574,	3196,	3197,	3198,	3199,	4656,	4657,	4658,	4659,	4660,	4661,	4662,	4663,	4664,	4665,	4666,	4667,	4668,	4669,	4670,	4671,	4672,	4673,	4674,	4675,	4676,	4677,	4678,	4679]
    # for col in unwanted:
    #    X[:,col]=False
    # print(X.shape)
    # X = sp.csr_matrix(X)

    # 针对model8002，需要修改line21加载存储的模型
    # unwanted = [1501, 1502, 1503, 1504, 1505, 1506, 1507, 1508, 4550, 4551, 4552, 4553, 4554, 4555, 4556, 4557, 4558,
    #             4559,
    #             4560, 4561, 4562, 4563, 4564, 17, 18, 19, 20, 285, 286, 287, 288, 601, 602, 603, 604, 605, 606, 607,
    #             1557,
    #             1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574,
    #             3196,
    #             3197, 3198, 3199, 4656, 4657, 4658, 4659, 4660, 4661, 4662, 4663, 4664, 4665, 4666, 4667, 4668, 4669,
    #             4670,
    #             4671, 4672, 4673, 4674, 4675, 4676, 4677, 4678, 4679]
    # for col in unwanted:
    #     X[:, col] = False
    # # print(X.shape)
    # X = sp.csr_matrix(X)

    wanted = [4550, 4551, 4552, 4553, 4554, 4555, 4556, 4557, 4558, 4559, 4560, 4561, 4562, 4563, 4564]
    for index in wanted:
        mask[index] = 0
    Y_test = X_test[:, mask == 0]
    Y_test = np.array(Y_test.todense())

    # print("Input shape is:", Y_test.shape)
    # print(Y_test)
    mask = (np.sum(Y_test.astype(np.uint8), axis=1) >= 1)
    Y_test = Y_test[mask]
    # print(Y_test)
    out = np.argmax(Y_test, axis=1)
    # print(out)
    out = sp.csr_matrix(out)
    sp.save_npz("X_test1.npz", X)
    sp.save_npz("Y_test.npz", out)

    # -----以下部分作加入VAE训练针对X_test1.npz-----
    with open(r"C:\Users\11818\Desktop\RL\Code\vae\latest_model\model.pkl", 'rb') as f:
        vae_model = dill.load(f)
    vae_model.model_dir = r"C:\Users\11818\Desktop\RL\Code\vae\latest_model"
    X = sp.load_npz("X_test1.npz")
    X = vae.Dataset(X, batch_size=1)
    Z_mean, Z_sd = vae_model.evaluate(X, tensors=['z_mean', 'z_sd'])
    X = np.concatenate(Z_mean)
    X = X.reshape((-1, 50))

    # ------------------------------------------

    # data = np.load('X_test1_mean.npz')
    data = X
    X_test = data.reshape(-1, 50)
    Y_test = sp.load_npz("Y_test.npz")
    Y_test = np.array(Y_test.todense()).T

    dtest = xgboost.DMatrix(X_test, label=Y_test)
    data = xgboost.DMatrix(X_test)
    preds = model.predict(dtest)
    preds = preds.reshape(-1, 1)
    print("###begin to test low embedding###")
    for i in range(len(content_dict)):
        try:
            print("prediction is {}; label is {}; token is {}".format(int(preds[i]), Y_test[i, 0],
                                                                      token_dict[Y_test[i, 0]]))
        except:
            print("we cannot predict such a label")

    error_rate = np.sum(preds != Y_test) * 1.0 / Y_test.shape[0]
    acc_of_test = 1 - error_rate
    print('Test error using softmax = {}'.format(error_rate))
    print('Test ACC = {}'.format(acc_of_test))


def demo2_high(content_dict):
    token_dict = ["10", "15", "16", "apache", "centos", "httpapi", "iis", "lighttpd", "microsoft", "nginx", "openssl",
                  "php", "server", "ubuntu", "webs"]
    params = {
        'eta': 0.1,
        'max_depth': 6,
        'booster': 'gbtree',
        'silent': 1,
        'objective': 'multi:softmax',
        'min_child_weight': 3,
        'num_class': 15,
        'nthread': 4
    }

    model = xgboost.Booster({'nthread': 4})  # 初始化模型
    model.load_model(r"C:\Users\11818\Desktop\RL\Code\vae\app2_high\model_8003")  # 加载已存储的模型

    with open(r"C:\Users\11818\Desktop\RL\Code\vae\app1_demo\vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    X_test = vectorizer.transform(content_dict)
    X = X_test
    X = np.array(X.todense())
    mask = np.ones((5987,))
    unwanted = [4550, 4551, 4552, 4553, 4554, 4555, 4556, 4557, 4558, 4559, 4560, 4561, 4562, 4563, 4564]
    for col in unwanted:
        X[:, col] = False
    X = sp.csr_matrix(X)

    for index in unwanted:
        mask[index] = 0
    Y_test = X_test[:, mask == 0]
    Y_test = np.array(Y_test.todense())
    mask = (np.sum(Y_test.astype(np.uint8), axis=1) >= 1)
    Y_test = Y_test[mask]
    out = np.argmax(Y_test, axis=1)
    out = sp.csr_matrix(out)

    X_test = X
    Y_test = out

    Y_test = np.array(Y_test.todense()).T
    dtest = xgboost.DMatrix(X_test, label=Y_test)
    data = xgboost.DMatrix(X_test)
    preds = model.predict(dtest)
    print("###begin to test high embedding###")
    for i in range(len(content_dict)):
        try:
            print("prediction is {}; label is {}; token is {}".format(int(preds[i]), Y_test[i, 0],
                                                                      token_dict[Y_test[i, 0]]))
        except:
            print("we cannot predict such a label")

    error_rate = np.sum(preds != Y_test) * 1.0 / Y_test.shape[0]
    acc_of_test = 1 - error_rate
    print('Test error using softmax = {}'.format(error_rate))
    print('Test ACC = {}'.format(acc_of_test))

