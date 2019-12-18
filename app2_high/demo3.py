#!/usr/bin/env python3
# coding=utf-8

# Created Time: 2019-12-16 18:58:00

import scipy
import scipy.sparse as sp
import scipy.io
import xgboost
import numpy as np
from sklearn.model_selection import train_test_split
import json
import pickle


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

    X_test = vectorizer.transform([content_dict, ])
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
    try:
        print("prediction is {} label is {} token is {}".format(int(preds[0]), Y_test[0, 0], token_dict[Y_test[0, 0]]))
    except:
        print("we cannot predict such a label")
    # 得到的preds就是预测的server
