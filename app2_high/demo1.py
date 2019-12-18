#!/usr/bin/env python3
# coding=utf-8

# Created Time: 2019-12-16 18:58:00

import scipy
import scipy.sparse as sp
import scipy.io
import xgboost
import numpy as np
from sklearn.model_selection import train_test_split

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
model.load_model("model_8001")  # 加载已存储的模型

# 改变for website in ["webs"]:中“ ”的值可以分段并处理json文件，如果需要对一个文件处理可以用注释的部分，将单个json文件呢名称放在[]内部。
# -----以下部分作测试用-----
import json
import pickle

X_test = []
Y_test = []
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


for website in ["webs"]:
    with open(website + ".json", "r") as f:
        for line in f:
            doc = json.loads(line)
            X_test.append(doc)

X_test = vectorizer.transform(X_test)
# -----------------------

print(X_test.shape)

# 下面是对输入的数据进行处理与demo展示无关，mask 80 端口headers server，medadata
X = X_test
X = np.array(X.todense())
print(X.shape)
mask = np.ones((5987,))
unwanted = [4550, 4551, 4552, 4553, 4554, 4555, 4556, 4557, 4558, 4559, 4560, 4561, 4562, 4563, 4564, 17, 18, 19, 20,
            285, 286, 287, 288, 601, 602, 603, 604, 605, 606, 607, 1557, 1558, 1559, 1560, 1561, 1562, 1563, 1564, 1565,
            1566, 1567, 1568, 1569, 1570, 1571, 1572, 1573, 1574, 3196, 3197, 3198, 3199, 4656, 4657, 4658, 4659, 4660,
            4661, 4662, 4663, 4664, 4665, 4666, 4667, 4668, 4669, 4670, 4671, 4672, 4673, 4674, 4675, 4676, 4677, 4678,
            4679]
for col in unwanted:
    X[:, col] = False
# print(X.shape)
X = sp.csr_matrix(X)
# print(X.shape) 
wanted = [4550, 4551, 4552, 4553, 4554, 4555, 4556, 4557, 4558, 4559, 4560, 4561, 4562, 4563, 4564]
for index in wanted:
    mask[index] = 0
Y_test = X_test[:, mask == 0]
Y_test = np.array(Y_test.todense())
print("Input shape is:", Y_test.shape)
# print(Y_test)
mask = (np.sum(Y_test.astype(np.uint8), axis=1) >= 1)
Y_test = Y_test[mask]
# print(Y_test)
out = np.argmax(Y_test, axis=1)
# print(out)
out = sp.csr_matrix(out)
sp.save_npz("test.npz", out)

X_test = X
Y_test = sp.load_npz("test.npz")
# print(Y_test.shape)
# print(Y_test)

Y_test = np.array(Y_test.todense()).T
dtest = xgboost.DMatrix(X_test, label=Y_test)
data = xgboost.DMatrix(X_test)
preds = model.predict(dtest)
# print(preds.shape)
preds = preds.reshape(-1, 1)
print("Prediction shape is:", preds.shape)
print("Label shape is:", Y_test.shape)
# print(preds)
# print(Y_test)

# 输出错误率与准确率
error_rate = np.sum(preds != Y_test) * 1.0 / Y_test.shape[0]
acc_of_test = 1 - error_rate
print('Test error using softmax = {}'.format(error_rate))
print('Test ACC = {}'.format(acc_of_test))

# 计算并输出fpr和tpr，不需要可以直接注释
ACC = []
FT = []
# number of examples and number of classes
rows, cols = len(preds), int(preds.max())

for col in range(cols + 1):
    print("Processing the {} class".format(col))
    pred = (preds == col).astype(np.uint8)
    label = (Y_test == col).astype(np.uint8)
    cnts = np.zeros((2, 2), dtype=np.int32)
    for i in range(rows):
        cnts[pred[i], label[i]] += 1
    print(cnts)
    ACC.append((100 * (cnts[1, 1] + cnts[0, 0]) / cnts.sum()))
    FPR = (100 * cnts[0, 1] / (cnts[0, 0] + cnts[0, 1]))
    TPR = (100 * cnts[1, 1] / (cnts[1, 0] + cnts[1, 1]))
    FT.append((FPR, TPR))
    print("ACC = %.2f%% " % ACC[-1])
    print("FPR = %.2f%%" % FPR)
    print("TPR = %.2f%%" % TPR)
    np.save("ACC.npy", ACC)
    np.save("FT.npy", FT)

# mask = np.ones((5987,))
# wanted=[4550,	4551,	4552,	4553,	4554,	4555,	4556,	4557,	4558,	4559,	4560,	4561,	4562,	4563,	4564]
# for index in wanted:
#     mask[index]=0
# Y_test=X_test[:,mask==0]
# Y_test = np.array(Y_test.todense())
# print("Input shape is:", Y_test.shape)
# print(Y_test)
# mask = (np.sum(Y_test.astype(np.uint8), axis=1) >= 1)
# Y_test = Y_test[mask]
# print(Y_test)
# out = np.argmax(Y_test, axis=1)
# print(out)
# out = sp.csr_matrix(out)
# sp.save_npz("test.npz",out)

# Y_test=sp.load_npz("test.npz")
# print(Y_test.shape)

# Y_test = np.array(Y_test.todense()).T

# print(Y_test.max())
# print(Y_test)
# print(Y_test.shape)
# data = xgboost.DMatrix(X_test)
# prediction = model.predict(data)
# print(prediction)
