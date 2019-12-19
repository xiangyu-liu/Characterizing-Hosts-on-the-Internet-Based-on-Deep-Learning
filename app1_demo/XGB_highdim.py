import xgboost
import numpy as np
import json
import pickle


def demo1_highdim(content_dict):
    model = xgboost.Booster({'nthread': 4})  # 初始化模型
    model.load_model(r"C:\Users\11818\Desktop\RL\Code\vae\app1_demo\model_XGB")  # 加载已存储的模型

    with open(r"C:\Users\11818\Desktop\RL\Code\vae\app1_demo\vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    X = vectorizer.transform(content_dict)  # 生成高维向量

    # 对测试集进行预测
    X = xgboost.DMatrix(X)
    prediction = model.predict(X)[0]  # 返回：坏主机的概率
    print("###begin test high embedding###")
    print("Pr[Bad guy] = %.2f" % prediction)
    if prediction >= 0.9:
        print("Beware of this host!")
    elif prediction <= 0.1:
        print("Don't worry, it's harmless.")
