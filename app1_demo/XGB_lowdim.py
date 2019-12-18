import xgboost
import numpy as np
import json
import pickle
import tensorflow
import dill
import vae

def demo1_lowdim(content_dict):
    model = xgboost.Booster({'nthread': 4})  # 初始化模型
    model.load_model(r"C:\Users\11818\Desktop\RL\Code\vae\app1_demo\model_XGB_embed")  # 加载已存储的模型

    with open(r"C:\Users\11818\Desktop\RL\Code\vae\app1_demo\vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    X = vectorizer.transform([content_dict, ])  # 生成高维向量

    '''
    === NOTE ===
    请将VAE降维放在这里
    '''

    with open(r"C:\Users\11818\Desktop\RL\Code\vae\latest_model\model.pkl", 'rb') as f:
        vae_model = dill.load(f)
    vae_model.model_dir = r"C:\Users\11818\Desktop\RL\Code\vae\latest_model"
    X = vae.Dataset(X, batch_size=1)
    Z_mean, Z_sd = vae_model.evaluate(X, tensors=['z_mean', 'z_sd'])
    X = Z_mean[0]
    X = X[np.newaxis]

    # 对测试集进行预测
    X = xgboost.DMatrix(X)
    prediction = model.predict(X)[0]  # 返回：坏主机的概率
    print("Pr[Bad guy] = %.2f" % prediction)
    if prediction >= 0.9:
        print("Beware of this host!")
    elif prediction <= 0.1:
        print("Don't worry, it's harmless.")
