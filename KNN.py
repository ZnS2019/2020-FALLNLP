#-*- coding:utf-8 -*-
import numpy as np

def dis(a,b):
    return np.linalg.norm(a-b)

def predict(x_train, x_test, y_train, k=15):
    y_pred = []
    for i in range(len(x_test)):
        x = x_test[i]
        dis = [(k,dis(x,y)) for y in x_train]
        dis = sorted(dis, key=lambda p: p[1],reverse=False)
        dis = dis[:k]
        classes = np.array([d[0] for d in dis])
        Ci = np.argmax(np.bincount(classes))
        y_pred.append(Ci)
    return y_pred
