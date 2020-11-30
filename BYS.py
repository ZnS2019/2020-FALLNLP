#-*- coding:utf-8 -*-
import numpy as np
def train(train_data, train_label, class_nums): 
    files, terms = train_data.shape
    data_2d = {}
    for i in range(class_nums):
        data_2d[i] = []
    for j in range(files):
        data_2d[train_label[j]].append(train_data[j])
    num = [len(data_2d[i] for i in range(class_nums))]
    PTermC = np.zeros((class_nums,terms))
    for i in range(class_nums):
        PTermC = np.sum(data_2d[i],axis=1)/np.sum(data_2d[i])
        PC = num[i]/files
    PTermC = np.mat(PTermC)
    return PC,PTermC

def predict(test_tf_data, PC, PTermC):
    class_nums = len(PC)
    shape = test_tf_data.shape
    # shape pc = 1, classnums
    # shape ptermc = classnums, terms
    pred = []
    for i in range(shape[0]):
        Pmax = 0
        Ci = -1
        tf = test_tf_data[i]
        TFi = np.mat(tf)
        # shape = 1, terms
        P = PC + TFi*PTermC.T
        for k in range(class_nums):
            if P[k]>Pmax:
                Ci, Pmax = k, P[k]
        pred.append(Ci)
    return pred 
            