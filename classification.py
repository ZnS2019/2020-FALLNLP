import numpy as np

import time
import os
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split, KFold
import BYS
import KNN

wp = os.getcwd()
cross_validation =  False

print('loading data')
xtf_data = np.loadtxt(os.path.join(wp,'tfdata.csv'), delimiter=',')
print('x_tfdata loaded')
y_vecs = np.loadtxt(os.path.join(wp,'y_vecs.csv'), delimiter=',')
print('y_vecs loaded')
t1 = time.time()

x_train, x_test,y_train,y_test = train_test_split(xtf_data, y_vecs, test_size = 0.1, random_state = 0)

def bayesian(x_train, x_test,y_train,y_test):
    PC,PTermC = BYS.train(x_train,y_train,class_nums=20)
    y_pred = BYS.predict(x_test, PC,PTermC)
    print(classification_report(y_test,y_pred))
    return accuracy_score(y_test,y_pred)

def knn(x_train, x_test,y_train,y_test):
    y_pred = KNN.predict(x_train,x_test,y_train)
    print(classification_report(y_test,y_pred))
    return accuracy_score(y_test,y_pred)

if cross_validation:
    kf = KFold(n_splits=5, shuffle=True)
    indexes = range(len(xtf_data))
    x_trains = y_trains = x_tests = y_tests = []
    for train, test in kf.split(indexes):
        x_trains.append(xtf_data[train])
        y_trains.append(y_vecs[train])
        x_tests.append(xtf_data[test])
        y_tests.append(y_vecs[test])

    acc = 0
    for i in range(5):
        acc += bayesian(x_trains[i], x_tests[i], y_trains[i], y_tests[i])
    print('mean accuracy of bayesian classification is : ', acc/5)

    acc = 0
    for i in range(5):
        acc += knn(x_trains[i], x_tests[i], y_trains[i], y_tests[i])
    print('mean accuracy of knn classification is : ', acc/5)
else:
    print('----------bayesian---------')
    bayesian(x_train,x_test, y_train, y_test)
    print('------------knn-----------')
    knn(x_train,x_test, y_train, y_test)

t2 = time.time()
print('time = ', t2-t1)