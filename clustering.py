import numpy as np
from sklearn.cluster import KMeans

import time
import os
import KMEANS
import HIERACHICAL

wp = os.getcwd()

print('loading data')
x_data = np.loadtxt(os.path.join(wp,'data.csv'), delimiter=',')
print('x_data loaded')
y_vecs = np.loadtxt(os.path.join(wp,'y_vecs.csv'), delimiter=',')
print('y_vecs loaded')
t1 = time.time()

from sklearn.metrics import adjusted_rand_score
# y_pred1 = KMeans(n_clusters=20).fit_predict(x_data)
y_pred1 = KMEANS.kmeans(x_data,20)
y_pred2 = HIERACHICAL.hirachical(x_data, HIERACHICAL.dist_avg, 20)
print('k means 聚类的兰德指数为 : ', adjusted_rand_score(y_vecs, y_pred1))
print('层次聚类的兰德指数为 : ', adjusted_rand_score(y_vecs, y_pred2))

t2 = time.time()
print('time = ', t2-t1)