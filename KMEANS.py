 #-*- coding:utf-8 -*-
import numpy as np

def rand_choose(data, k):
	shape = data.shape
	indexes = np.random.randint(shape[0], size=(k))
	elmts = [data[i] for i in indexes]
	return elmts

def distance(a, b):
    return np.sqrt(np.sum(np.square(a-b)))

def kmeans(data, k):  
	m = data.shape[0]  
	centroids = rand_choose(data, k)  
	counter = 0  
	cluster = np.zeros((m, 1))  
	dist = {'0': [], '1': [], '2': [], '3': [], '4': [], '5': [], '6': [], '7': [], '8': [], '9': [],'10': [],'11': [], '12': [], '13': [], '14': [], '15': [], '16': [], '17': [], '18': [], '19': []}  
	while counter <= 10:  
		dist_cluster = dist  
		cluster_change = False  
		for i in range(m):
			min_index = -1  
			min_dist = float('inf')  
			for j in range(k):  
				dist = distance(centroids[j, :], data[i, :])  
				if dist < min_dist:  
					min_dist = dist  
					min_index = j  
			if cluster[i] != min_index:  
				cluster_change = True  
			cluster[i] = min_index  
			dist_cluster[str(min_index)].append(i)  
		for c in range(k):  
			data_of_c = data[cluster[:, 0] == c]  
			if len(data_of_c != 0):  
				centroids[c, :] = np.mean(data_of_c, axis=0)  
		counter += 1  
		print(counter)  
		if not cluster_change:  
			break  
	return centroids, cluster, counter, dist_cluster