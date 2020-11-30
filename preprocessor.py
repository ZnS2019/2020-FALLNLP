#!/usr/bin/env python
# coding: utf-8
# # 20_newsgroups_clusering

import os
import numpy as np
import re
log_interval = 100
max_vcb = 20000

# ## 1.导入数据库

corpus = int(input('type 1 to use 20_newsgroups or 0 to use mini_newsgroups:  '))
if corpus:
    corpus = '20_newsgroups'
else:
    corpus = 'mini_newsgroups'

cwd = os.getcwd()
corpus_path = os.path.join(cwd, corpus)
print(corpus_path)
text_groups = os.listdir(corpus_path)
# text_groups

# 文件路径生成
# 20个类中每个文件的地址
subpaths = []
y_vecs = []
print('initializing file paths ...')
for i in range(len(text_groups)):
    subdir = text_groups[i]
    subpath = os.path.join(corpus_path,subdir) 
    subpaths.append([os.path.join(subpath,f) for f in os.listdir(subpath)])
    y_vecs +=([i]*len(os.listdir(subpath)))

print(sum(y_vecs))
print(len(subpaths),[len(path)for path in subpaths])

# ## 2. 文本预处理
# define stop words, mainly copy from internet
stop_words = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours', 
            'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their', 
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once', 
            'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you', 
            'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will', 
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be', 
            'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself', 
            'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both', 
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn', 
            'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about', 
            'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn', 
            'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which',
            # 以下是自己新加的
            'us', 'may']

rec = re.compile(r"(?u)\b\w\w+\b")                  
def preprocess(f):
    start = 0
    t = f.readlines()
    for i in range(len(t)):
        if (t[i]=='\n'):
            start = i+1
            break
    text = t[start:]
    text = ' '.join(text)
    text = text.lower()
    text = rec.findall(text)
    text = [w for w in text if (w not in stop_words) and (len(w)>2)]
    return text

textlists = []
totalfile = 0
doc_idx = []
for i in range(20):
    textlis_c = []
    for j in range(len(subpaths[i])):
        subpath = subpaths[i][j]
        f = open(subpath, 'r', encoding='unicode_escape', errors='ignore')
        words= preprocess(f)
        textlis_c.append(words)
    textlists.append(textlis_c)
    doc_idx.append(range(totalfile,len(textlis_c)+totalfile))
    totalfile += len(textlis_c)

print('counting the vocabulary....')


# 3. 词频统计
vocabulary = {}
vocab_df = {}
for i in range(20):
    for j in range(len(textlists[i])):
        words = textlists[i][j]
        for word in words:
            if word in vocabulary:
                vocabulary[word] += 1
                vocab_df[word].append(doc_idx[i][j])
            else:
                vocabulary[word] = 1
                vocab_df[word] = [doc_idx[i][j]]

print('initializing my tf-idf...')
# 4. 去重->df
for word in vocab_df:
    vocab_df[word] = list(set(vocab_df[word]))
    vocab_df[word] = len(vocab_df[word])

print(len(vocabulary))
# a =  input('type in your max vcb length : ')
# if a :
#     max_vcb = int(a)
# else:
#     max_vcb = len(vocabulary)
# # 高频的前max vcb个词作为主要特征
# vocabulary = sorted(vocabulary.items(), key = lambda d:d[1], reverse=True)
# vocabulary = vocabulary[:max_vcb]

# voc = {}
# for item in vocabulary:
#     voc[item[0]] = item[1]
# vocabulary = voc

# # print(vocabulary)
# for i, word in enumerate(vocabulary):
#     if not i in range(max_vcb):
#         vocabulary.pop(word)
#         vocab_df.pop(word)

max_vcb = len(vocabulary)

# IDF 向量    
IDF = np.zeros(max_vcb)
vcb_idx = {} 
total_tf = 0
for i,word in enumerate(vocabulary):
    IDF[i] = np.log(totalfile/vocab_df[word])
    vcb_idx[word] = i
    total_tf += vocabulary[word]
IDF = np.mat(IDF)
IDF.reshape(max_vcb,1)

# 5. tf矩阵生成
TF = np.zeros((totalfile,max_vcb))
TF = np.mat(TF)
print('initiating tf mat...')
for i in range(20):
    for j in range(len(textlists[i])):
        words = textlists[i][j]
        for word in words:
            if word in vocabulary:
                TF[doc_idx[i][j],vcb_idx[word]]+=1

from sklearn.decomposition import PCA
pca_f = PCA(n_components=0.97, svd_solver='full')

# tf 加一平滑
TF = TF + np.mat(np.ones((totalfile, max_vcb)))
TF/= (total_tf+max_vcb)
# pca_f.fit(TF)
# TF_f = pca_f.transform(TF)
np.savetxt('tfdata.csv', TF, delimiter=',')
np.savetxt('y_vecs.csv', y_vecs, delimiter=',')

print('TF SAVED')

# 6. 新的特征矩阵
X = np.multiply(TF, IDF)

# 7.降维
pca_f.fit(X)
X_f = pca_f.transform(X)
# save vecs of newsgroup

np.savetxt('data.csv', X, delimiter=',')
