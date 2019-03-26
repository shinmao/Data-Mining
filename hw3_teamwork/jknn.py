import re
import pandas as pd
import numpy as np
import scipy
import math
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from copy import deepcopy
from random import randint
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import silhouette_score


features_path = './feature.dat'
test_path = './input.dat'
features = []
NUM_OF_FEATURES = 126373
NUM_OF_RECORDS = 8580
test_data = [[0]* NUM_OF_FEATURES for i in range(NUM_OF_RECORDS)]
K = 7

def main():
    read_data()
    test_tfidf = tf_idf()
    test_svd = svd(test_tfidf)
    # kmeans = KMeans(n_clusters=7, random_state=42)
    # clusters = kmeans.fit_predict(test_svd)
    # with open('output1.txt', 'w') as f:
    #     for i in clusters:
    #         f.write("%s\n" % str(int(clusters[i])+1))
    
    k_means(test_svd)

def read_data():
    global features, test_data
    test_buf = []
    # read features data
    with open(features_path) as f:
        features = f.read().splitlines()
    
    with open(test_path) as f:
        data = f.read().splitlines()
        for row in range(NUM_OF_RECORDS):
            test_buf.append(data[row].split())
    for row in range(NUM_OF_RECORDS):
        l = len(test_buf[row])
        for col in range(0,l,2):
            feature_id = int(test_buf[row][col]) - 1
            count = int(test_buf[row][col+1])
            test_data[row][feature_id] = count
  
def tf_idf():
    tfidf_transformer = TfidfTransformer().fit(test_data)
    test_tfidf = tfidf_transformer.transform(test_data)
    return test_tfidf

def svd(test_tfidf):
    # In particular, truncated SVD works on term count/tf-idf matrices 
    # it is known as latent semantic analysis (LSA).
    # For LSA, a value of 100 is recommended.
    # 100 = 0.25064564; 500 = 0.48713001; 1000 = 0.62768052; 2000 = 0.77843753; 2500 = 0.82; 5000: 0.85
    svd = TruncatedSVD(n_components=100, random_state=42)
    test_svd = svd.fit_transform(test_tfidf)
    # print(svd.explained_variance_.cumsum())
    return test_svd
    # print(test_svd.shape)
    

def initialize(test_svd, K):
    c1 = randint(0,NUM_OF_RECORDS)
    C = [test_svd[c1]]
    for k in range(1, K):
        D2 = scipy.array([min([scipy.inner(c-x,c-x) for c in C]) for x in test_svd])
        probs = D2/D2.sum()
        cumprobs = probs.cumsum()
        r = scipy.rand()
        for j,p in enumerate(cumprobs):
            if r < p:
                i = j
                break
        C.append(test_svd[i])
    return C

def k_means(test_svd):
    print("k means")
    # test_svd = normalize(test_svd)

    # test_svd.shape[1] = # features after reduction
    num_features = test_svd.shape[1]
    centers_old = np.zeros((K,num_features))
    # random K centers
    rand_points = initialize(test_svd,K)
    # print(rand_points)
    centers_new = np.array(rand_points)
    
    # store clusters for each record
    clusters = np.zeros(NUM_OF_RECORDS)
    # store distance between each record and centers
    distance = np.zeros((NUM_OF_RECORDS,K))
    # print(centers_new[0])
    # check if there's any center was changed
    error = np.linalg.norm(centers_new - centers_old)
    j = 0
    while j<300:
        for row in range(NUM_OF_RECORDS):
            for i in range(K):
                # compute distance between each record and center 
                distance[row][i] = get_cosine_sim(test_svd[row],centers_new[i])
        # assign the points to the closest cluster
        # argmin: get the index of minimum distance
        clusters = np.argmax(distance, axis = 1)
        # print(clusters)
        centers_old = deepcopy(centers_new)
        # recompute the new k mean centers
        for i in range(K):
            centers_new[i] = np.mean(test_svd[clusters == i],axis =0)
        print(centers_new)
        # check if there's any center was changed
        error = np.linalg.norm(centers_new - centers_old)
        j+=1

    with open('output.txt', 'w') as f:
        for i in clusters:
            f.write("%s\n" % str(int(clusters[i])+1))

    
def get_corrleation(v1,v2):
    corr = np.corrcoef(v1, v2)
    return corr[0][1]

def get_cosine_sim(v1, v2):
    
    return np.dot(v1,v2)/(np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))

def sil_score(data, predicted):
    # score between -1 and 1, the bigger the better
    score = silhouette_score(data, predicted, metric='cosine')
    return score

main()



    