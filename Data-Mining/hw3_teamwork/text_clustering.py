import re
import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
features_path = './features.data'
test_path = './test.data'
features = []
NUM_OF_FEATURES = 126373
NUM_OF_RECORDS = 8580
test_data = [[0]* NUM_OF_FEATURES for i in range(NUM_OF_RECORDS)]

def main():
    read_data()
    test_tfidf = tf_idf()
    test_svd = svd(test_tfidf)

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
    # 100 = 0.25064564; 500 = 0.48713001; 1000 = 0.62768052; 2000 = 0.77843753; 2500 = 0.82
    svd = TruncatedSVD(n_components=2500, random_state=42)
    test_svd = svd.fit_transform(test_tfidf)
    return test_svd
    # print(test_svd.shape)
    # print(svd.explained_variance_.cumsum())

main()


    