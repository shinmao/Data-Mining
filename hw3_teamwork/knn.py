# python3
from sklearn.decomposition import PCA
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer 
import numpy as np
from math import log

input_path = "./input.dat"
feature_path = "./feature.dat"

sparse_matrix = [[0]*126373 for i in range(3147)]
feature = []
tf_matrix = [[0]*126373 for i in range(3147)]
idf_matrix = [0 for i in range(126373)]
tfidf = [[0]*126373 for i in range(3147)]

def read_sparse():
    with open(input_path) as f:
        data = f.readlines()
    f.close()
    for i in range(3147):
        temp = []
        temp = data[i].rstrip().split(" ")
        for j in range(0, len(temp), 2):
            idx = int(temp[j])
            # the feature id starts from 1 in the input.dat
            sparse_matrix[i][idx-1] = int(temp[j+1])

def read_feature():
    with open(feature_path) as f:
        data = f.readlines()
        for i in range(126373):
            feature.append(data[i].rstrip())
    f.close()

# help calculate for idf
def help_counter():
    counter = [0 for i in range(126373)]
    for i in range(3147):
        for j in range(126373):
            if sparse_matrix[i][j] == 0:
                counter[j] += 0
            else:
                counter[j] += 1
    return counter

def cal_tfidf():
    # calculate tf
    for i in range(3147):
        sum = 0
        for j in range(126373):
            sum += sparse_matrix[i][j]
        for j in range(126373):
            tf_matrix[i][j] = sparse_matrix[i][j] / sum
    counter = help_counter()
    # calculate idf
    for x in range(126373):
        idf_matrix[x] = log(float(3147) / ( counter[x] + 1 ) )
    # calculate tf-idf
    for a in range(3147):
        for b in range(126373):
            tfidf[a][b] = tf_matrix[a][b] * idf_matrix[b]

def dimension(x, n):
    pca = PCA(n_components=n)
    return pca.fit_transform(x)

def main():
    read_sparse()
    read_feature()
    # transform np array
    #sparse = np.array(sparse_matrix)
    cal_tfidf()
    '''
    with open('test.txt', 'w+') as f:
        f.write(str(tfidf[0]))
    f.close()
    '''


if __name__ == "__main__":
    main()