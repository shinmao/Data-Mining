# python3
import re
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from collections import Counter
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

train_path = './train_drugs.data'
test_path = './test.data'

# get the max length of feature
size = []
test_size = []
# class
cs = []

train_data = [[] for i in range(800)]
test_data = [[] for i in range(350)]
# handle the train data into array
with open(train_path) as f:
    data = f.readlines()

with open(test_path) as f:
    tdata = f.readlines()

# train_data[row][0] to get drug active or inactive
# train_data[row][1][index] to get feature element
for row in range(800):
    train_data[row] = data[row].split('\t')
    train_data[row][1] = train_data[row][1].split(' ')
    train_data[row][1].remove('\n')
    size.append(len(train_data[row][1]))

# test_data[row][index] to get feature element
for row in range(350):
    test_data[row] = tdata[row].rstrip().split(' ')
    # remove the endline
    if '' in test_data[row]:
        test_data[row].remove('')
    test_size.append(len(test_data[row]))

# cs[row] to show the class of row th
for i in range(800):
    cs.append(train_data[i][0])
#print(cs)

# train_set[row][col] to show the col th of row th data
train_set = [[0]*6061 for i in range(800)]
for row in range(800):
    for col in range(6061):
        if col >= len(train_data[row][1]):
            # panda consider NaN as missing value
            train_set[row][col] = 0
        else:
            train_set[row][col] = train_data[row][1][col]

test_set = [[0]*6061 for i in range(350)]
for row in range(350):
    for col in range(6061):
        if col >= len(test_data[row]):
            test_set[row][col] = 0
        else:
            test_set[row][col] = test_data[row][col]

# create column name for dataframe
name = []
for i in range(6061):
    name.append(str(i))

# create dataframe for train data
filled_train = pd.DataFrame(train_set, columns = name, dtype = 'float')
filled_test = pd.DataFrame(test_set, columns = name, dtype = 'float')

# use PCA to do dimension reduction
scaler = StandardScaler()
train_std = scaler.fit_transform(filled_train)
test_std = scaler.transform(filled_test)
pca = PCA(0.99)
filled_train = pca.fit_transform(train_std)
filled_test = pca.transform(test_std)

# split to train set and test set from resample set
# with ratio of 0.75:0.25
# X for feature y for class
X_train, X_test, y_train, y_test = train_test_split(filled_train, cs, test_size=0.25)

# sampling with smote and edited nearest neighbors
# https://imbalanced-learn.readthedocs.io/en/stable/combine.html
model_smote = SMOTEENN(random_state=0)
train_resample, class_resample = model_smote.fit_resample(X_train, y_train)
#print(Counter(class_resample))

# use SVM-svc to train
classifier = SVC(kernel='linear')
classifier.fit(train_resample, class_resample)
pred_t = classifier.predict(filled_test)
f = open('./output.txt', mode = 'a')
for i in range(len(pred_t)):
    f.write(pred_t[i])
    f.write('\n')
f.close()