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

train_path = './train_drugs.data'
test_path = './test.data'

# get the max length of feature
size = []
# class
cs = []

train_data = [[] for i in range(800)]
# handle the train data into array
with open(train_path) as f:
    data = f.readlines()

# train_data[row][0] to get drug active or inactive
# train_data[row][1][index] to get feature element
for row in range(800):
    train_data[row] = data[row].split('\t')
    train_data[row][1] = train_data[row][1].split(' ')
    train_data[row][1].remove('\n')
    size.append(len(train_data[row][1]))

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
            train_set[row][col] = np.nan
        else:
            train_set[row][col] = train_data[row][1][col]

# create column name for dataframe
name = []
for i in range(6061):
    name.append(str(i))

# create dataframe for panda
traindf = pd.DataFrame(train_set, columns = name, dtype = 'float')
# calculate the ratio of missing value

# fill missing value
filled_train = traindf.fillna({ idx:traindf[idx].mean() for idx in traindf.columns })
# 注意： dataframe[idx] print 的是某個feature下所有data的值
#print(filled_train.head)

# use PCA to do dimension reduction
pca = PCA(0.95)
pca.fit(filled_train)
filled_train = pca.transform(filled_train)
#print(filled_train)

# sampling with smote and edited nearest neighbors
# https://imbalanced-learn.readthedocs.io/en/stable/combine.html
model_smote = SMOTEENN(random_state=0)
train_resample, class_resample = model_smote.fit_resample(filled_train, cs)
#print(Counter(class_resample))

# split to train set and test set from resample set
# with ratio of 0.75:0.25
X_train, X_test, y_train, y_test = train_test_split(train_resample, class_resample, test_size=0.25)

# use SVM-svc to train
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)
# predict
y_pred = classifier.predict(X_test)

# evaluation
#print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))