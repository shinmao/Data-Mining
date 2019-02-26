# python3
import re
import pandas as pd
import numpy as np

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
print(filled_train)