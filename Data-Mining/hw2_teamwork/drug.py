# python3
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.svm import SVC
from sklearn.metrics import classification_report
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
    # size.append(len(train_data[row][1]))

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

# create features for dataframe (columns)
features = []
for i in range(6061):
    features.append(str(i))

test_data = []
# handle the train data into array 
with open(test_path) as f:
    data = f.readlines()

# train_data[row][0] to get drug active or inactive
# train_data[row][1][index] to get feature element
for row in range(350):
    test_data.append(data[row].split()) 

test_set = [[0]*6061 for i in range(350)]

for row in range(350):
    for col in range(6061):
        if col >= len(test_data[row]):
            # panda consider NaN as missing value
            test_set[row][col] = np.nan
        else:
            test_set[row][col] = test_data[row][col]

def split_data():
    # split data
    train_split, test_split, train_class, test_class = train_test_split(train_set, cs, test_size=1/4.0, random_state=42)

    # create dataframe for panda
    traindf = pd.DataFrame(train_split, columns = features, dtype = 'float')
    traindf.name='traindf'
    testdf = pd.DataFrame(test_split, columns = features, dtype = 'float')
    testdf.name = 'testdf'

    # print(testdf)
    traindf['class'] = train_class
    return traindf, testdf,train_class,test_class

def create_dataframe():
    # # split data
    # train_split, test_split, train_class, test_class = train_test_split(train_set, cs, test_size=1/4.0, random_state=0)

    # create dataframe for panda
    traindf = pd.DataFrame(train_set, columns = features, dtype = 'float')
    traindf.name='traindf'
    testdf = pd.DataFrame(test_set, columns = features, dtype = 'float')
    testdf.name = 'testdf'

    # print(testdf)
    train_class = cs
    traindf['class'] = train_class
    return traindf, testdf,train_class

# fill missing value with attribute mean for all samples belonging to the same class
def fill_missing_value(dataframe):
    # if dataframe.name == "traindf":
    #     for idx in features:
    #         dataframe[idx] = dataframe.groupby('class')[idx].transform(lambda x: x.fillna(x.mean()))
    # # dataframe.fillna(0,inplace=True)
    dataframe.fillna(traindf.mean(), inplace=True)
    return dataframe
    # print(filled_train)

traindf, testdf,train_class,test_class = split_data()
# traindf,testdf,train_class = create_dataframe()
filled_train = fill_missing_value(traindf)
filled_test = fill_missing_value(testdf)

print(filled_train)
print(filled_test)
# drop the class column before standardizing
filled_train = filled_train.drop('class',1)

# #Plotting the Cumulative Summation of the Explained Variance
# pca = PCA().fit(filled_train_std)
# plt.figure()
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Variance (%)') #for each component
# plt.title('Drug Dataset PCA Explained Variance')
# plt.show()

def training(filled_train, filled_test):
    # standardize dataset before applying PCA 
    scaler = StandardScaler()
    train_std = scaler.fit_transform(filled_train) 
    test_std = scaler.transform(filled_test)
    # choose the minimum number of principal components such that 99% of the variance is retained.
    pca = PCA(.99)
    train_pca = pca.fit_transform(train_std)
    test_pca = pca.transform(test_std)

    # print(np.cumsum(pca.explained_variance_ratio_)) #0.98115453

    # Combine over- and under-sampling using SMOTE and Tomek links.
    smote_tomek = SMOTETomek(random_state=42)
    train_resampled, class_resampled = smote_tomek.fit_resample(train_pca,train_class)

    classifier = SVC(kernel='linear')
    classifier.fit(train_resampled,class_resampled)
    # predict
    predict = classifier.predict(test_pca)
    for i in range(len(predict)):
        print(predict[i])

    evaluation(test_class,predict)

def evaluation(test_class,predict):
    print(classification_report(test_class,predict))


training(filled_train,filled_test)