# python3
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import ClusterCentroids
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from numpy  import array
train_path = './train_drugs.data'
test_path = './test.data'
TRAIN_SIZE = 800
TEST_SIZE = 350
# number of classifiers for cross validation to choose the best model
# turned out that random forest might be the best model, set clf_num =1  
clf_num = 1
pca_lambda = 0.8
# number of estimaters for bagging and random forest classifiers
num_trees = 100

# class
cs = []
features = []
max_d=0
train_data = [[] for i in range(800)]
test_data = []
train_set = [[0]*(max_d+1) for i in range(800)]
test_set = [[0]*(max_d+1) for i in range(350)]


def main():
    global train_set, test_set
    read_data()
    get_max_dimension()
    train_set = sparse_to_dense(TRAIN_SIZE,train_data)
    test_set = sparse_to_dense(TEST_SIZE,test_data)
    test_class=[]
    # split_data() only one time 
    # traindf, testdf,train_class,test_class = split_data()

    # split_data() 10 times randomly, and do cross validation for each classifier
    cross_validation()

    # create data frame for real testing
    # traindf,testdf,train_class = create_dataframe()
    # resample data for real training
    # test_red, train_resampled, class_resampled = training(traindf,testdf,train_class)

    # classifiers
    # random_forest_classifier(train_resampled,class_resampled,test_red,test_class)
    # SVM_classifier(train_resampled,class_resampled,test_red)
    # decision_tree_classifier(train_resampled,class_resampled,test_red,test_class)
    # naive_bayes_classifier(train_resampled,class_resampled,test_red)
    


def read_data():
    # handle the train data into array 
    with open(train_path) as f:
        data = f.readlines()
    # train_data[row][0] to get drug active or inactive
    # train_data[row][1][index] to get feature element
    train_buffer = [[] for i in range(800)]
    for row in range(TRAIN_SIZE):
        train_buffer[row] = data[row].split('\t')
        train_buffer[row][1] = train_buffer[row][1].split(' ')
        train_buffer[row][1].remove('\n')
        # size.append(len(train_data[row][1]))

    # put training data into a 2D list
    for row in range(TRAIN_SIZE):
        for col in range(6061):
            if col< len(train_buffer[row][1]):
                train_data[row].append(train_buffer[row][1][col])

    # cs[row] to show the class of row th
    for i in range(800):
        cs.append(int(train_buffer[i][0]))

    # handle the test data into array 
    with open(test_path) as f:
        data = f.readlines()
    for row in range(350):
        test_data.append(data[row].split()) 

# get maximum dimension
def get_max_dimension():
    global max_d
    for row in range(TRAIN_SIZE):
        for col in range(6061):
            if col < len(train_data[row]):
                x = int(train_data[row][col])
                if x > max_d:
                    max_d= int(train_data[row][col])
    # features name
    for i in range(max_d+1):
        features.append(str(i))

# convert sparse binary matrix to dense matrix
def sparse_to_dense(rows,data):
    dense_matrix = [[0]*(max_d+1) for i in range(rows)]
    for row in range(rows):
        for col in range(6061):
            if col < len(data[row]):
                    index = int(data[row][col])
                    dense_matrix[row][index] = 1
    return dense_matrix

def cross_validation():
    cross_validation_res = np.zeros(shape=(clf_num,2))
    f1_score_arr = np.zeros(shape=(clf_num,10,2))
    
    # list for each clasifier function 
    # classifier_functions=[SVM_classifier,decision_tree_classifier,naive_bayes_classifier,random_forest_classifier,\
    #     adaboost_classifier]

    classifier_functions=[random_forest_classifier]
    i = 0
    while i<10:
        # while doing the cross validation, the random state of split data need to be "None"s
        traindf, testdf,train_class,test_class = split_data()
        test_red, train_resampled, class_resampled = training(traindf,testdf,train_class)
        # fill in f1 score of each classifier in the array
        for j in range(len(classifier_functions)):
            score = classifier_functions[j](train_resampled,class_resampled,test_red,test_class)
            f1_score_arr[j][i]=score
        i+=1

    # get average f1 score of each classifier
    for i in range(clf_num):
        cross_validation_res[i] = f1_score_arr[i].mean(axis=0)

    print(cross_validation_res)

# split training data into training and testing for validation  
def split_data():
    train_split, test_split, train_class, test_class = train_test_split(train_set, cs, test_size=1/4.0)

    # create dataframe for panda
    traindf = pd.DataFrame(train_split, columns = features, dtype = 'float')
    traindf.name='traindf'
    testdf = pd.DataFrame(test_split, columns = features, dtype = 'float')
    testdf.name = 'testdf'

    # print(testdf)
    # traindf['class'] = train_class
    return traindf, testdf,train_class,test_class


# create dataframe for panda without splitting training data
def create_dataframe():
    # create dataframe for panda
    traindf = pd.DataFrame(train_set, columns = features, dtype = 'float')
    traindf.name='traindf'
    testdf = pd.DataFrame(test_set, columns = features, dtype = 'float')
    testdf.name = 'testdf'

    # print(testdf)
    train_class = cs
    # traindf['class'] = train_class
    return traindf, testdf,train_class



def training(traindf, testdf,train_class):
    # standardize dataset before applying PCA 
    scaler = StandardScaler()
    train_std = scaler.fit_transform(traindf) 
    test_std = scaler.transform(testdf)
 
    train_red, test_red = pca_reduction(train_std,test_std)

    # Combine over- and under-sampling using SMOTE and Tomek links.
    smote_tomek = SMOTETomek(random_state=42)
    train_resampled, class_resampled = smote_tomek.fit_resample(train_red,train_class)
    return test_red, train_resampled, class_resampled


def pca_reduction(train_std,test_std):

    #Plotting the Cumulative Summation of the Explained Variance
    # pca = PCA().fit(train_std)
    # plt.figure()
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    # plt.xlabel('Number of Components')
    # plt.ylabel('Variance (%)') #for each component
    # plt.title('Drug Dataset PCA Explained Variance')
    # plt.show()
    # choose the minimum number of principal components such that 99% of the variance is retained.
    pca = PCA(pca_lambda)
    train_pca = pca.fit_transform(train_std)
    test_pca = pca.transform(test_std)
    print(train_pca.shape)
    return train_pca,test_pca



def SVM_classifier(train_resampled,class_resampled,test_red,test_class):
    classifier = SVC(kernel='linear')
    classifier.fit(train_resampled,class_resampled)
    # predict
    predict = classifier.predict(test_red)
    # for i in range(len(predict)):
    #     print(predict[i])
    print("SVM_classifier:\n")
    print("pca lambda = " + str(pca_lambda))
    f1_score = evaluation(test_class,predict)
    return f1_score

def random_forest_classifier(train_resampled,class_resampled,test_red,test_class):
    f1_score =0.0
    predict = RandomForestClassifier (n_estimators=num_trees, random_state=0,min_samples_leaf=50).fit(train_resampled, class_resampled).predict(test_red) 
    # for i in range(TEST_SIZE):
    #     print(predict[i])
    print("random forest classifier classifier:\n"+ "number of trees = "+ str(num_trees))
    print("pca lambda = " + str(pca_lambda))
    f1_score = evaluation(test_class,predict)
    print(f1_score)
    return f1_score
    
def decision_tree_classifier(train_resampled,class_resampled,test_red,test_class):
   
    classifier = tree.DecisionTreeClassifier()
    f1_score =0.0
    # predict
    predict = BaggingClassifier(base_estimator=classifier, n_estimators=num_trees, random_state=0).fit(train_resampled, class_resampled).predict(test_red) 
    # print(predict)
    # for i in range(len(predict)):
    #     print(predict[i])
    print("decision tree classifier:\n"+ "number of trees = "+ str(num_trees))
    print("pca lambda = " + str(pca_lambda))
    f1_score = evaluation(test_class,predict)
    return f1_score
    

def naive_bayes_classifier(train_resampled,class_resampled,test_red,test_class):
    gnb = GaussianNB()
    gnb.fit(train_resampled,class_resampled)
    predict = gnb.predict(test_red)
    # for i in range(len(predict)):
    #     print(predict[i])

    print("naive_bayes_classifier:\n")
    print("pca lambda = " + str(pca_lambda))
    f1_score = evaluation(test_class,predict)
    return f1_score
def adaboost_classifier(train_resampled,class_resampled,test_red,test_class):
    f1_score =0.0
    # predict
    predict = AdaBoostClassifier(random_state=0).fit(train_resampled, class_resampled).predict(test_red) 
    # print(predict)
    # for i in range(len(predict)):
    #     print(predict[i])
    print("adaboost classifier:\n")
    print("pca lambda = " + str(pca_lambda))
    f1_score = evaluation(test_class,predict)
    return f1_score

def evaluation(test_class,predict):
    # print(classification_report(test_class,predict))
    return f1_score(test_class, predict,labels = [0,1],average=None) 
  
main()
