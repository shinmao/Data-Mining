import re

train_path = './train_drugs.data'
test_path = './test.data'

size = []

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

train_set = [[0]*6062 for i in range(800)]
for row in range(800):
    for col in range(6062):
        if col == 0:
            train_set[row][0] = train_data[row][0]
        elif col > len(train_data[row][1]):
            train_set[row][col] = 0
        else:
            train_set[row][col] = train_data[row][1][col-1]

#print(train_set[0][1])