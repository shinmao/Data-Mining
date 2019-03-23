# python3

input_path = "./input.dat"
feature_path = "./feature.dat"

sparse_matrix = [[0]*126373 for i in range(3147)]

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

def main():
    read_sparse()

if __name__ == "__main__":
    main()
