import numpy as np
import scipy.io
from scipy.linalg import pinv
import pdb

def objective(theta, x, y):
    beta = theta[:-1]
    bias = theta[-1]
    return np.sum((x.dot(beta) + bias - y) ** 2)

# input 
# v: the original vector
# target: the reference digit
# 
# returns v(sign(v.i))
def to_binary(v, target = 0):
    return np.where(v == target, 1, -1)


# input filename string
# find trainX, trainY, testX, testY in the "filename" then return the formated version

# returns trainX, trainY, testX, testY
def parse_data(filename):
    mnist = scipy.io.loadmat(filename)
    TRAIN_X = mnist['trainX']
    TRAIN_Y = mnist['trainY']
    TEST_X = mnist['testX']
    TEST_Y = mnist['testY']
    
    return TRAIN_X, TRAIN_Y, TEST_X, TEST_Y
    
# format for the 1v1
def formatting(m ,v, target_num = 0):
    Y = to_binary(v, target_num).T
    # testY = to_binary(testY, target_num).T
    X = m.astype(float) / 255.0
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    return X, Y

def train1v1(x_1, y_1, target_num):
    x_1, y_1 = formatting(x_1, y_1, target_num)
    return pinv(x_1.T.dot(x_1)).dot(x_1.T).dot(y_1)

def evaluate1v1(m ,tx_1, ty_1, target_num):
    tx_1, ty_1 = formatting(tx_1, ty_1, target_num)
    predictions = np.sign(tx_1.dot(m))
    return sum(predictions == ty_1) / len(ty_1)

# the high level function for classifier
def multiclassifier_1vN(model_list, x):
    most_prob = -1
    most_likely_num = -1
    for i in range(10):
        probability = x.dot(model_list[i])
        if (probability > most_prob):
            most_likely_num = i
            most_prob=  probability
    return most_likely_num
    
    
def part1(inputfile = 'mnist.mat',savefile = "1v1Matrix.mat"):
    print("Starting...")

    # Load the dataset (assuming mnist.mat is available)
    trainX, trainY, testX, testY = parse_data(inputfile)
    testY = testY.T
    
    # predictions = np.sign(tx_1.dot(m))
    dummyY = testY
    result = np.ones([10000,1])
    tx_1, dummy = formatting(testX, dummyY)
    confusion_m = np.zeros([10,10], dtype=int)
    count_error = 0
    
    f_n = []
    # training data
    for i in range(10):
        model = train1v1(trainX, trainY, i)
        f_n.append(model)


    # test the result
    for i in range(10000):
        prediction = multiclassifier_1vN(f_n, tx_1[i])
        result[i] = int(prediction)
        if testY[i][0] != prediction:
            count_error += 1
        confusion_m[testY[i][0]][int(prediction)] += 1

    
    print(sum(result == testY) / len(testY))
    
    
    # Confusion matrix
    print("Confusion Matrix:")
    print(confusion_m)

    # # saving
    # mdic ={"beta": model}
    # scipy.io.savemat(savefile, mdic)
    # print("The matrix has been saved to " + savefile + "using label 'beta'")

if __name__ == "__main__":
    part1('mnist.mat', "part1.mat")

    



