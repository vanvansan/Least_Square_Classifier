import numpy as np
import scipy.io

from scipy.linalg import pinv

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
    
# returns the answer of the num
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
    
    
def part1(inputfile = 'mnist.mat',savefile = "1v1Matrix.mat"):
    print("Starting...")

    # Load the dataset (assuming mnist.mat is available)
    trainX, trainY, testX, testY = parse_data(inputfile)
    
    # training data
    optimal_theta_with_bias = train1v1(trainX, trainY, 0)
    
    # evaluating result
    accuracy =  evaluate1v1(optimal_theta_with_bias, testX, testY, 0)
    print(accuracy)

    # saving
    mdic ={"beta": optimal_theta_with_bias}
    scipy.io.savemat(savefile, mdic)
    print("The matrix has been saved to " + savefile + "using label 'beta'")

if __name__ == "__main__":
    part1('mnist.mat', "part1.mat")

    



