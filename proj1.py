import numpy as np
import scipy.io

from scipy.linalg import lstsq, pinv

def objective(theta, x, y):
    beta = theta[:-1]
    bias = theta[-1]
    return np.sum((x.dot(beta) + bias - y) ** 2)

print("Starting...")

# Load the dataset (assuming mnist.mat is available)
mnist = scipy.io.loadmat('mnist.mat')
trainX = mnist['trainX']
trainY = mnist['trainY']
testY = mnist['testY']
testX = mnist['testX']

# formatting 
trainY = (trainY == 0).astype(float).T
testY = (testY == 0).astype(float).T
trainY = np.where(trainY == 0, 1, -1)
testY = np.where(testY == 0, 1, -1)


trainX = trainX.astype(float) / 255.0
testX = testX.astype(float) / 255.0
X_with_bias = np.hstack((trainX, np.ones((trainX.shape[0], 1))))
testX_with_bias = np.hstack((testX, np.ones((testX.shape[0], 1))))

# training data
result = lstsq(trainX, trainY)
# result_with_bias = lstsq(X_with_bias, trainY)
result_with_bias = pinv(X_with_bias.T.dot(X_with_bias)).dot(X_with_bias.T).dot(trainY)
optimal_theta = result[0]
optimal_theta_with_bias = result_with_bias
print(result_with_bias.shape)


# evaluating result
predictions = np.sign(testX_with_bias.dot(optimal_theta_with_bias))
# all -1 and 1, change to 1 and 0
# predictions +=1
# predictions = np.sign(predictions)

accuracy = np.sum(predictions == testY) / len(testY)
print(testX_with_bias.shape)
print(accuracy)


# saving
mdic ={"beta": optimal_theta_with_bias}
scipy.io.savemat("1v1Matrix.mat", mdic)

print("The matrix has been saved to 1v1Matrix.mat")





