import numpy as np
import scipy.io
from scipy.linalg import lstsq

def objective(theta, x, y):
    beta = theta[:-1]
    bias = theta[-1]
    return np.sum((x.dot(beta) + bias - y) ** 2)

print("Starting...")
# Load the dataset (assuming mnist.mat is available)
mnist = scipy.io.loadmat('mnist.mat')
trainX = mnist['trainX']
trainY = mnist['trainY']

trainY = (trainY == 0).astype(float).T
trainX = trainX.astype(float) / 255.0
X_with_bias = np.hstack((trainX, np.ones((trainX.shape[0], 1))))

# print(X_with_bias.shape)

result = lstsq(trainX, trainY)
result_with_bias = lstsq(X_with_bias, trainY)

optimal_theta = result[0]
optimal_theta_with_bias = result_with_bias[0]

print(optimal_theta_with_bias.shape)

# predictions = np.sign(trainX.dot(optimal_theta))
# print(X_with_bias.shape)


predictions = np.sign(X_with_bias.dot(optimal_theta_with_bias))
# all -1 and 1, change to 1 and 0
predictions +=1
predictions = np.sign(predictions)
print(predictions)

# Evaluate the accuracy
accuracy = np.sum(predictions == trainY) / len(trainY)
print(accuracy)

# print(trainX[4])
# print(trainY[0])




