import numpy as np
import scipy.io
from scipy.optimize import minimize

print("Starting...")

# Load the dataset (assuming mnist.mat is available)
mnist = scipy.io.loadmat('mnist.mat')
trainX = mnist['trainX']
trainY = mnist['trainY']

# Convert uint8 to float and normalize to [0, 1]
trainX = trainX.astype(float) / 255.0

# Define the function to convert labels to binary (0 or 1)
def is_zero(y):
    return (y == 0).astype(float)

# Convert labels to binary
y_binary = is_zero(trainY)

# Define the objective function
def objective(theta, X, y):
    beta = theta[:-1]
    bias = theta[-1]
    return np.sum((X.dot(beta) + bias - y) ** 2)

# Initial guess for parameters
initial_guess = np.random.randn(trainX.shape[1] + 1)

# Minimize the objective function
result = minimize(objective, initial_guess, args=(trainX, y_binary), method='BFGS')

# Extract the optimal parameters
optimal_theta = result.x[:-1]
optimal_bias = result.x[-1]

# Make predictions on the training data
predictions_train = np.sign(trainX.dot(optimal_theta) + optimal_bias)

# Evaluate the accuracy on the training data
accuracy_train = np.sum(predictions_train == y_binary) / len(y_binary)

print(f'Optimal Theta: {optimal_theta}')
print(f'Optimal Bias: {optimal_bias}')
print(f'Training Accuracy: {accuracy_train * 100}%')

print("Complete")
