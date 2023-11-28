import numpy as np
import scipy.io
from scipy.optimize import fmin


# Load the dataset (assuming mnist.mat is available)


import numpy as np
from scipy.optimize import minimize

def load_data():
    # Load your dataset here
    # Example: mnist = scipy.io.loadmat('mnist.mat')
    # trainX, trainY = mnist['trainX'], mnist['trainY']
    mnist = scipy.io.loadmat('mnist.mat')
    trainX = mnist['trainX']
    trainY = mnist['trainY']
    return trainX, trainY

def preprocess_data(trainX, trainY):
    # Preprocess your data here
    # Convert uint8 to float and normalize to [0, 1]
    trainX = trainX.astype(float) / 255.0
    # Convert labels to binary
    y_binary = is_zero(trainY)
    # y_binary = y_binary.flatten()  # or y.ravel() to make it a 1D array
    y_transposed = y_binary.T 

    print("y shape is ")
    print(y_transposed.shape)
    print("x shape is ")
    print(trainX.shape)


    return trainX, y_transposed


def is_zero(y):
    # Function to convert labels to binary (0 or 1)
    return (y == 0).astype(float)

def least_squares_objective(beta, X, y):
    # Least squares objective function
    # initialize beta

    print("beta shape is ")
    print(beta.shape)
    # return np.sum((X.dot(beta) + bias - y) ** 2)
    return np.sum((X.dot(beta) - y) ** 2)

def train_least_squares_classifier(X, y):
    # Initial guess for parameters
    beta_0 = np.random.randn(X.shape[1], 1)

    # Minimize the objective function
    result = fmin(least_squares_objective, beta_0, args=(X, y))

    # Extract the optimal parameters
    optimal_beta = result.x
    print("result.x shape is ")
    print(result.x.shape)
    # optimal_bias = result.x[-1]

    # return optimal_theta, optimal_bias
    return optimal_beta

def evaluate_classifier(X, beta_hat, y):
    # Make predictions on the data
    # predictions = np.sign(X.dot(beta) + optimal_bias)
    predictions = np.sign(X.dot(beta_hat))

    # Evaluate the accuracy
    accuracy = np.sum(predictions == y) / len(y)

    print(f'Optimal beta: {beta_hat}')
    # print(f'Optimal Bias: {optimal_bias}')
    print(f'Accuracy: {accuracy * 100}%')

if __name__ == "__main__":
    print("Starting...")

    # Load and preprocess your data
    trainX, trainY = load_data()
    trainX, y_binary = preprocess_data(trainX, trainY)

    # Train the least squares classifier
    beta_hat = train_least_squares_classifier(trainX, y_binary)

    # Evaluate the classifier
    evaluate_classifier(trainX, beta_hat, y_binary)

    print("Complete")



'''
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
'''
