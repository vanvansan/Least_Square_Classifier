import scipy.io
import numpy as np
from sklearn.metrics import accuracy_score
import itertools

##part 1
def load_mnist_data(mat_path='mnist.mat'):
    #Load data using scipy module
    mat = scipy.io.loadmat(mat_path)
    X_train, y_train = mat['trainX'], mat['trainY'].flatten()
    X_test, y_test = mat['testX'], mat['testY'].flatten()
    X_train, X_test = X_train.astype(float) / 255.0, X_test.astype(float) / 255.0

    #make sure data are in correct type
    y_train, y_test = y_train.astype(int), y_test.astype(int)

    return X_train, y_train, X_test, y_test

def binary_least_squares_classifier(X, y):
    """
    Binary least squares classifier implementation.

    Parameters:
    - X: Feature matrix
    - y: Binary labels (+1 or -1)

    Returns:
    - Tuple of weights (beta, alpha)
    """
    X_bias = np.c_[np.ones(X.shape[0]), X]
    w = np.linalg.lstsq(X_bias, y, rcond=None)[0] #use linalg.lstsq to solve least square problem
    alpha, beta = w[0], w[1:] #getting output alpha and beta
    return beta, alpha

def train_one_vs_one(X_train, y_train):
    """
    Train one-versus-one multi-class classifier.

    Parameters:
    - X_train: Training feature matrix
    - y_train: Training labels (0 to K-1)

    Returns:
    - Dictionary of binary classifier weights {(i, j): (beta, alpha)}
    """
    K = np.max(y_train) + 1 #determine number of iterations
    classifiers = {} #build empty set for classifiers

    for i, j in itertools.combinations(range(K), 2):
        #Select data points for classes i and j
        mask_i = (y_train == i)
        mask_j = (y_train == j)
        indices = np.logical_or(mask_i, mask_j)
        binary_labels = np.where(mask_i, 1, -1)

        #call binary classifier to train
        beta, alpha = binary_least_squares_classifier(X_train[indices], binary_labels[indices])
        classifiers[(i, j)] = (beta, alpha)

    return classifiers

def train_one_vs_all(X_train, y_train):
    """
    Train one-versus-all multi-class classifier.

    Parameters:
    - X_train: Training feature matrix
    - y_train: Training labels (0 to K-1)

    Returns:
    - Dictionary of binary classifier weights {i: (beta, alpha)}
    """
    K = np.max(y_train) + 1
    classifiers = {}

    for i in range(K):
        binary_labels = np.where(y_train == i, 1, -1)
        beta, alpha = binary_least_squares_classifier(X_train, binary_labels)
        classifiers[i] = (beta, alpha)

    return classifiers


mat_path = 'mnist.mat'
X_train, y_train, X_test, y_test = load_mnist_data(mat_path)
classifiers_one_vs_one = train_one_vs_one(X_train, y_train)
for classes, weights in classifiers_one_vs_one.items():
    print(f"Classes {classes}: Beta = {weights[0]}, Alpha = {weights[1]}")



#part2
from sklearn.metrics import confusion_matrix, accuracy_score

def predict_one_vs_one(X, classifiers):
    """
    Predict using one-versus-one multi-class classifier.

    Parameters:
    - X: Test feature matrix
    - classifiers: Dictionary of binary classifier weights {(i, j): (beta, alpha)}

    Returns:
    - Predicted labels for X
    """
    #Determine the number of classes (K) by finding the maximum label in the classifiers dictionary
    K = max(max(pair) for pair in classifiers.keys()) + 1
    #Initialize a matrix to store predictions for each class
    predictions = np.zeros((X.shape[0], K))

    #Iterate through each pair of classes and their corresponding binary classifier weights
    for classes, weights in classifiers.items():
        i, j = classes
        beta, alpha = weights
        #Calculate the binary predictions for each class pair
        binary_prediction = np.dot(X, beta) + alpha
        #Update the predictions array based on the binary predictions
        predictions[:, i] += (binary_prediction > 0)
        predictions[:, j] += (binary_prediction <= 0)

    #Determine the class with the maximum accumulated votes for each sample
    predicted_labels = np.argmax(predictions, axis=1)
    #return array of labels
    return predicted_labels


def predict_one_vs_all(X, classifiers):
    """
    Predict using one-versus-all multi-class classifier.

    Parameters:
    - X: Test feature matrix
    - classifiers: Dictionary of binary classifier weights {i: (beta, alpha)}

    Returns:
    - Predicted labels for X
    """
    #Determine the number of classes (K) by getting the length of the classifiers dictionary
    K = len(classifiers)
    #Initialize an array to store predictions for each class
    predictions = np.zeros((X.shape[0], K))

    #Iterate through each class and its corresponding binary classifier weights
    for class_label, weights in classifiers.items():
        beta, alpha = weights
        #Calculate the binary predictions for the current class
        binary_prediction = np.dot(X, beta) + alpha
        #Update the predictions array for the current class
        predictions[:, class_label] = binary_prediction

    #Determine the class with the maximum prediction for each sample
    predicted_labels = np.argmax(predictions, axis=1)
    return predicted_labels


mat_path = 'mnist.mat'
X_train, y_train, X_test, y_test = load_mnist_data(mat_path)

# Train one-versus-one classifiers
classifiers_one_vs_one = train_one_vs_one(X_train, y_train)

# Train one-versus-all classifiers
classifiers_one_vs_all = train_one_vs_all(X_train, y_train)

# Predict using one-versus-one classifiers on the training data
predicted_labels_one_vs_one_train = predict_one_vs_one(X_train, classifiers_one_vs_one)
# Predict using one-versus-all classifiers on the training data
predicted_labels_one_vs_all_train = predict_one_vs_all(X_train, classifiers_one_vs_all)

# Calculate error rates and confusion matrices
error_rate_one_vs_one_train = 1 - accuracy_score(y_train, predicted_labels_one_vs_one_train)
confusion_matrix_one_vs_one_train = confusion_matrix(y_train, predicted_labels_one_vs_one_train)

error_rate_one_vs_all_train = 1 - accuracy_score(y_train, predicted_labels_one_vs_all_train)
confusion_matrix_one_vs_all_train = confusion_matrix(y_train, predicted_labels_one_vs_all_train)
print("One-vs-One Training Error Rate:", error_rate_one_vs_one_train)
print("One-vs-One Training Confusion Matrix:\n", confusion_matrix_one_vs_one_train)

print("\nOne-vs-All Training Error Rate:", error_rate_one_vs_all_train)
print("One-vs-All Training Confusion Matrix:\n", confusion_matrix_one_vs_all_train)

#part 3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix

# Function to plot confusion matrix
def plot_confusion(confusion_matrix, title, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Plot confusion matrices for both classifiers on test data
plot_confusion(confusion_matrix_one_vs_one_train, "One-vs-One Test Confusion Matrix", range(10))
plot_confusion(confusion_matrix_one_vs_all_train, "One-vs-All Test Confusion Matrix", range(10))




