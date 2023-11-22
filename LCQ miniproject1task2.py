import numpy as np
import scipy.io
from sklearn.metrics import accuracy_score, confusion_matrix

def load_mnist_data(mat_path='mnist.mat'):
    mat = scipy.io.loadmat(mat_path)
    X_train, y_train = mat['trainX'], mat['trainY'].flatten()
    X_test, y_test = mat['testX'], mat['testY'].flatten()
    X_train, X_test = X_train.astype(float) / 255.0, X_test.astype(float) / 255.0
    y_train, y_test = y_train.astype(int), y_test.astype(int)
    return X_train, y_train, X_test, y_test

def generate_random_features(X, num_features, W, b, g):
    return np.array([g(np.dot(xi, W.T) + b) for xi in X]) #this corresponding to feature functions
# g is the function g(x)

def feature_space_least_squares_classifier(X, y):
    X_bias = np.c_[np.ones(X.shape[0]), X]
    w = np.linalg.lstsq(X_bias, y, rcond=None)[0] #use this to solve linear sqaures
    alpha, beta = w[0], w[1:]
    return beta, alpha

def train_feature_space_classifier(X_train, y_train, num_features, L, g):
    K = np.max(y_train) + 1
    classifiers = {}

    for i in range(K):
        W = np.random.normal(0, 1, size=(L, X_train.shape[1]))
        b = np.random.normal(0, 1, size=(L,))
        features = generate_random_features(X_train, num_features, W, b, g)
        beta, alpha = feature_space_least_squares_classifier(features, (y_train == i).astype(int))
        classifiers[i] = (beta, alpha, W, b)

    return classifiers

def predict_feature_space(X, classifiers, num_features, L, g):
    K = len(classifiers)
    predictions = np.zeros((X.shape[0], K))

    for class_label, weights in classifiers.items():
        beta, alpha, W, b = weights
        features = generate_random_features(X, num_features, W, b, g)
        print(features)
        print(features.shape)
        binary_prediction = np.dot( features, beta) + alpha
        predictions[:, class_label] = binary_prediction.reshape(-1)  #reshape to make it 1D
    #determine the final predicted labels based on the class with maximum votes
    predicted_labels = np.argmax(predictions, axis=1) 
    return predicted_labels


#non-linear mapping functions
def identity_function(x):
    return x

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

def sinusoidal_function(x):
    return np.sin(x)

def relu_function(x):
    return np.maximum(x, 0)

#Load MNIST data
mat_path = 'mnist.mat'
X_train, y_train, X_test, y_test = load_mnist_data(mat_path)

#Choose the non-linear mapping function
non_linear_mapping = relu_function

#Train classifiers in the feature space
classifiers_feature_space = train_feature_space_classifier(X_train, y_train, num_features=2000, L=2000, g=non_linear_mapping)
#Predict using classifiers in the feature space
predicted_labels_feature_space = predict_feature_space(X_test, classifiers_feature_space, num_features=2000, L=2000, g=non_linear_mapping)
#Evaluate performance
error_rate_feature_space = 1 - accuracy_score(y_test, predicted_labels_feature_space)
confusion_matrix_feature_space = confusion_matrix(y_test, predicted_labels_feature_space)
print("Error Rate (Feature Space):", error_rate_feature_space)
print("Confusion Matrix (Feature Space):")
print(confusion_matrix_feature_space)



