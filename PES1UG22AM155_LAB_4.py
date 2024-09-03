import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def crossEntropy(self, x, y):
        epsilon = 1e-15
        x = np.clip(x, epsilon, 1 - epsilon)
        return -np.sum(y * np.log(x) + (1 - y) * np.log(1 - x)) / len(y)

    def fit(self, X_train, y_train):
        num_samples, num_features = X_train.shape
        self.weights = np.zeros(num_features)
        self.bias = 0
        for i in range(self.num_iterations):
            z = np.dot(X_train, self.weights) + self.bias
            a = self.sigmoid(z)
            # if i % 100 == 0:
            #     print(self.crossEntropy(a, y_train))
            dz = a - y_train
            dw = np.dot(X_train.T, dz) / num_samples
            db = np.sum(dz) / num_samples
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X_test):
        a = self.predict_proba(X_test)
        return np.astype(np.round(a), np.int32)

    def predict_proba(self, X_test):
        z = np.dot(X_test, self.weights) + self.bias
        return self.sigmoid(z)


lr = LogisticRegression(learning_rate=0.1, num_iterations=1000)
np.random.seed(240)
num_samples = 100
X_train = np.random.rand(num_samples, 2)
y_train = (np.random.rand(num_samples) > 0.5).astype(int)
X_test = np.array([[0.5, 0.5]])
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)
print(f"Predicted class label for X_test: {predictions[0]}")
