import numpy as np


class NaiveBayesClassifier:
    def __init__(self):
        """
        Initialize the Naive Bayes classifier.
        """
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        """
        Train the Naive Bayes classifier.
        Parameters:
        X (np.ndarray): Training data, shape (num_samples, num_features)
        y (np.ndarray): Training labels, shape (num_samples,)
        """
        self.classes = np.unique(y)
        num_features = X.shape[1]
        self.mean = np.zeros((len(self.classes), num_features), dtype=np.float64)
        self.var = np.zeros((len(self.classes), num_features), dtype=np.float64)
        self.priors = np.zeros(len(self.classes), dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            self.mean[idx, :] = X_c.mean(axis=0)
            self.var[idx, :] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / float(X.shape[0])

    def predict(self, X):
        """
        Predict the class labels for the input data.
        Parameters:
        X (np.ndarray): Test data, shape (num_samples, num_features)
        Returns:
        np.ndarray: Predicted class labels, shape (num_samples,)
        """
        y_pred = [self.__calculate_posterior(x) for x in X]
        return np.array(y_pred)

    def __calculate_posterior(self, x):
        posteriors = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            posterior = np.sum(np.log(self.__calculate_likelihood(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self.classes[np.argmax(posteriors)]

    def __calculate_likelihood(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator


if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(240)
    X_train = np.random.randn(100, 2)
    y_train = np.random.randint(0, 2, 100)
    # Create and train Naive Bayes classifier
    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)
    # Predict on new data
    X_test = np.random.randn(5, 2)
    print(f"Test data: {X_test}")
    predictions = nb.predict(X_test)
    print(f"Predicted class labels: {predictions}")
