import numpy as np
from collections import Counter

class KNNClassifier:
	def __init__(self, k, p=3):
		self.k = k
		self.p = p
		self.X_train = None
		self.y_train = None

	def minkowski_distance(self, x1, x2):
		a = []
		for i, j in zip(x1, x2):
			a.append( abs((i - j) ** self.p))
		return np.sum(a) ** (1 / self.p)


	def get_neighbors(self, test_instance):
		dist = []
		for j, i in enumerate(self.X_train):
			dist.append((self.minkowski_distance(i, test_instance), j))
		return sorted(dist, key=lambda x : x[0])[:self.k]

	def predict_classification(self, neighbors):
		l = [self.y_train[i[1]] for i in neighbors]
		l = Counter(l)
		return l.most_common(1)[0][0]


	def fit(self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X_test):
		predictions = []
		for test_instance in X_test:
			neighbors = self.get_neighbors(test_instance)
			majority_vote = self.predict_classification(neighbors)
			predictions.append(majority_vote)
		return predictions


knn = KNNClassifier(k=3, p=3)
np.random.seed(240)
num_samples = 100
X_train = np.random.rand(num_samples, 3)
y_train = np.random.randint(0, 4, size=num_samples)
X_test = np.array([[0.4, 0.6, 0.8]]) 
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(f"Predicted class label for X_test: {predictions}")