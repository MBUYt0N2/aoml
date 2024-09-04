import numpy as np


class Node:
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value
        self.left = None
        self.right = None
        self.leaf = None


class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X_train, y_train):
        self.tree = self._build_tree(X_train, y_train, 0)

    def predict(self, X_test):
        res = []
        for i in X_test:
            node = self.tree
            while node.leaf is None:
                if i[node.feature] < node.value:
                    node = node.left
                else:
                    node = node.right
            res.append(node.leaf)
        return res

    def _best_split(self, X, y):
        best = float("inf")
        split = None
        for i in range(X.shape[1]):
            x = X[:, i]
            for j in x:
                g = self._gini([k for k in range(len(x)) if x[k] > j], y) + self._gini(
                    [k for k in range(len(x)) if x[k] < j], y
                )
                if g < best:
                    best = g
                    split = (i, j)
        return split

    def _gini(self, indices, y):
        total = len(indices)
        if total == 0:
            return 0
        count_0 = np.sum(y[indices] == 0)
        count_1 = np.sum(y[indices] == 1)
        p_0 = count_0 / total
        p_1 = count_1 / total
        return 1 - p_0**2 - p_1**2

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth:
            node = Node(None, None)
            if len([i for i in y if i == 0]) > len([i for i in y if i == 1]):
                node.leaf = 0
            else:
                node.leaf = 1
            return node
        else:
            split = self._best_split(X, y)
            if split is None:
                node = Node(None, None)
                if len([i for i in y if i == 0]) > len([i for i in y if i == 1]):
                    node.leaf = 0
                else:
                    node.leaf = 1
                return node
            feature, value = split
            node = Node(feature, value)
            left = [i for i in range(len(X)) if X[i, feature] < value]
            right = [i for i in range(len(X)) if X[i, feature] > value]
            node.left = self._build_tree(X[left], y[left], depth + 1)
            node.right = self._build_tree(X[right], y[right], depth + 1)
            return node


# Example usage
tree = DecisionTreeClassifier(max_depth=3)
np.random.seed(240)
num_samples = 100
X_train = np.random.rand(num_samples, 2)
y_train = (np.random.rand(num_samples) > 0.5).astype(int)
s = (np.random.rand(num_samples) > 0.5).astype(int)
X_test = np.array([[0.5, 0.5]])  # Single test instance
tree.fit(X_train, y_train)
predictions = tree.predict(X_test)
print(f"Predicted class label for X_test: {predictions[0]}")
