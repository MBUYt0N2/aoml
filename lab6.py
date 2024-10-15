from pprint import pprint
import time
import json
import argparse
import math
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV


def get_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def shuffle_and_get_data():
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist.data, mnist.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    X_test = X_test[:1000]
    y_test = y_test[:1000]

    return X_train, X_test, y_train, y_test


def perform_grid_search(model, X_train, y_train):
    param_grid = {
        "kernel": ["rbf"],
        "C": [10**i for i in range(3)],  
        "gamma": ["scale", 0.1, 1, 10, 100],
    }
    grid_search = GridSearchCV(model, param_grid, cv=2, verbose=10)
    grid_search.fit(X_train, y_train)
    return grid_search


def perform_random_search(model, X_train, y_train):
    param_distributions = {
        "kernel": ["linear", "rbf"],
        "C": [10**i for i in range(11)], 
        "gamma": ["scale", 0.1, 1, 10, 100],
    }
    random_search = RandomizedSearchCV(
        model, param_distributions, cv=2, verbose=10, random_state=42, n_iter=5
    )
    random_search.fit(X_train, y_train)
    return random_search


def perform_bayesian_search(model, X_train, y_train):
    search_spaces = {
        "kernel": ["linear", "rbf"],
        "C": [10**i for i in range(11)], 
        "gamma": ["scale", 0.1, 1, 10, 100],
    }
    bayes_search = BayesSearchCV(
        model, search_spaces, cv=2, verbose=10, random_state=42, n_iter=5
    )
    bayes_search.fit(X_train, y_train)
    return bayes_search


def main():
    X_train, X_test, y_train, y_test = shuffle_and_get_data()

    # train and test Grid search
    print("-----Grid Search-----")
    print("Training...")
    grid_clf = perform_grid_search(SVC(), X_train, y_train)
    print("Training Finished!")
    grid_acc = get_accuracy(y_test, grid_clf.predict(X_test))
    print("Ideal_parameters: ", dict(grid_clf.best_params_))
    print("Test accuracy", grid_acc)
    print()

    # train and test Bayesian Search
    print("-----Bayesian Search-----")
    print("Training...")
    bayes_clf = perform_bayesian_search(SVC(), X_train, y_train)
    print("Training Finished!")
    bayes_acc = get_accuracy(y_test, bayes_clf.predict(X_test))
    print("Ideal parameters: ", dict(bayes_clf.best_params_))
    print("Test accuracy:", bayes_acc)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search algorithm selector")
    parser.add_argument(
        "method",
        choices=["grid", "random", "bayes", "all"],
        help="Select the search method to use",
    )
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = shuffle_and_get_data()

    print()

    if args.method in ["grid", "all"]:
        # train and test Grid Search
        start = time.time()
        print("-----Grid Search-----")
        print("Training...")
        grid_clf = perform_grid_search(SVC(), X_train, y_train)
        print("Training Finished!")
        grid_acc = get_accuracy(y_test, grid_clf.predict(X_test))
        print("Ideal_parameters: ", dict(grid_clf.best_params_))
        print("Test accuracy", grid_acc)
        end = time.time()
        grid_time = end - start
        print("Time taken for grid search:", grid_time)
        print()

    if args.method in ["random", "all"]:
        # train and test Random Search
        start = time.time()
        print("-----Random Search-----")
        print("Training...")
        random_clf = perform_random_search(SVC(), X_train, y_train)
        print("Training Finished!")
        random_acc = get_accuracy(y_test, random_clf.predict(X_test))
        print("Ideal_parameters: ", dict(random_clf.best_params_))
        print("Test accuracy", random_acc)
        end = time.time()
        random_time = end - start
        print("Time taken for random search:", random_time)
        print()

    if args.method in ["bayes", "all"]:
        # train and test Bayesian Search
        start = time.time()
        print("-----Bayesian Search-----")
        print("Training...")
        bayes_clf = perform_bayesian_search(SVC(), X_train, y_train)
        print("Training Finished!")
        bayes_acc = get_accuracy(y_test, bayes_clf.predict(X_test))
        print("Ideal parameters: ", dict(bayes_clf.best_params_))
        print("Test accuracy:", bayes_acc)
        end = time.time()
        bayes_time = end - start
        print("Time taken for bayes search:", bayes_time)
        print()

    if args.method == "all":
        d = dict()
        d["grid"] = {
            "accuracy": grid_acc,
            "params": grid_clf.param_grid,
            "best_params": dict(sorted(dict(grid_clf.best_params_).items())),
        }

        d["random"] = {
            "accuracy": random_acc,
            "params": random_clf.param_distributions,
            "best_params": dict(sorted(dict(random_clf.best_params_).items())),
        }

        d["bayes"] = {
            "accuracy": bayes_acc,
            "params": bayes_clf.search_spaces,
            "best_params": dict(sorted(dict(bayes_clf.best_params_).items())),
        }

        pprint(d)
        with open("output_all.json", "w") as w:
            json.dump(d, w, indent=4)
        print()

    print(f"Finished {args.method}")
