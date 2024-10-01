# test_case.py
import unittest
from lab5 import generate_data, train_and_evaluate
from sklearn.model_selection import train_test_split

class TestRegularization(unittest.TestCase):
    def test_mse(self):
        X, y = generate_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        mse_lr, mse_ridge, mse_lasso = train_and_evaluate(X_train, y_train, X_test, y_test)

        # MSE values
        print(f"Linear Regression MSE: {mse_lr}")
        print(f"Ridge Regression MSE: {mse_ridge}")
        print(f"Lasso Regression MSE: {mse_lasso}")

        # Check if MSE values are within a reasonable range
        self.assertTrue(mse_lr > 0)
        self.assertTrue(mse_ridge > 0)
        self.assertTrue(mse_lasso > 0)

if __name__ == '__main__':
    unittest.main()
