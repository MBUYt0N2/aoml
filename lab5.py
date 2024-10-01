# assignment.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error

# Function to generate synthetic data
def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.rand(n_samples, 1) * 10
    y = 2 * X.squeeze() + 1 + np.random.randn(n_samples)
    return X, y

# Function to train and evaluate models
def train_and_evaluate(X_train, y_train, X_test, y_test):
    # TODO: Initialize Linear Regression model
    lr_model = LinearRegression()

    # TODO: Train the model
    lr_model.fit(X_train, y_train)

    # TODO: Predict on test data
    y_pred_lr = lr_model.predict(X_test)

    # Calculate MSE for Linear Regression
    mse_lr = mean_squared_error(y_test, y_pred_lr)

    # TODO: Initialize Ridge Regression model with alpha=1.0
    ridge_model = Ridge(alpha=1.0)

    # TODO: Train the Ridge model
    ridge_model.fit(X_train, y_train)

    # TODO: Predict on test data with Ridge model
    y_pred_ridge = ridge_model.predict(X_test)

    # Calculate MSE for Ridge Regression
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)

    # TODO: Initialize Lasso Regression model with alpha=1.0
    lasso_model = Lasso(alpha=1.0)

    # TODO: Train the Lasso model
    lasso_model.fit(X_train, y_train)

    # TODO: Predict on test data with Lasso model
    y_pred_lasso = lasso_model.predict(X_test)

    # Calculate MSE for Lasso Regression
    mse_lasso = mean_squared_error(y_test, y_pred_lasso)

    return mse_lr, mse_ridge, mse_lasso

def main():
    X, y = generate_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    mse_lr, mse_ridge, mse_lasso = train_and_evaluate(X_train, y_train, X_test, y_test)
    
    print(f"Linear Regression MSE: {mse_lr}")
    print(f"Ridge Regression MSE: {mse_ridge}")
    print(f"Lasso Regression MSE: {mse_lasso}")

if __name__ == "__main__":
    main()
