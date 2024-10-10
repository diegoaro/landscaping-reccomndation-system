import pytest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from src.model_trainer import train_model, evaluate_model


def test_train_model():
    """Test if the RandomForest model trains successfully."""

    # Generate dummy data
    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    # Train the model
    model, X_test, y_test = train_model(X, y)

    assert isinstance(model, RandomForestRegressor), "Model should be an instance of RandomForestRegressor"
    assert X_test.shape[0] > 0, "Test set should not be empty"


def test_evaluate_model():
    """Test the evaluation function for the model."""

    # Generate dummy data
    X = np.random.rand(100, 10)
    y = np.random.rand(100)

    # Train the model
    model, X_test, y_test = train_model(X, y)

    # Evaluate the model
    mse = evaluate_model(model, X_test, y_test)

    assert mse >= 0, "Mean squared error should be non-negative"
