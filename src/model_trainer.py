from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import pickle
import time
import os


def train_model(X, y, n_estimators=300, test_size=0.2):
    """Train the RandomForest model and return it."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model with a timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = f'../data/models/lawn_score_model_{timestamp}.pkl'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    print(f"Model saved to {model_path}")
    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Feature importance
    feature_importances = model.feature_importances_
    print("Feature Importances:", feature_importances)

    return mse


# Example usage
if __name__ == "__main__":
    # Placeholder for training data
    X, y = [[...]], [...]  # Replace with actual data
    model, X_test, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)