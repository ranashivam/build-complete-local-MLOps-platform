
import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import argparse
import os
import joblib
import mlflow
import mlflow.sklearn

def main():

    # Connect this Python application to our MLflow server
    mlflow.set_tracking_uri("http://localhost:5000")

    parser = argparse.ArgumentParser(
        description="Train a simple Iris classification model."
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Percentage of data used for testing. Default: 0.30",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=100,
        help="Random state for reproducibility. Default: 100",
    )

    args = parser.parse_args()

    # Create the experiment if it doesn't exist,
    # or use it if it already exists.
    mlflow.set_experiment("Iris Classification-1")


    # Start an MLflow run.
    # Everything inside this block will belong to this run.
    with mlflow.start_run():
        print("Loading Iris dataset...")

        iris = load_iris()

        X = iris.data
        y = iris.target

        print(f"Total samples: {len(X)}")


        print("\nSplitting data...")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")

        # Log the parameters used for this training run
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)


        print("\nTraining model...")

        model = LogisticRegression(
            max_iter=200,
            random_state=args.random_state,
        )

        # Log the model configuration
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("max_iter", 200)


        model.fit(X_train, y_train)

        print("Model training completed.")

        print("\nMaking predictions...")

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        # Log the model performance to MLflow
        mlflow.log_metric("accuracy", accuracy)


        print("\nModel evaluation:")
        print(f"Test samples: {len(X_test)}")
        print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()