import argparse
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():

    # -----------------------------------------
    # Command-line arguments
    # -----------------------------------------

    parser = argparse.ArgumentParser(
        description="Train Iris model and track experiment with MLflow"
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=100
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=200
    )

    args = parser.parse_args()

    # -----------------------------------------
    # Connect to MLflow
    # -----------------------------------------

    mlflow.set_tracking_uri("http://localhost:5000")

    mlflow.set_experiment("Iris Classification")

    print("Connected to MLflow")
    print("Experiment: Iris Classification")

    # -----------------------------------------
    # Start MLflow run
    # -----------------------------------------

    with mlflow.start_run():

        print("\nLoading Iris dataset...")

        iris = load_iris()

        X = iris.data
        y = iris.target

        print(f"Total samples: {len(X)}")

        # -----------------------------------------
        # Train / Test Split
        # -----------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")

        # -----------------------------------------
        # Log parameters
        # -----------------------------------------

        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("model", "LogisticRegression")

        # -----------------------------------------
        # Train model
        # -----------------------------------------

        print("\nTraining model...")

        model = LogisticRegression(
            max_iter=args.max_iter,
            random_state=args.random_state
        )

        model.fit(X_train, y_train)

        print("Training completed.")

        # -----------------------------------------
        # Make predictions
        # -----------------------------------------

        print("\nMaking predictions...")

        predictions = model.predict(X_test)

        # -----------------------------------------
        # Calculate metric
        # -----------------------------------------

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(f"Accuracy: {accuracy:.4f}")

        # -----------------------------------------
        # Log metric
        # -----------------------------------------

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        # -----------------------------------------
        # Save model artifact
        # -----------------------------------------

        os.makedirs("artifacts", exist_ok=True)

        model_path = "artifacts/iris_model.pkl"

        joblib.dump(
            model,
            model_path
        )

        mlflow.log_artifact(
            model_path
        )

        # -----------------------------------------
        # Create experiment report
        # -----------------------------------------

        report_path = "artifacts/experiment_report.txt"

        with open(report_path, "w") as file:

            file.write("Iris Classification Experiment\n")
            file.write("==============================\n\n")

            file.write(f"Test Size: {args.test_size}\n")
            file.write(f"Random State: {args.random_state}\n")
            file.write(f"Max Iterations: {args.max_iter}\n")
            file.write("Model: LogisticRegression\n")
            file.write(f"Accuracy: {accuracy:.4f}\n")

        # -----------------------------------------
        # Log report artifact
        # -----------------------------------------

        mlflow.log_artifact(
            report_path
        )

        print("\nExperiment completed successfully.")

        print("\nMLflow information:")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Accuracy: {accuracy:.4f}")

        print("\nOpen MLflow:")
        print("http://localhost:5000")


if __name__ == "__main__":
    main()