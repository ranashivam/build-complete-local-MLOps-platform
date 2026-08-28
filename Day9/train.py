# ============================================================
# STEP 1 — IMPORTS
# ============================================================

import argparse

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# STEP 1 PLACEHOLDER:
# Add MLflow import here
# Example:
# import mlflow


def main():

    # ========================================================
    # STEP 2 — ARGUMENT PARSER
    # ========================================================

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

    # ========================================================
    # STEP 2 PLACEHOLDER:
    # MLflow connection goes here
    # ========================================================


    # ========================================================
    # STEP 3 — CREATE / SELECT MLflow EXPERIMENT
    # ========================================================

    # STEP 3 PLACEHOLDER:
    # Add the MLflow experiment code here
    #
    # Example:
    # mlflow.set_experiment("Iris Classification")


    # ========================================================
    # STEP 4 — START MLflow RUN
    # ========================================================

    # STEP 4 PLACEHOLDER:
    # The training code below will eventually go inside:
    #
    # with mlflow.start_run():
    #
    # Do NOT add log_param(), log_metric(), etc. here yet.


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

    # ========================================================
    # STEP 5 — LOG TRAINING PARAMETERS
    # ========================================================

    # STEP 5 PLACEHOLDER:
    # Add mlflow.log_param() calls here.
    #
    # Example:
    # mlflow.log_param("test_size", args.test_size)
    # mlflow.log_param("random_state", args.random_state)


    print("\nTraining model...")

    model = LogisticRegression(
        max_iter=200,
        random_state=args.random_state,
    )

    # ========================================================
    # STEP 6 — LOG MODEL PARAMETERS
    # ========================================================

    # STEP 6 PLACEHOLDER:
    # Add model configuration tracking here.
    #
    # Example:
    # mlflow.log_param("model", "LogisticRegression")
    # mlflow.log_param("max_iter", 200)


    model.fit(X_train, y_train)

    print("Model training completed.")

    print("\nMaking predictions...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    # ========================================================
    # STEP 7 — LOG METRIC
    # ========================================================

    # STEP 7 PLACEHOLDER:
    # Add the accuracy metric here.
    #
    # Example:
    # mlflow.log_metric("accuracy", accuracy)


    print("\nModel evaluation:")
    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()