import argparse

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
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

    print("\nTraining model...")

    model = LogisticRegression(
        max_iter=200,
        random_state=args.random_state,
    )

    model.fit(X_train, y_train)

    print("Model training completed.")

    print("\nMaking predictions...")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nModel evaluation:")
    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()