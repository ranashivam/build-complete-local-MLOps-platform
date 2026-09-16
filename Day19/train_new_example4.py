import argparse
import json
import os
import joblib
import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from kfp import dsl, compiler
from kfp.dsl import Dataset, Model, Metrics


# ============================================================
# ORIGINAL APPLICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # MLflow tracking URI
    #
    # Local execution:
    #   http://localhost:5000
    #
    # Kubernetes execution:
    #   MLFLOW_TRACKING_URI is set by the Kubeflow task
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://localhost:5000"
        )
    )

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
    mlflow.set_experiment(
        "Iris Classification-1"
    )

    # Start an MLflow run.
    with mlflow.start_run():

        print("Loading Iris dataset...")

        iris = load_iris()

        X = iris.data
        y = iris.target

        print(
            f"Total samples: {len(X)}"
        )

        print("\nSplitting data...")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
        )

        print(
            f"Training samples: {len(X_train)}"
        )

        print(
            f"Testing samples: {len(X_test)}"
        )

        # Log parameters
        mlflow.log_param(
            "test_size",
            args.test_size
        )

        mlflow.log_param(
            "random_state",
            args.random_state
        )

        print("\nTraining model...")

        model = LogisticRegression(
            max_iter=200,
            random_state=args.random_state,
        )

        mlflow.log_param(
            "model",
            "LogisticRegression"
        )

        mlflow.log_param(
            "max_iter",
            200
        )

        model.fit(
            X_train,
            y_train
        )

        print(
            "Model training completed."
        )

        print("\nMaking predictions...")

        predictions = model.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        # Log metric
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        # ----------------------------------------------------
        # Create artifacts
        # ----------------------------------------------------

        os.makedirs(
            "artifacts",
            exist_ok=True
        )

        # Save trained model
        model_path = "artifacts/iris_model.pkl"

        joblib.dump(
            model,
            model_path
        )

        mlflow.log_artifact(
            model_path
        )

        # Create experiment report
        report_path = "artifacts/experiment_report.txt"

        with open(
            report_path,
            "w"
        ) as file:

            file.write(
                "Iris Classification Experiment\n"
            )

            file.write(
                "==============================\n\n"
            )

            file.write(
                f"Test Size: {args.test_size}\n"
            )

            file.write(
                f"Random State: {args.random_state}\n"
            )

            file.write(
                "Model: LogisticRegression\n"
            )

            file.write(
                "Max Iterations: 200\n"
            )

            file.write(
                f"Accuracy: {accuracy:.4f}\n"
            )

        mlflow.log_artifact(
            report_path
        )

        print("\nArtifacts logged to MLflow:")

        print(
            f"- {model_path}"
        )

        print(
            f"- {report_path}"
        )

        print("\nModel evaluation:")

        print(
            f"Test samples: {len(X_test)}"
        )

        print(
            f"Accuracy: {accuracy:.4f}"
        )


# ============================================================
# KUBEFLOW RUNTIME FUNCTIONS
# ============================================================

def prepare_data_runtime(output_path):

    print("Preparing Iris dataset...")

    iris = load_iris()

    X = iris.data
    y = iris.target

    # Write directly to the KFP artifact path.
    # Using a file handle prevents numpy from
    # automatically appending ".npz" to the path.
    with open(
        output_path,
        "wb"
    ) as file:

        np.savez(
            file,
            X=X,
            y=y,
        )

    print(
        "Dataset prepared successfully."
    )

    print(
        f"Total samples: {len(X)}"
    )

    print(
        f"Dataset saved to: {output_path}"
    )


def train_model_runtime(
    dataset_path,
    model_path,
):

    print("Loading prepared dataset...")

    data = np.load(
        dataset_path
    )

    X = data["X"]
    y = data["y"]

    print("Dataset loaded successfully.")

    # Intentional error for Video 19 debugging demonstration
    raise RuntimeError(
        "Intentional training failure for Video 19 debugging demo"
    )

    print("Splitting data...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=100,
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples: {len(X_test)}"
    )

    print(
        "Training LogisticRegression model..."
    )

    model = LogisticRegression(
        max_iter=200,
        random_state=100,
    )

    model.fit(
        X_train,
        y_train
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        "Model training completed."
    )

    print(
        f"Model saved to: {model_path}"
    )


def evaluate_model_runtime(
    dataset_path,
    model_path,
    metrics_path,
):

    print("Loading dataset...")

    data = np.load(
        dataset_path
    )

    X = data["X"]
    y = data["y"]

    print(
        "Loading trained model..."
    )

    model = joblib.load(
        model_path
    )

    print(
        "Creating test split..."
    )

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=100,
    )

    print(
        "Making predictions..."
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    # Save metrics for the next pipeline stage.
    with open(
        metrics_path,
        "w"
    ) as file:

        json.dump(
            {
                "accuracy": accuracy
            },
            file,
        )

    print(
        f"Metrics saved to: {metrics_path}"
    )


def log_mlflow_runtime(
    model_path,
    metrics_path,
):

    print(
        "Connecting to MLflow..."
    )

    # Local execution:
    #   http://localhost:5000
    #
    # Kubeflow execution:
    #   MLFLOW_TRACKING_URI is supplied
    #   by the pipeline task.
    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://localhost:5000"
        )
    )

    mlflow.set_experiment(
        "Iris Classification-1"
    )

    print(
        "Starting MLflow run..."
    )

    with mlflow.start_run():

        # Log parameters

        mlflow.log_param(
            "test_size",
            0.30
        )

        mlflow.log_param(
            "random_state",
            100
        )

        mlflow.log_param(
            "model",
            "LogisticRegression"
        )

        mlflow.log_param(
            "max_iter",
            200
        )

        # Load evaluation metrics

        with open(
            metrics_path,
            "r"
        ) as file:

            metrics = json.load(
                file
            )

        accuracy = metrics["accuracy"]

        # Log metric

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        # Log model artifact

        mlflow.log_artifact(
            model_path
        )

        print(
            f"Accuracy logged to MLflow: {accuracy:.4f}"
        )

        print(
            "Model artifact logged to MLflow."
        )

        print(
            "MLflow run completed."
        )


# ============================================================
# KUBEFLOW COMPONENT 1 -- Prepare Data
# ============================================================

@dsl.container_component
def prepare_data(
    dataset: dsl.Output[Dataset],
):

    return dsl.ContainerSpec(

        image="shivamrana28/iris-m1-app:v16",

        command=[
            "python",
            "-c",
            "import sys, train_new; "
            "train_new.prepare_data_runtime(sys.argv[1])",
        ],

        args=[
            dataset.path,
        ],
    )


# ============================================================
# KUBEFLOW COMPONENT 2 -- Train Model
# ============================================================

@dsl.container_component
def train_model(
    dataset: dsl.Input[Dataset],
    model: dsl.Output[Model],
):

    return dsl.ContainerSpec(

        image="shivamrana28/iris-m1-app:v16",

        command=[
            "python",
            "-c",
            "import sys, train_new; "
            "train_new.train_model_runtime(sys.argv[1], sys.argv[2])",
        ],

        args=[
            dataset.path,
            model.path,
        ],
    )


# ============================================================
# KUBEFLOW COMPONENT 3 -- Evaluate Model
# ============================================================

@dsl.container_component
def evaluate_model(
    dataset: dsl.Input[Dataset],
    model: dsl.Input[Model],
    metrics: dsl.Output[Metrics],
):

    return dsl.ContainerSpec(

        image="shivamrana28/iris-m1-app:v16",

        command=[
            "python",
            "-c",
            "import sys, train_new; "
            "train_new.evaluate_model_runtime("
            "sys.argv[1], "
            "sys.argv[2], "
            "sys.argv[3]"
            ")",
        ],

        args=[
            dataset.path,
            model.path,
            metrics.path,
        ],
    )


# ============================================================
# KUBEFLOW COMPONENT 4 -- Log to MLflow
# ============================================================

@dsl.container_component
def log_mlflow(
    model: dsl.Input[Model],
    metrics: dsl.Input[Metrics],
):

    return dsl.ContainerSpec(

        image="shivamrana28/iris-m1-app:v16",

        command=[
            "python",
            "-c",
            "import sys, train_new; "
            "train_new.log_mlflow_runtime("
            "sys.argv[1], "
            "sys.argv[2]"
            ")",
        ],

        args=[
            model.path,
            metrics.path,
        ],
    )


# ============================================================
# KUBEFLOW PIPELINE
# ============================================================

@dsl.pipeline(
    name="iris-pipeline"
)
def iris_pipeline():

    prepare = prepare_data()

    train = train_model(
        dataset=prepare.outputs["dataset"]
    )

    evaluate = evaluate_model(
        dataset=prepare.outputs["dataset"],
        model=train.outputs["model"]
    )

    log_task = log_mlflow(
        model=train.outputs["model"],
        metrics=evaluate.outputs["metrics"]
    )

    # MLflow is running inside the Kubernetes cluster,
    # so only this Kubeflow task receives the Kubernetes URI.
    log_task.set_env_variable(
        "MLFLOW_TRACKING_URI",
        "http://mlflow-service.kubeflow:5000"
    )


# ============================================================
# RUN APPLICATION OR COMPILE PIPELINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the Kubeflow pipeline instead of running training.",
    )

    args, remaining_args = parser.parse_known_args()

    if args.compile:

        compiler.Compiler().compile(
            pipeline_func=iris_pipeline,
            package_path="iris_pipeline.yaml",
        )

        print(
            "Pipeline compiled successfully!"
        )

        print(
            "Generated file: iris_pipeline.yaml"
        )

    else:

        # Pass remaining command-line arguments
        # to the original application.

        import sys

        sys.argv = [
            sys.argv[0]
        ] + remaining_args

        main()