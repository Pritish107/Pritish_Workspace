import os
import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    print("Received output_dir:", args.output_dir)

    # Ensure directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------
    # Connect MLflow to Azure Machine Learning Workspace
    # ---------------------------------------------------
    mlflow.set_tracking_uri(
        "azureml://centralindia.api.azureml.ms/mlflow/v1.0/subscriptions/e3de6195-2978-4e31-ad7e-26f75e50845d/resourceGroups/pritish_rg/providers/Microsoft.MachineLearningServices/workspaces/PritishMLWorkspace"
    )

    # Set experiment name
    mlflow.set_experiment("iris_experiment")

    with mlflow.start_run():

        # Load dataset
        iris = load_iris()
        X, y = iris.data, iris.target

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model parameters
        max_iter = 200

        # Train model
        model = LogisticRegression(max_iter=max_iter)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print("Training & Evaluation completed")
        print("Accuracy:", accuracy)

        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("test_size", 0.2)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        #mlflow.sklearn.log_model(model, artifact_path="model")

        # Save evaluation results locally
        output_file_path = os.path.join(args.output_dir, "results.txt")
        print("Writing results to:", output_file_path)

        with open(output_file_path, "w") as f:
            f.write(f"Accuracy: {accuracy}\n")
            f.write(report)

        print("Results file successfully written.")


if __name__ == "__main__":
    main()