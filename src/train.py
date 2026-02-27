import os
import argparse
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

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Training & Evaluation completed")
    print("Accuracy:", accuracy)

    # Write output file
    output_file_path = os.path.join(args.output_dir, "results.txt")
    print("Writing results to:", output_file_path)

    with open(output_file_path, "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(report)

    print("Results file successfully written.")


if __name__ == "__main__":
    main()

