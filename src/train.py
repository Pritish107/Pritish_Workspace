import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(report)

print("Training & Evaluation completed")
print("Accuracy:", accuracy)


