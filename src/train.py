from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

print("Loading Iris dataset...")
iris = load_iris()
X = iris.data
y = iris.target

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

print("Testing model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("===== RESULTS =====")
print("Accuracy:", accuracy)
print("Classification Report:\n", report)

