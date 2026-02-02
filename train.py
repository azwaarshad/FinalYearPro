import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Loading Data
iris = load_iris()
X = iris.data
y = iris.target

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

joblib.dump(model, 'iris_model.pkl')
joblib.dump({'accuracy': acc, 'matrix': matrix}, 'model_metrics.pkl')

print(f"Success! Model trained with {acc*100:.2f}% accuracy.")
print("Files saved: 'iris_model.pkl' and 'model_metrics.pkl'")