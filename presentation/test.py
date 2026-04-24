import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test  = pd.read_csv("y_test.csv").squeeze()

# y_train_log = np.log1p(y_train)
ridge = RidgeCV(alphas=[1], cv=5)
ridge.fit(X_train, y_train)

train_preds = ridge.predict(X_train)
test_preds  = ridge.predict(X_test)

print(f"In-sample  R²: {r2_score(y_train, train_preds):.3f}")
print(f"Out-of-sample R²: {r2_score(y_test, test_preds):.3f}")

print(f"\nTrain target — mean: {y_train.mean():.2f}%, median: {y_train.median():.2f}%")
print(f"Test  target — mean: {y_test.mean():.2f}%, median: {y_test.median():.2f}%")
print(f"\nNaive mean baseline R² on test: "
      f"{r2_score(y_test, np.full_like(y_test, y_train.mean())):.3f}")