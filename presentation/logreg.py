from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
import pandas as pd
X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()  # squeeze to Series
y_test  = pd.read_csv("y_test.csv").squeeze()

threshold = y_train.quantile(0.75)
print(f"Elevated layoff threshold: {threshold:.2f}%")

y_train_cls = (y_train > threshold).astype(int)
y_test_cls  = (y_test  > threshold).astype(int)

clf = LogisticRegressionCV(cv=5, max_iter=1000)
clf.fit(X_train, y_train_cls)

print(classification_report(y_test_cls, clf.predict(X_test)))