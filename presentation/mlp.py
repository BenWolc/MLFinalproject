from sklearn.neural_network import MLPRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()  # squeeze to Series
y_test  = pd.read_csv("y_test.csv").squeeze()

def report(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n{'─'*45}")
    print(f"  {name}")
    print(f"{'─'*45}")
    print(f"  MAE  : {mae:.4f}%")
    print(f"  RMSE : {rmse:.4f}%")
    print(f"  R²   : {r2:.4f}")
    return {"Model": name, "MAE": round(mae,4), "RMSE": round(rmse,4), "R2": round(r2,4)}

mlp = MLPRegressor(alpha=20)

mlp.fit(X=X_train, y=y_train)
preds = np.clip(mlp.predict(X_test), 0, None)

rep = report(
    "Model 7: Multi-Layer Perceptron",
    y_test,
    preds
)

print(rep)