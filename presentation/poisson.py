import pandas as pd
import numpy as np
from sklearn.linear_model import PoissonRegressor
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

pr = PoissonRegressor(max_iter=10000, alpha=5)

pr.fit(X_train, y_train)
pr_preds = np.clip(pr.predict(X_test), 0, None)
rep = report(
    "Model 2: Poisson Regression",
    y_test,
    pr_preds
)

print(rep)


feature_names = X_train.columns.tolist()
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": pr.coef_
}).reindex(pd.Series(pr.coef_).abs().sort_values(ascending=False).index)

print("\n  Top 10 most important features (by |coefficient|):")
print(coef_df.head(10).to_string(index=False))