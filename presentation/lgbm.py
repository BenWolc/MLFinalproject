import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")
X_validate = pd.read_csv("X_validate.csv")
y_train = pd.read_csv("y_train.csv").squeeze()  # squeeze to Series
y_test  = pd.read_csv("y_test.csv").squeeze()
y_validate = pd.read_csv("y_validate.csv").squeeze()

# Feature scaling
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
onehot_cols = [c for c in numeric_cols if c.startswith("co_")]
scale_cols  = [c for c in numeric_cols if c not in onehot_cols]

scaler = StandardScaler()
X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_validate[scale_cols] = scaler.fit_transform(X_validate[scale_cols])
X_test[scale_cols]  = scaler.transform(X_test[scale_cols])

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

alphas = [0.1, 1, 5, 10, 100]

best_alpha, best_score = None, -np.inf

for alpha in alphas:
    model = LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, reg_lambda=alpha)
    model.fit(X_train, y_train)
    model_preds = model.predict(X_validate)
    score = r2_score(y_validate, model_preds)
    if score > best_score:
        best_score = score
        best_alpha = alpha

final_model = LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, reg_lambda=best_alpha)

final_model.fit(X_train, y_train)

model_preds = final_model.predict(X_test)

rep = report(
    "Model 3: XGBoost Predictions",
    y_test,
    model_preds
)

print(rep)