import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib

from cleaning import preprocess_features

def main():
    # 1) Read raw data
    src = "model_features.csv"
    df = pd.read_csv(src)

    if "Views" not in df.columns:
        print("'Views' doesnt exist in csv.")
        sys.exit(0)

    # 2) Preprocessing FOR TRAINING (is_training=True)
    X = preprocess_features(df, is_training=True)
    y = pd.to_numeric(df["Views"], errors="coerce").fillna(0).astype("float32")

    # 3) Alignment & sanity checks
    assert np.isfinite(X.to_numpy()).all(), "features conțin inf/NaN"
    assert len(X) == len(y), "X and y have different lenghts"

    # 4) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 5) Model
    regressor = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1
    )

    regressor.fit(X_train, y_train)

    # 6) Evaluation
    y_pred = regressor.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Train done. RMSE={rmse:.3f} | R2={r2:.3f}")

    # 7) Save the model
    joblib.dump(regressor, "xgb_regressor.pkl")
    joblib.dump(list(X.columns), "feature_columns.pkl")
    print("Saved: xgb_regressor.pkl, feature_columns.pkl")

if __name__ == "__main__":
    main()