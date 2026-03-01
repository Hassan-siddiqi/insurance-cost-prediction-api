import os
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/insurance.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

def main():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Separate features and target
    X = df.drop(columns=["charges"])
    y = df["charges"]

    # Define columns
    num_cols = ["age", "bmi", "children"]
    cat_cols = ["sex", "smoker", "region"]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # Model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    # Pipeline
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("✅ Training completed")
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.3f}")

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    dump(pipeline, MODEL_PATH)

    print(f"✅ Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    main()