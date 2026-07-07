import datetime
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


BASE_DIR = Path(__file__).resolve().parent
BASE_DATASET = BASE_DIR / "dataset" / "Dataset of Diabetes .csv"
FEEDBACK_DATASET = BASE_DIR / "dataset" / "doctor_feedback.csv"
MODELS_DIR = BASE_DIR / "Models"
RESULTS_DIR = BASE_DIR / "Results"


def load_training_data():
    base_df = pd.read_csv(BASE_DATASET)
    frames = [base_df]
    if FEEDBACK_DATASET.exists():
        feedback_df = pd.read_csv(FEEDBACK_DATASET)
        if not feedback_df.empty:
            frames.append(feedback_df[base_df.columns])
    df = pd.concat(frames, ignore_index=True)
    df["CLASS"] = df["CLASS"].astype(str).str.strip()
    return df


def prepare_features(df):
    df = df.copy()
    df.drop(columns=["ID", "No_Pation"], inplace=True, errors="ignore")
    df["Gender"] = df["Gender"].astype(str).str.strip().map({"M": 0, "F": 1})
    df["Gender"] = pd.to_numeric(df["Gender"], errors="coerce").fillna(0)

    numeric_columns = ["AGE", "Urea", "Cr", "HbA1c", "Chol", "TG", "HDL", "LDL", "VLDL", "BMI"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["Chol_HDL_Ratio"] = df["Chol"] / (df["HDL"] + 0.001)
    df["LDL_HDL_Ratio"] = df["LDL"] / (df["HDL"] + 0.001)
    df["BMI_Age"] = df["BMI"] * df["AGE"]
    df["HbA1c_Age"] = df["HbA1c"] * df["AGE"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["CLASS"])
    X = df.drop("CLASS", axis=1)
    feature_names = X.columns.tolist()
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    return X, y, label_encoder, feature_names


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = load_training_data()
    X, y, label_encoder, feature_names = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_names)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_names)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    num_classes = len(label_encoder.classes_)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42)
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            random_state=42
        )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_model = None
    best_name = ""
    best_accuracy = -1.0
    rows = []

    for name, candidate in models.items():
        cv_scores = cross_val_score(candidate, X_train_scaled, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        candidate.fit(X_train_scaled, y_train)
        y_pred = candidate.predict(X_test_scaled)
        cv_accuracy = float(cv_scores.mean())
        rows.append({
            "Run_Name": "auto_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            "Model": name,
            "CV_Accuracy": cv_accuracy,
            "CV_Std": float(cv_scores.std()),
            "Test_Accuracy": float(accuracy_score(y_test, y_pred)),
            "Train_Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Training_Rows": len(df),
            "Feedback_Rows": len(pd.read_csv(FEEDBACK_DATASET)) if FEEDBACK_DATASET.exists() else 0
        })
        if cv_accuracy > best_accuracy:
            best_accuracy = cv_accuracy
            best_model = candidate
            best_name = name

    pickle.dump(best_model, open(MODELS_DIR / "best_model.pkl", "wb"))
    pickle.dump(scaler, open(MODELS_DIR / "best_scaler.pkl", "wb"))
    pickle.dump(imputer, open(MODELS_DIR / "best_imputer.pkl", "wb"))
    pickle.dump(label_encoder, open(MODELS_DIR / "label_encoder.pkl", "wb"))
    pickle.dump(feature_names, open(MODELS_DIR / "feature_names.pkl", "wb"))

    history_df = pd.DataFrame(rows)
    history_df["Is_Best_Model"] = history_df["Model"] == best_name
    history_path = RESULTS_DIR / "auto_training_history.csv"
    if history_path.exists():
        old = pd.read_csv(history_path)
        history_df = pd.concat([old, history_df], ignore_index=True)
    history_df.to_csv(history_path, index=False)

    print(f"Auto retrain completed. Best model: {best_name}, CV accuracy: {best_accuracy:.4f}, rows: {len(df)}")


if __name__ == "__main__":
    main()
