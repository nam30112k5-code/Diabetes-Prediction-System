import datetime
import importlib
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


BASE_DIR = Path(__file__).resolve().parent
BASE_DATASET = BASE_DIR / "dataset" / "Dataset of Diabetes .csv"
FEEDBACK_DATASET = BASE_DIR / "dataset" / "update" / "doctor_feedback.csv"
MODELS_DIR = BASE_DIR / "Models"
RESULTS_DIR = BASE_DIR / "Results"
COMBINED_DATASET_FILE = RESULTS_DIR / "combined_dataset_used_for_training.csv"
LATEST_WANDB_URL_FILE = RESULTS_DIR / "latest_wandb_url.txt"
LOCAL_TRAINING_DASHBOARD_URL = "http://localhost:5000/training-dashboard"
WANDB_HOME_URL = "https://wandb.ai/home"
WANDB_WORKSPACE_URL = "https://wandb.ai/nam30112k5-no/diabetes-prediction/workspace?nw=nwusernam30112k5"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5


def import_external_wandb():
    project_dir = str(BASE_DIR)
    original_sys_path = sys.path[:]
    try:
        sys.path = [
            path for path in sys.path
            if str(Path(path or ".").resolve()) != project_dir
        ]
        return importlib.import_module("wandb")
    finally:
        sys.path = original_sys_path


def start_wandb_run(run_name):
    try:
        wandb = import_external_wandb()
        run = wandb.init(
            project="diabetes-prediction",
            name=run_name,
            config={
                "test_size": TEST_SIZE,
                "random_state": RANDOM_STATE,
                "cv_folds": CV_FOLDS,
                "feedback_dataset": str(FEEDBACK_DATASET),
                "base_dataset": str(BASE_DATASET)
            }
        )
        return wandb, run
    except Exception as exc:
        print(f"W&B logging skipped: {exc}")
        return None, None


def get_feedback_row_count():
    if not FEEDBACK_DATASET.exists():
        return 0
    return len(pd.read_csv(FEEDBACK_DATASET))


def load_training_data():
    base_df = pd.read_csv(BASE_DATASET)
    frames = [base_df]
    if FEEDBACK_DATASET.exists():
        feedback_df = pd.read_csv(FEEDBACK_DATASET)
        if not feedback_df.empty:
            frames.append(feedback_df[base_df.columns])
    df = pd.concat(frames, ignore_index=True)
    df["CLASS"] = df["CLASS"].astype(str).str.strip()
    COMBINED_DATASET_FILE.parent.mkdir(exist_ok=True)
    df.to_csv(COMBINED_DATASET_FILE, index=False)
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


def log_to_wandb(wandb, rows, best_name, best_accuracy, training_rows):
    try:
        wandb.summary["Training_Rows"] = training_rows
        wandb.summary["Feedback_Rows"] = rows[0]["Feedback_Rows"] if rows else 0
        wandb.summary["Best_Model"] = best_name
        wandb.summary["Best_CV_Accuracy"] = best_accuracy
        metric_payload = {}
        for row in rows:
            safe_name = row["Model"].replace(" ", "_")
            metric_payload.update({
                f"{safe_name}_CV_Accuracy": row["CV_Accuracy"],
                f"{safe_name}_CV_Std": row["CV_Std"],
                f"{safe_name}_Train_Accuracy": row["Train_Accuracy"],
                f"{safe_name}_Test_Accuracy": row["Test_Accuracy"],
                f"{safe_name}_Overfit_Gap": row["Overfit_Gap"],
                f"{safe_name}_Macro_F1": row["Macro_F1"],
                f"{safe_name}_Weighted_F1": row["Weighted_F1"],
            })
        if metric_payload:
            wandb.log(metric_payload)
        wandb.log({"Auto_Training_History": wandb.Table(dataframe=pd.DataFrame(rows))})
    except Exception as exc:
        print(f"W&B link/logging failed: {exc}", flush=True)
    finally:
        wandb.finish()


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    run_name = "auto_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb, wandb_run = start_wandb_run(run_name)

    df = load_training_data()
    X, y, label_encoder, feature_names = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
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
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE
        )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best_model = None
    best_name = ""
    best_accuracy = -1.0
    rows = []
    feedback_rows = get_feedback_row_count()

    for name, candidate in models.items():
        cv_scores = cross_val_score(candidate, X_train_scaled, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        candidate.fit(X_train_scaled, y_train)
        train_pred = candidate.predict(X_train_scaled)
        y_pred = candidate.predict(X_test_scaled)
        cv_accuracy = float(cv_scores.mean())
        train_accuracy = float(accuracy_score(y_train, train_pred))
        test_accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        rows.append({
            "Run_Name": run_name,
            "Model": name,
            "CV_Accuracy": cv_accuracy,
            "CV_Std": float(cv_scores.std()),
            "Train_Accuracy": train_accuracy,
            "Test_Accuracy": test_accuracy,
            "Overfit_Gap": train_accuracy - test_accuracy,
            "Macro_F1": report["macro avg"]["f1-score"],
            "Weighted_F1": report["weighted avg"]["f1-score"],
            "Train_Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Training_Rows": len(df),
            "Feedback_Rows": feedback_rows
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
    wandb_url = wandb_run.get_url() if wandb_run is not None else ""
    history_df["WandB_Run_URL"] = wandb_url
    history_path = RESULTS_DIR / "auto_training_history.csv"
    if history_path.exists():
        old = pd.read_csv(history_path)
        history_df = pd.concat([old, history_df], ignore_index=True)
    history_df.to_csv(history_path, index=False)
    if wandb_run is not None:
        LATEST_WANDB_URL_FILE.write_text(WANDB_WORKSPACE_URL, encoding="utf-8")
        print(f"W&B Run URL: {wandb_url}", flush=True)
        print(f"W&B Workspace: {WANDB_WORKSPACE_URL}", flush=True)
        log_to_wandb(wandb, rows, best_name, best_accuracy, len(df))
    else:
        LATEST_WANDB_URL_FILE.write_text(WANDB_WORKSPACE_URL, encoding="utf-8")
        print(f"W&B: {WANDB_WORKSPACE_URL}", flush=True)

    print(f"Auto retrain completed. Best model: {best_name}, CV accuracy: {best_accuracy:.4f}, rows: {len(df)}")
    print(f"Training dashboard: {LOCAL_TRAINING_DASHBOARD_URL}", flush=True)


if __name__ == "__main__":
    main()
