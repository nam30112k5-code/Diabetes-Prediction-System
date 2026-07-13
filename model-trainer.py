import datetime
import hashlib
import importlib
import pickle
import sys
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
UPDATE_DATASET = BASE_DIR / "dataset" / "update" / "doctor_feedback.csv"
UPDATES_FOLDER = BASE_DIR / "dataset" / "updates"
MODELS_DIR = BASE_DIR / "Models"
RESULTS_DIR = BASE_DIR / "Results"
LATEST_WANDB_URL_FILE = RESULTS_DIR / "latest_wandb_url.txt"
WANDB_HOME_URL = "https://wandb.ai/home"
WANDB_WORKSPACE_URL = "https://wandb.ai/nam30112k5-no/diabetes-prediction/workspace?nw=nwusernam30112k5"

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5


def import_external_wandb():
    original_sys_path = sys.path[:]
    try:
        sys.path = [
            path for path in sys.path
            if str(Path(path or ".").resolve()) != str(BASE_DIR)
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
                "base_dataset": str(BASE_DATASET),
                "update_dataset": str(UPDATE_DATASET),
            },
        )
        return wandb, run
    except Exception as exc:
        print(f"W&B logging skipped: {exc}")
        return None, None


def load_update_files(base_columns):
    update_frames = []
    update_names = []

    if UPDATE_DATASET.exists():
        update_df = pd.read_csv(UPDATE_DATASET)
        update_df.columns = update_df.columns.str.strip()
        update_frames.append(update_df[base_columns])
        update_names.append(str(UPDATE_DATASET.relative_to(BASE_DIR)))

    if UPDATES_FOLDER.exists():
        for file_path in sorted(UPDATES_FOLDER.glob("*.csv")):
            update_df = pd.read_csv(file_path)
            update_df.columns = update_df.columns.str.strip()
            missing_columns = set(base_columns) - set(update_df.columns)
            if missing_columns:
                print(f"Skip {file_path.name}: missing columns {sorted(missing_columns)}")
                continue
            update_frames.append(update_df[base_columns])
            update_names.append(str(file_path.relative_to(BASE_DIR)))

    return update_frames, update_names


def load_dataset():
    base_df = pd.read_csv(BASE_DATASET)
    base_df.columns = base_df.columns.str.strip()

    update_frames, update_names = load_update_files(base_df.columns)
    df = pd.concat([base_df, *update_frames], ignore_index=True)
    df["CLASS"] = df["CLASS"].astype(str).str.strip()
    df["Gender"] = df["Gender"].astype(str).str.strip()

    dataset_hash = hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()

    combined_path = RESULTS_DIR / "combined_dataset_used_for_training.csv"
    df.to_csv(combined_path, index=False)

    print("Dataset rows:", len(df))
    print("Update files:", update_names if update_names else "none")
    print("Class distribution:")
    print(df["CLASS"].value_counts())
    print("Dataset hash:", dataset_hash)

    return df, dataset_hash, update_names


def prepare_features(df):
    df = df.copy()
    df.drop(columns=["ID", "No_Pation"], inplace=True, errors="ignore")

    df["Gender"] = df["Gender"].map({"M": 0, "F": 1})
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

    print("Class mapping:")
    for index, label in enumerate(label_encoder.classes_):
        print(f"  {label} -> {index}")

    return X, y, label_encoder, feature_names


def build_models(num_classes):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
        )
    return models


def train_models(X_train, X_test, y_train, y_test, num_classes, run_name, dataset_rows, dataset_hash, wandb):
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best_model = None
    best_name = ""
    best_accuracy = -1.0
    history_rows = []

    for name, model in build_models(num_classes).items():
        print(f"\nTraining: {name}")
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        y_pred = model.predict(X_test)

        cv_accuracy = float(cv_scores.mean())
        train_accuracy = float(accuracy_score(y_train, train_pred))
        test_accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        history_rows.append({
            "Run_Name": run_name,
            "Dataset_Rows": dataset_rows,
            "Dataset_Hash": dataset_hash,
            "Model": name,
            "CV_Accuracy": cv_accuracy,
            "CV_Std": float(cv_scores.std()),
            "Train_Accuracy": train_accuracy,
            "Test_Accuracy": test_accuracy,
            "Overfit_Gap": train_accuracy - test_accuracy,
            "Macro_Precision": report["macro avg"]["precision"],
            "Macro_Recall": report["macro avg"]["recall"],
            "Macro_F1": report["macro avg"]["f1-score"],
            "Weighted_Precision": report["weighted avg"]["precision"],
            "Weighted_Recall": report["weighted avg"]["recall"],
            "Weighted_F1": report["weighted avg"]["f1-score"],
            "Train_Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

        if wandb is not None:
            safe_name = name.replace(" ", "_")
            wandb.log({
                f"{safe_name}_CV_Accuracy": cv_accuracy,
                f"{safe_name}_Train_Accuracy": train_accuracy,
                f"{safe_name}_Test_Accuracy": test_accuracy,
                f"{safe_name}_Overfit_Gap": train_accuracy - test_accuracy,
                f"{safe_name}_Macro_F1": report["macro avg"]["f1-score"],
            })

        print(f"CV accuracy: {cv_accuracy:.4f}")
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Overfit gap: {train_accuracy - test_accuracy:.4f}")

        if cv_accuracy > best_accuracy:
            best_accuracy = cv_accuracy
            best_model = model
            best_name = name

    return best_model, best_name, best_accuracy, pd.DataFrame(history_rows)


def append_csv(path, new_df):
    if path.exists():
        old_df = pd.read_csv(path)
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.to_csv(path, index=False)


def save_outputs(best_model, scaler, imputer, label_encoder, feature_names, history_df, best_name, best_accuracy, run_name, dataset_rows, dataset_hash):
    history_df["Is_Best_Model"] = history_df["Model"] == best_name
    append_csv(RESULTS_DIR / "training_history_detail.csv", history_df)

    summary_df = pd.DataFrame([{
        "Run_Name": run_name,
        "Dataset_Rows": dataset_rows,
        "Dataset_Hash": dataset_hash,
        "Best_Model": best_name,
        "Best_CV_Accuracy": best_accuracy,
        "Train_Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }])
    append_csv(RESULTS_DIR / "training_history_summary.csv", summary_df)

    pickle.dump(best_model, open(MODELS_DIR / "best_model.pkl", "wb"))
    pickle.dump(scaler, open(MODELS_DIR / "best_scaler.pkl", "wb"))
    pickle.dump(imputer, open(MODELS_DIR / "best_imputer.pkl", "wb"))
    pickle.dump(label_encoder, open(MODELS_DIR / "label_encoder.pkl", "wb"))
    pickle.dump(feature_names, open(MODELS_DIR / "feature_names.pkl", "wb"))


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    run_name = datetime.datetime.now().strftime("diabetes_%Y%m%d_%H%M%S")
    wandb, wandb_run = start_wandb_run(run_name)

    df, dataset_hash, update_names = load_dataset()
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

    best_model, best_name, best_accuracy, history_df = train_models(
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        len(label_encoder.classes_),
        run_name,
        len(df),
        dataset_hash,
        wandb,
    )

    save_outputs(
        best_model,
        scaler,
        imputer,
        label_encoder,
        feature_names,
        history_df,
        best_name,
        best_accuracy,
        run_name,
        len(df),
        dataset_hash,
    )

    if wandb is not None and wandb_run is not None:
        wandb.config.update({
            "dataset_rows": len(df),
            "dataset_hash": dataset_hash,
            "update_files": update_names,
        })
        wandb.summary["Best_Model"] = best_name
        wandb.summary["Best_CV_Accuracy"] = best_accuracy
        wandb.log({"Training_History_Table": wandb.Table(dataframe=history_df)})
        LATEST_WANDB_URL_FILE.write_text(WANDB_WORKSPACE_URL, encoding="utf-8")
        print("W&B Run URL:", wandb_run.get_url())
        print("W&B Workspace:", WANDB_WORKSPACE_URL)
        wandb.finish()
    else:
        LATEST_WANDB_URL_FILE.write_text(WANDB_WORKSPACE_URL, encoding="utf-8")
        print("W&B:", WANDB_WORKSPACE_URL)

    print("\nBest model:", best_name)
    print("Best CV accuracy:", round(best_accuracy, 4))
    print("Saved model files in Models/")
    print("Saved history files in Results/")


if __name__ == "__main__":
    main()
