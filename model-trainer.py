import pandas as pd
import numpy as np
import pickle
import os
import importlib
import sys
import datetime
import hashlib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


def import_external_wandb():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    original_sys_path = sys.path[:]

    try:
        sys.path = [
            path for path in sys.path
            if os.path.abspath(path or os.getcwd()) != project_dir
        ]
        return importlib.import_module("wandb")
    finally:
        sys.path = original_sys_path


wandb = import_external_wandb()


# =========================
# CREATE FOLDERS
# =========================

os.makedirs("Models", exist_ok=True)
os.makedirs("Results", exist_ok=True)
os.makedirs("dataset/updates", exist_ok=True)


# =========================
# W&B START
# =========================

run_name = datetime.datetime.now().strftime(
    "diabetes_%Y%m%d_%H%M%S"
)

wandb.init(
    project="diabetes-prediction",
    name=run_name,
    config={
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 5,
        "models": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ]
    }
)


# =========================
# LOAD MAIN DATA + UPDATE DATA
# =========================

print("Loading dataset...")

main_dataset_path = "dataset/Dataset of Diabetes .csv"
updates_folder = "dataset/updates"

df_main = pd.read_csv(main_dataset_path)

df_main.columns = df_main.columns.str.strip()

print("Main dataset rows:", df_main.shape[0])

dataframes = [df_main]

update_files = []

if os.path.exists(updates_folder):

    for file_name in os.listdir(updates_folder):

        if file_name.endswith(".csv"):

            file_path = os.path.join(
                updates_folder,
                file_name
            )

            print("Loading update file:", file_path)

            df_update = pd.read_csv(file_path)

            df_update.columns = df_update.columns.str.strip()

            missing_columns = set(df_main.columns) - set(df_update.columns)

            if missing_columns:
                print("Skip file because missing columns:", file_name)
                print("Missing:", missing_columns)
                continue

            df_update = df_update[df_main.columns]

            print("Update rows:", df_update.shape[0])

            dataframes.append(df_update)
            update_files.append(file_name)

else:
    print("Updates folder not found:", updates_folder)


df = pd.concat(
    dataframes,
    ignore_index=True
)

df["CLASS"] = df["CLASS"].astype(str).str.strip()
df["Gender"] = df["Gender"].astype(str).str.strip()

print("\nUnique CLASS labels:")
print(df["CLASS"].unique())

print("\nFinal dataset shape:", df.shape)

print("\nClass distribution:")
print(df["CLASS"].value_counts())

dataset_hash = hashlib.md5(
    pd.util.hash_pandas_object(df, index=True).values
).hexdigest()

print("\nDataset Hash:", dataset_hash)

combined_dataset_path = "Results/combined_dataset_used_for_training.csv"

df.to_csv(
    combined_dataset_path,
    index=False
)

print("\nSaved combined dataset:", combined_dataset_path)

wandb.config.update({
    "dataset_rows": df.shape[0],
    "dataset_columns": df.shape[1],
    "dataset_hash": dataset_hash,
    "update_files": update_files,
    "combined_dataset_path": combined_dataset_path
})

print("\nPreview:")
print(df.head())


# =========================
# DROP USELESS COLUMNS
# =========================

df.drop(
    columns=[
        "ID",
        "No_Pation"
    ],
    inplace=True
)


# =========================
# ENCODE GENDER
# =========================

df["Gender"] = df["Gender"].map({
    "M": 0,
    "F": 1
})


# =========================
# FEATURE ENGINEERING
# =========================

df["Chol_HDL_Ratio"] = (
    df["Chol"] /
    (df["HDL"] + 0.001)
)

df["LDL_HDL_Ratio"] = (
    df["LDL"] /
    (df["HDL"] + 0.001)
)

df["BMI_Age"] = (
    df["BMI"] *
    df["AGE"]
)

df["HbA1c_Age"] = (
    df["HbA1c"] *
    df["AGE"]
)


# =========================
# ENCODE CLASS
# =========================

label_encoder = LabelEncoder()

df["CLASS"] = label_encoder.fit_transform(
    df["CLASS"]
)

print("\nClass Mapping:")

for i, label in enumerate(label_encoder.classes_):
    print(label, "->", i)


# =========================
# SPLIT X AND Y
# =========================

X = df.drop(
    "CLASS",
    axis=1
)

y = df["CLASS"]

feature_names = X.columns.tolist()

X.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# HANDLE MISSING VALUES
# =========================

imputer = SimpleImputer(
    strategy="median"
)

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=feature_names
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=feature_names
)


# =========================
# STANDARDIZE DATA
# =========================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# =========================
# CROSS VALIDATION
# =========================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# =========================
# DEFINE MODELS
# =========================

num_classes = len(label_encoder.classes_)

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=3000
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),

    "XGBoost":
        XGBClassifier(
            objective="multi:softprob",
            num_class=num_classes,
            eval_metric="mlogloss",
            random_state=42
        )
}


# =========================
# TRAIN MODELS
# =========================

best_model = None
best_name = ""
best_accuracy = 0

training_history = []


for name, model in models.items():

    print("\n" + "=" * 50)
    print("Training:", name)

    cv_scores = cross_val_score(
        model,
        X_train_scaled,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    cv_accuracy = cv_scores.mean()
    cv_std = cv_scores.std()

    print(
        "CV Accuracy:",
        round(cv_accuracy, 4),
        "+/-",
        round(cv_std, 4)
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    y_pred = model.predict(
        X_test_scaled
    )

    test_accuracy = accuracy_score(
        y_test,
        y_pred
    )

    report_text = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(
        "Test Accuracy:",
        round(test_accuracy, 4)
    )

    print(report_text)

    print(cm)

    safe_name = name.replace(" ", "_")

    wandb.log({
        f"{safe_name}_CV_Accuracy": cv_accuracy,
        f"{safe_name}_CV_Std": cv_std,
        f"{safe_name}_Test_Accuracy": test_accuracy,
        f"{safe_name}_Macro_F1": report_dict["macro avg"]["f1-score"],
        f"{safe_name}_Weighted_F1": report_dict["weighted avg"]["f1-score"]
    })

    training_history.append({
        "Run_Name": run_name,
        "Dataset_Rows": df.shape[0],
        "Dataset_Hash": dataset_hash,
        "Model": name,
        "CV_Accuracy": cv_accuracy,
        "CV_Std": cv_std,
        "Test_Accuracy": test_accuracy,
        "Macro_Precision": report_dict["macro avg"]["precision"],
        "Macro_Recall": report_dict["macro avg"]["recall"],
        "Macro_F1": report_dict["macro avg"]["f1-score"],
        "Weighted_Precision": report_dict["weighted avg"]["precision"],
        "Weighted_Recall": report_dict["weighted avg"]["recall"],
        "Weighted_F1": report_dict["weighted avg"]["f1-score"],
        "Train_Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    if cv_accuracy > best_accuracy:

        best_accuracy = cv_accuracy
        best_model = model
        best_name = name


# =========================
# SAVE TRAINING HISTORY
# =========================

history_df = pd.DataFrame(training_history)

history_df["Is_Best_Model"] = history_df["Model"] == best_name

detail_history_file = "Results/training_history_detail.csv"

if os.path.exists(detail_history_file):

    old_detail_history = pd.read_csv(detail_history_file)

    final_detail_history = pd.concat(
        [old_detail_history, history_df],
        ignore_index=True
    )

else:

    final_detail_history = history_df

final_detail_history.to_csv(
    detail_history_file,
    index=False
)


summary_row = pd.DataFrame([{
    "Run_Name": run_name,
    "Dataset_Rows": df.shape[0],
    "Dataset_Hash": dataset_hash,
    "Best_Model": best_name,
    "Best_CV_Accuracy": best_accuracy,
    "Train_Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
}])

summary_history_file = "Results/training_history_summary.csv"

if os.path.exists(summary_history_file):

    old_summary_history = pd.read_csv(summary_history_file)

    final_summary_history = pd.concat(
        [old_summary_history, summary_row],
        ignore_index=True
    )

else:

    final_summary_history = summary_row

final_summary_history.to_csv(
    summary_history_file,
    index=False
)


# =========================
# SAVE BEST MODEL
# =========================

pickle.dump(
    best_model,
    open(
        "Models/best_model.pkl",
        "wb"
    )
)

pickle.dump(
    scaler,
    open(
        "Models/best_scaler.pkl",
        "wb"
    )
)

pickle.dump(
    imputer,
    open(
        "Models/best_imputer.pkl",
        "wb"
    )
)

pickle.dump(
    label_encoder,
    open(
        "Models/label_encoder.pkl",
        "wb"
    )
)

pickle.dump(
    feature_names,
    open(
        "Models/feature_names.pkl",
        "wb"
    )
)


# =========================
# SAVE FINAL RESULT TO W&B
# =========================

wandb.summary["Dataset_Rows"] = df.shape[0]
wandb.summary["Dataset_Hash"] = dataset_hash
wandb.summary["Best_Model"] = best_name
wandb.summary["Best_CV_Accuracy"] = best_accuracy

wandb.log({
    "Training_History_Table": wandb.Table(
        dataframe=history_df
    )
})


# =========================
# FINISH
# =========================

print("\nW&B Run URL:", wandb.run.get_url())

wandb.finish()


# =========================
# PRINT FINAL RESULT
# =========================

print("\n" + "=" * 50)

print("BEST MODEL:", best_name)

print(
    "BEST CV ACCURACY:",
    round(best_accuracy, 4)
)

print("\nDataset rows used:", df.shape[0])
print("Dataset hash:", dataset_hash)

print("\nSaved model files:")
print("Models/best_model.pkl")
print("Models/best_scaler.pkl")
print("Models/best_imputer.pkl")
print("Models/label_encoder.pkl")
print("Models/feature_names.pkl")

print("\nSaved history files:")
print("Results/training_history_detail.csv")
print("Results/training_history_summary.csv")
print("Results/combined_dataset_used_for_training.csv")
