from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import pickle
import pyodbc
import numpy as np
import os
import subprocess
import sys
import threading
import unicodedata
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
FEEDBACK_DATASET = BASE_DIR / "dataset" / "update" / "doctor_feedback.csv"
TRAINING_HISTORY_FILE = BASE_DIR / "Results" / "auto_training_history.csv"
LATEST_WANDB_URL_FILE = BASE_DIR / "Results" / "latest_wandb_url.txt"
RETRAIN_SCRIPT = BASE_DIR / "retrain_auto.py"
LOCAL_TRAINING_DASHBOARD_URL = "http://localhost:5000/training-dashboard"
LOCAL_TRAINING_HISTORY_URL = "http://localhost:5000/training-history"
WANDB_HOME_URL = "https://wandb.ai/home"
TRAIN_LOCK = threading.Lock()
TRAINING_STATUS = {"running": False, "last_status": "idle", "last_message": ""}

def calibrate_probabilities(probs, temperature=2.5):
    """
    Apply temperature scaling to make extreme probabilities less extreme.
    Higher temperature = more spread out probabilities
    """
    # Convert to log space, apply temperature, convert back
    if np.min(probs) <= 0 or np.max(probs) >= 1:
        return probs
    
    # Clip extreme values slightly to avoid log(0)
    probs_clipped = np.clip(probs, 1e-6, 1 - 1e-6)
    
    # Apply temperature scaling in log space
    log_probs = np.log(probs_clipped)
    scaled_log_probs = log_probs / temperature
    
    # Convert back to probabilities
    scaled_probs = np.exp(scaled_log_probs)
    scaled_probs = scaled_probs / np.sum(scaled_probs)  # Renormalize
    
    return scaled_probs

def get_json_value(data, *keys):
    for key in keys:
        if key in data:
            return data[key]
    return None


def normalize_json(data):
    if not isinstance(data, dict):
        return {}
    normalized = {str(k).lower(): v for k, v in data.items()}
    # map common alias fields
    if 'idl' in normalized and 'ldl' not in normalized:
        normalized['ldl'] = normalized['idl']
    if 'recordid' in normalized and 'record_id' not in normalized:
        normalized['record_id'] = normalized['recordid']
    if 'healthrecordid' in normalized and 'health_record_id' not in normalized:
        normalized['health_record_id'] = normalized['healthrecordid']
    return normalized


def require_fields(data, fields):
    missing = [field for field in fields if get_json_value(data, field) is None]
    if missing:
        raise ValueError('Missing JSON fields: ' + ', '.join(missing))

# Load các file model (Đảm bảo đường dẫn đúng)
def load_model_artifacts():
    global model, scaler, label_encoder, feature_names
    model = pickle.load(open(BASE_DIR / "Models" / "best_model.pkl", "rb"))
    scaler = pickle.load(open(BASE_DIR / "Models" / "best_scaler.pkl", "rb"))
    label_encoder = pickle.load(open(BASE_DIR / "Models" / "label_encoder.pkl", "rb"))
    feature_names = pickle.load(open(BASE_DIR / "Models" / "feature_names.pkl", "rb"))


load_model_artifacts()


def normalize_diagnosis_to_class(diagnosis):
    value = str(diagnosis or "").strip().lower()
    plain = unicodedata.normalize("NFD", value)
    plain = "".join(ch for ch in plain if unicodedata.category(ch) != "Mn")
    plain = plain.replace("đ", "d").replace("Đ", "d")
    if value in ("y", "diabetes", "tiểu đường", "tieu duong") or "tieu duong" in plain:
        return "Y"
    if (value in ("p", "pre-diabetes", "pre diabetes", "tiền tiểu đường", "tien tieu duong")
            or "tien tieu duong" in plain):
        return "P"
    if value in ("n", "normal", "bình thường", "binh thuong") or "binh thuong" in plain:
        return "N"
    raise ValueError("Unsupported final diagnosis: " + str(diagnosis))


def gender_to_dataset_value(gender):
    value = str(gender or "").strip().lower()
    if value in ("1", "f", "female", "nữ", "nu"):
        return "F"
    return "M"


def append_doctor_feedback(data):
    FEEDBACK_DATASET.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "ID", "No_Pation", "Gender", "AGE", "Urea", "Cr", "HbA1c",
        "Chol", "TG", "HDL", "LDL", "VLDL", "BMI", "CLASS"
    ]
    record_id = int(get_json_value(data, "record_id", "health_record_id"))
    patient_id = int(get_json_value(data, "patient_id", "no_pation", "No_Pation") or record_id)
    row = {
        "ID": record_id,
        "No_Pation": patient_id,
        "Gender": gender_to_dataset_value(get_json_value(data, "gender")),
        "AGE": float(get_json_value(data, "age")),
        "Urea": float(get_json_value(data, "urea")),
        "Cr": float(get_json_value(data, "cr")),
        "HbA1c": float(get_json_value(data, "hba1c")),
        "Chol": float(get_json_value(data, "chol")),
        "TG": float(get_json_value(data, "tg")),
        "HDL": float(get_json_value(data, "hdl")),
        "LDL": float(get_json_value(data, "ldl", "idl")),
        "VLDL": float(get_json_value(data, "vldl")),
        "BMI": float(get_json_value(data, "bmi")),
        "CLASS": normalize_diagnosis_to_class(get_json_value(data, "diagnosis", "final_diagnosis", "class"))
    }
    if FEEDBACK_DATASET.exists():
        existing = pd.read_csv(FEEDBACK_DATASET)
        existing = existing[existing["ID"].astype(str) != str(record_id)]
        feedback_df = pd.concat([existing, pd.DataFrame([row], columns=columns)], ignore_index=True)
    else:
        feedback_df = pd.DataFrame([row], columns=columns)
    feedback_df.to_csv(FEEDBACK_DATASET, index=False)
    return row


def retrain_in_background():
    with TRAIN_LOCK:
        TRAINING_STATUS["running"] = True
        TRAINING_STATUS["last_status"] = "running"
        TRAINING_STATUS["last_message"] = "Retraining model"
        try:
            completed = subprocess.run(
                [sys.executable, str(RETRAIN_SCRIPT)],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                timeout=600
            )
            if completed.stdout:
                print(completed.stdout, end="", flush=True)
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr, flush=True)
            if completed.returncode != 0:
                TRAINING_STATUS["last_status"] = "error"
                TRAINING_STATUS["last_message"] = completed.stderr[-1000:]
            else:
                load_model_artifacts()
                TRAINING_STATUS["last_status"] = "success"
                TRAINING_STATUS["last_message"] = (
                    completed.stdout[-1000:]
                    + f"\nTraining dashboard: {LOCAL_TRAINING_DASHBOARD_URL}"
                )
                print(f"Training dashboard: {LOCAL_TRAINING_DASHBOARD_URL}", flush=True)
        except Exception as exc:
            TRAINING_STATUS["last_status"] = "error"
            TRAINING_STATUS["last_message"] = str(exc)
        finally:
            TRAINING_STATUS["running"] = False


def start_retraining_async():
    if TRAINING_STATUS["running"]:
        return False
    thread = threading.Thread(target=retrain_in_background, daemon=True)
    thread.start()
    return True


def get_training_history_rows():
    if not TRAINING_HISTORY_FILE.exists():
        return []
    history_df = pd.read_csv(TRAINING_HISTORY_FILE)
    history_df = history_df.replace({np.nan: None})
    return history_df.to_dict(orient="records")


def get_latest_wandb_url(rows=None):
    if LATEST_WANDB_URL_FILE.exists():
        return LATEST_WANDB_URL_FILE.read_text(encoding="utf-8").strip()
    for row in reversed(rows or []):
        url = row.get("WandB_Run_URL")
        if url:
            return url
    return WANDB_HOME_URL


def print_startup_links():
    print("", flush=True)
    print("AI training links:", flush=True)
    print(f"  Dashboard: {LOCAL_TRAINING_DASHBOARD_URL}", flush=True)
    print(f"  History JSON: {LOCAL_TRAINING_HISTORY_URL}", flush=True)
    wandb_url = get_latest_wandb_url()
    print(f"  W&B: {wandb_url}", flush=True)
    print("", flush=True)

# Hàm lưu vào database
def save_to_db(record_id, diabetes_prob, pre_prob, normal_prob):

    try:
        db_parts = [
            f"DRIVER={{{os.getenv('DB_DRIVER', 'SQL Server')}}}",
            f"SERVER={os.getenv('DB_SERVER', r'DESKTOP-KPBJQCT\\SQLEXPRESS')}",
            f"DATABASE={os.getenv('DB_NAME', 'project')}",
        ]
        db_uid = os.getenv("DB_UID")
        db_pwd = os.getenv("DB_PWD")
        if db_uid and db_pwd:
            db_parts.extend([f"UID={db_uid}", f"PWD={db_pwd}"])
        else:
            db_parts.append("Trusted_Connection=yes")
        conn_str = ";".join(db_parts) + ";"
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()

        
        try:
            cursor.execute("SELECT 1 FROM [dbo].[Healthy_Record] WHERE health_record_id = ?", (int(record_id),))
            if cursor.fetchone() is None:
                msg = f"LỖI LƯU DB: health_record_id {record_id} không tồn tại trong Healthy_Record"
                with open('db_log.txt', 'a', encoding='utf-8') as f:
                    f.write(msg + "\n")
                try:
                    print(msg)
                except Exception:
                    try:
                        print(msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
                    except Exception:
                        pass
                cursor.close()
                conn.close()
                return False, msg
        except Exception:
            
            pass

        
        try:
            cursor.execute("DELETE FROM [dbo].[Doctor_AI] WHERE health_record_id = ?", (record_id,))
        except Exception:
            
            pass

        sql = """INSERT INTO [dbo].[Doctor_AI] 
                 (health_record_id, diabetes_probability, pre_diabetes_probability, normal_probability) 
                 VALUES (?, ?, ?, ?)"""

        cursor.execute(sql, (int(record_id), float(diabetes_prob), float(pre_prob), float(normal_prob)))
        conn.commit()
        cursor.close()
        conn.close()
        msg = f"Lưu DB thành công cho ID: {record_id}"
        # Ghi log đơn giản
        with open('db_log.txt', 'a', encoding='utf-8') as f:
            f.write(msg + "\n")
        try:
            print(msg)
        except Exception:
            try:
                print(msg.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
            except Exception:
                pass
        return True, msg
    except Exception as e:
        err = f"LỖI LƯU DB: {str(e)}"
        with open('db_log.txt', 'a', encoding='utf-8') as f:
            f.write(err + "\n")
        try:
            print(err)
        except Exception:
            try:
                # Fallback to avoid UnicodeEncodeError on some consoles
                print(err.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
            except Exception:
                pass
        return False, err


@app.route("/doctor-feedback", methods=["POST"])
def doctor_feedback():
    try:
        data_json = normalize_json(request.get_json())
        require_fields(data_json, [
            "record_id", "patient_id", "gender", "age", "urea", "cr",
            "hba1c", "chol", "tg", "hdl", "ldl", "vldl", "bmi", "diagnosis"
        ])
        row = append_doctor_feedback(data_json)
        retrain_started = start_retraining_async()
        return jsonify({
            "status": "success",
            "message": "Doctor diagnosis added to training dataset",
            "saved_row": row,
            "retrain_started": retrain_started,
            "training_status": TRAINING_STATUS,
            "training_status_url": "/training-status",
            "training_history_url": "/training-history",
            "training_dashboard_url": "/training-dashboard",
            "training_dashboard_full_url": LOCAL_TRAINING_DASHBOARD_URL,
            "wandb_url": get_latest_wandb_url()
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/training-status", methods=["GET"])
def training_status():
    return jsonify({
        **TRAINING_STATUS,
        "training_history_url": "/training-history",
        "training_dashboard_url": "/training-dashboard",
        "training_dashboard_full_url": LOCAL_TRAINING_DASHBOARD_URL,
        "wandb_url": get_latest_wandb_url()
    })


@app.route("/training-history", methods=["GET"])
def training_history():
    rows = get_training_history_rows()
    latest_rows = rows[-20:]
    latest_rows.reverse()
    return jsonify({
        "status": "success",
        "history_file": str(TRAINING_HISTORY_FILE),
        "count": len(rows),
        "training_dashboard_url": "/training-dashboard",
        "training_dashboard_full_url": LOCAL_TRAINING_DASHBOARD_URL,
        "wandb_url": get_latest_wandb_url(rows),
        "rows": latest_rows
    })


@app.route("/training-dashboard", methods=["GET"])
def training_dashboard():
    rows = get_training_history_rows()
    latest_rows = list(reversed(rows[-50:]))
    wandb_url = get_latest_wandb_url(rows)
    best_rows = [row for row in rows if row.get("Is_Best_Model") in (True, "True", "true", 1, "1")]
    last_best = best_rows[-1] if best_rows else (rows[-1] if rows else {})
    return render_template_string("""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Training Dashboard</title>
  <style>
    body { margin: 0; font-family: Arial, sans-serif; background: #f7f8fa; color: #1f2933; }
    main { max-width: 1160px; margin: 0 auto; padding: 28px 18px; }
    header { display: flex; justify-content: space-between; gap: 16px; align-items: center; margin-bottom: 18px; }
    h1 { font-size: 28px; margin: 0; }
    .actions { display: flex; gap: 10px; flex-wrap: wrap; }
    a.button { background: #1565c0; color: #fff; padding: 10px 14px; border-radius: 6px; text-decoration: none; font-weight: 700; }
    a.secondary { background: #374151; }
    .summary { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 18px; }
    .metric { background: #fff; border: 1px solid #d8dee9; border-radius: 8px; padding: 14px; }
    .label { color: #637083; font-size: 12px; text-transform: uppercase; margin-bottom: 8px; }
    .value { font-size: 20px; font-weight: 700; word-break: break-word; }
    table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #d8dee9; border-radius: 8px; overflow: hidden; }
    th, td { padding: 10px 12px; border-bottom: 1px solid #e5e9f0; text-align: left; font-size: 14px; }
    th { background: #eef2f7; color: #364152; }
    tr:last-child td { border-bottom: 0; }
    .best { color: #0f766e; font-weight: 700; }
    .empty { background: #fff; border: 1px solid #d8dee9; border-radius: 8px; padding: 18px; }
    @media (max-width: 820px) {
      header { align-items: flex-start; flex-direction: column; }
      .summary { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      table { display: block; overflow-x: auto; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Training Dashboard</h1>
      <div class="actions">
        {% if wandb_url %}<a class="button" href="{{ wandb_url }}" target="_blank" rel="noopener">Open W&B</a>{% endif %}
        <a class="button secondary" href="/training-history">JSON</a>
      </div>
    </header>
    <section class="summary">
      <div class="metric"><div class="label">Runs</div><div class="value">{{ rows|length }}</div></div>
      <div class="metric"><div class="label">Latest Best</div><div class="value">{{ last_best.get("Model", "-") }}</div></div>
      <div class="metric"><div class="label">CV Accuracy</div><div class="value">{{ "%.4f"|format(last_best.get("CV_Accuracy", 0) or 0) }}</div></div>
      <div class="metric"><div class="label">Feedback Rows</div><div class="value">{{ last_best.get("Feedback_Rows", "-") }}</div></div>
    </section>
    {% if latest_rows %}
    <table>
      <thead>
        <tr>
          <th>Train Time</th>
          <th>Run</th>
          <th>Model</th>
          <th>CV Accuracy</th>
          <th>Test Accuracy</th>
          <th>Rows</th>
          <th>Best</th>
        </tr>
      </thead>
      <tbody>
        {% for row in latest_rows %}
        <tr>
          <td>{{ row.get("Train_Time", "") }}</td>
          <td>{{ row.get("Run_Name", "") }}</td>
          <td>{{ row.get("Model", "") }}</td>
          <td>{{ "%.4f"|format(row.get("CV_Accuracy", 0) or 0) }}</td>
          <td>{{ "%.4f"|format(row.get("Test_Accuracy", 0) or 0) }}</td>
          <td>{{ row.get("Training_Rows", "") }}</td>
          <td class="{% if row.get("Is_Best_Model") in [true, "True", "true", 1, "1"] %}best{% endif %}">
            {{ row.get("Is_Best_Model", "") }}
          </td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% else %}
    <div class="empty">No training history yet.</div>
    {% endif %}
  </main>
</body>
</html>
    """, rows=rows, latest_rows=latest_rows, last_best=last_best, wandb_url=wandb_url)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        
        raw_json = request.get_json()
        data_json = normalize_json(raw_json)
        print("INCOMING_JSON:", data_json)
        with open('incoming_json.txt', 'a', encoding='utf-8') as f:
            f.write(str(data_json) + "\n")

        record_id = int(get_json_value(data_json, 'record_id', 'health_record_id', 'healthrecordid'))

        require_fields(data_json, [
            'gender', 'age', 'urea', 'cr', 'hba1c', 'chol', 'tg', 'hdl', 'ldl', 'vldl', 'bmi'
        ])

        # 2. Tạo dictionary dữ liệu để predict
        # Safe conversions with fallbacks
        try:
            age = float(get_json_value(data_json, 'age'))
        except Exception:
            age = 0.0
        try:
            hdl = float(get_json_value(data_json, 'hdl'))
            if hdl == 0:
                hdl = 1.0
        except Exception:
            hdl = 1.0
        try:
            bmi = float(get_json_value(data_json, 'bmi'))
        except Exception:
            bmi = 0.0

        # Normalize gender: accept 'M','F','Male','Female', numeric strings, or ints
        raw_gender = get_json_value(data_json, 'gender')
        def gender_to_int(g):
            # Training maps: M -> 0, F -> 1
            if g is None:
                return 0
            if isinstance(g, (int, float)):
                return int(g)
            s = str(g).strip().lower()
            if s in ('m', 'male'):
                return 0
            if s in ('f', 'female'):
                return 1
            # fallback: try parse int (for '0'/'1')
            try:
                return int(s)
            except Exception:
                return 0
        gender_num = gender_to_int(raw_gender)

        # Accept either 'ldl' or legacy 'idl' keys
        ldl_val = get_json_value(data_json, 'ldl', 'idl')

        def to_float_or_default(v, default=0.0):
            try:
                return float(v)
            except Exception:
                return float(default)

        data = {
            "Gender": gender_num,
            "AGE": age,
            "Urea": to_float_or_default(get_json_value(data_json, 'urea')),
            "Cr": to_float_or_default(get_json_value(data_json, 'cr')),
            "HbA1c": to_float_or_default(get_json_value(data_json, 'hba1c', 'hbA1c', 'hba1C')),
            "Chol": to_float_or_default(get_json_value(data_json, 'chol')),
            "TG": to_float_or_default(get_json_value(data_json, 'tg')),
            "HDL": hdl,
            "LDL": to_float_or_default(ldl_val),
            "VLDL": to_float_or_default(get_json_value(data_json, 'vldl')),
            "BMI": bmi,
            "Chol_HDL_Ratio": to_float_or_default(get_json_value(data_json, 'chol')) / hdl if hdl != 0 else 0,
            "LDL_HDL_Ratio": to_float_or_default(ldl_val) / hdl if hdl != 0 else 0,
            "BMI_Age": bmi * age,
            "HbA1c_Age": to_float_or_default(get_json_value(data_json, 'hba1c', 'hbA1c', 'hba1C')) * age
        }

        # 3. Chuyển thành DataFrame
        df = pd.DataFrame([data])
        df = df[feature_names]

        # 4. Dự đoán
        scaled = scaler.transform(df)
        prediction = model.predict(scaled)
        probabilities = model.predict_proba(scaled)[0]
        
        # Apply probability calibration to reduce extreme values
        probabilities = calibrate_probabilities(probabilities, temperature=2.5)

        # 5. Lưu vào Database (và trả về trạng thái lưu)
        return jsonify({
            "status": "success",
            "result": label_encoder.inverse_transform(prediction)[0],
            "probabilities": {
                "Normal": float(probabilities[0]),
                "Pre-Diabetes": float(probabilities[1]),
                "Diabetes": float(probabilities[2])
            },
            "db_save": {"ok": True, "message": "Skipped in AI service; Java application saves Doctor_AI."}
        })
    
    except Exception as e:
        # Safe print to avoid UnicodeEncodeError in some consoles
        try:
            print("LỖI XỬ LÝ:", str(e))
        except Exception:
            try:
                print(str(e).encode('utf-8', errors='replace').decode('utf-8', errors='replace'))
            except Exception:
                pass
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    if os.environ.get("WERKZEUG_RUN_MAIN") in (None, "true"):
        print_startup_links()
    app.run(host='0.0.0.0', port=5000, debug=True)
