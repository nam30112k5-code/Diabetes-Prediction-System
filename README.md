# Diabetes Prediction System

AI-powered diabetes risk prediction system using Flask and machine learning.

This project predicts a patient's diabetes risk from health indicators and can store prediction results in SQL Server. It also supports doctor feedback data for retraining the model.

## Features

- Predicts diabetes risk through a Flask API
- Uses trained machine learning artifacts from the `Models/` folder
- Supports 3 prediction classes: normal, pre-diabetes, and diabetes
- Saves prediction probability results to SQL Server
- Collects doctor feedback through an API endpoint
- Supports automatic retraining with updated feedback data

## Project Structure

```text
.
├── app.py
├── model-trainer.py
├── retrain_auto.py
├── requirements.txt
├── Models/
├── dataset/
└── Results/
```

## API Endpoints

```text
POST /predict
POST /doctor-feedback
GET  /training-status
```

## Setup

Install dependencies:

```powershell
py -m pip install -r requirements.txt
```

Run the Flask app:

```powershell
py app.py
```

## SQL Server Configuration

The app reads SQL Server credentials from environment variables.

```powershell
$env:DB_UID="sa"
$env:DB_PWD="your_password"
py app.py
```

Optional variables:

```powershell
$env:DB_DRIVER="SQL Server"
$env:DB_SERVER="DESKTOP-KPBJQCT\SQLEXPRESS"
$env:DB_NAME="project"
```

If `DB_UID` and `DB_PWD` are not set, the app uses Windows Authentication.

## Model Training

Run:

```powershell
py model-trainer.py
```

Automatic retraining:

```powershell
py retrain_auto.py
```

## Description

This is an AI diabetes prediction project built to support early risk screening. It is not a replacement for professional medical diagnosis.
