# MLB Pitcher Injury Risk Prediction API

This project provides a FastAPI-based machine learning inference API to predict injury risk for MLB pitchers using different classification models (XGBoost, Random Forest, and Logistic Regression). It is designed to support model comparison and experimentation.

## Features

- Accepts JSON input via HTTP POST requests
- Supports multiple models:
  - `XGBoost`
  - `Random Forest` 
  - `Logistic Regression`
- Dynamically plots feature effect on predicted injury probability
- Designed with modular notebooks and API files

---

## Project Structure

```
.
├── api/                   # FastAPI scripts (e.g., api_rf_and_lr.py for random forest and logistic regression)
├── models/                # Saved ML models (.pkl files)
├── notebooks/             # Jupyter notebooks for model training & results
├── data/                  # Datasets and sample input file
├── requirements.txt       # Python dependencies
└── .gitignore             # Ignored files and folders
```

---

## Setup Instructions

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/mlb-injury-api.git
cd mlb-injury-api
```

2. **Create a virtual environment**:

```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Run the API** (example for RF + LR API):

```bash
uvicorn api.api_rf_and_lr:app --host 0.0.0.0 --port 8084 --reload
```

---

## Sample Request (to `predict_v2` endpoint)

**Request**:

```bash
POST http://127.0.0.1:8084/predict_v2
```

**JSON Body**:
```json
{
  "pitch_entropy": 2.1,
  "IP_per_G": 6.5,
  "Pitches_per_IP": 15.5,
  "FBv": 94.2,
  "avg_pitch_velocity": 85.1,
  "IP": 110.0,
  "model_choice": 1
}
```

- `model_choice = 1` → Random Forest  
- `model_choice = 0` → Logistic Regression

---

## Visualizations

Use the provided notebooks to:

- Explore how features like pitch entropy, FBv, or innings pitched affect injury risk.
- Compare different model predictions.
- Generate plots for interpretability.

---

## To-Do / Future Improvements

- Add support for additional models (e.g., neural networks)
- Containerize the API using Docker
- Deploy on a cloud platform (e.g., Render, AWS, Azure)

---

## Contact

Built by Nathan Comstock (https://github.com/nathancom407)  
Email: nathanc.personal@gmail.com
