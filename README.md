# MLB Pitcher Injury Risk Prediction API

This project provides a FastAPI-based machine learning inference API to predict severe arm-injury (Tommy John Surgery) risk for MLB pitchers using different classification models (XGBoost, Random Forest, and Logistic Regression). It is designed to support model comparison, experimentation, and supervised learning.

## Features

- Accepts JSON input via HTTP POST requests
- Supports multiple models:
  - `XGBoost`
  - `Random Forest` 
  - `Logistic Regression`
- Dynamically plots feature effect on predicted injury probability
- Designed with modular notebooks and API files

## Model Selection Guide

This API predicts **severe pitcher injuries**, specifically those requiring **Tommy John surgery**. Based on your priorities (e.g. catching all possible injuries vs. avoiding false alarms), different models offer different advantages:

| Model                     | Use Case Summary |
|--------------------------|------------------|
| **Logistic Regression (Raw)** | Captures **70% of actual injuries** (highest injured recall). Ideal for **conservative use cases** where missing an injury is costly. Tends to produce **more false positives**, so healthy pitchers may also be flagged. Use this model if **you want to catch as many injury cases as possible**, even if it includes false alarms. |
| **Random Forest (Tuned)**     | Captures **81% of healthy cases** with fewer false positives, but only identifies **40% of injuries**. Best suited when the cost of false positives is high. Use this model if **you prefer precision and are okay with missing some injury cases**. |
| **XGBoost (Post-Lasso)**      | Offers the **best overall performance** (highest average F1 score: 0.60). Strikes a **balance between detecting injuries and avoiding false positives**. Recommended as the **default model** for most use cases. |

### Default Recommendation:
Use **XGBoost (Post-Lasso)** for a balanced approach.  
Use **Logistic Regression** for conservative, injury-averse strategies.  
Use **Random Forest** to reduce false positives in high-stakes environments.

> ⚠️ Note: This model is trained to predict **severe injuries** that typically result in **Tommy John surgery**, not minor or short-term issues.
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
