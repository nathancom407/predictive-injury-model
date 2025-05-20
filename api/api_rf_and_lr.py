from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import uvicorn
import joblib
import traceback

rf_model = joblib.load("rf_final_model.pkl")
lr_model = joblib.load("logreg_final_model.pkl")

app = FastAPI(title="Injury Risk API v2", description="Supports Logistic Regression and Random Forest")

class ModelInput(BaseModel):
    pitch_entropy: float
    IP_per_G: float
    Pitches_per_IP: float
    FBv: float
    avg_pitch_velocity: float
    IP: float
    model_choice: int #0 for logistic regression, 1 for random forest

@app.post("/predict_v2")
def predict_risk(input_data: ModelInput):
    try:
        data_dict = input_data.dict()
        model_choice = data_dict.pop("model_choice")
        input_df = pd.DataFrame([data_dict])

        model = rf_model if model_choice == 1 else lr_model
        prob = model.predict_proba(input_df)[0][1]
        risk_class = int(prob >= 0.55)

        return {
            "Model": "Random Forest" if model_choice == 1 else "Logistic Regression",
            "Injury Risk Probability": round(prob, 3),
            "Predicted Class": risk_class,
            "Threshold": 0.55
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}
    
if __name__ == "__main__":
    uvicorn.run("api_rf_and_lr:app", host="0.0.0.0", port=8084, reload=True)