
from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import uvicorn
import joblib
import traceback


app = FastAPI(title="Injury Risk Prediction API", description="Predict injury risk for MLB pitchers using chosen model.")

# input schema
class PitcherInput(BaseModel):
    pitch_entropy: float
    IP_per_G: float
    FBv: float
    IP: float

#loading xgboost model
model = joblib.load("xgboost_final_post_lasso.pkl") 

# inference route
@app.post("/predict")
def predict_injury_risk(data: PitcherInput):
    print("Request received")
    try:
        input_df = pd.DataFrame([data.model_dump()])
        prob = model.predict_proba(input_df)[0][1]
        risk_class = int(prob >= 0.55)

        return {
            "Injury Risk Probability": float(prob),
            "Predicted Class": int(risk_class),
            "Threshold": 0.55
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run("xgiboost_api:app", host="0.0.0.0", port=8084, reload=True)
