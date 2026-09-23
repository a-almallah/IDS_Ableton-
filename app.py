from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import pandas as pd
import joblib
import warnings
from contextlib import asynccontextmanager
import uvicorn

# Suppress sklearn warnings about feature names if they appear
warnings.filterwarnings('ignore', category=UserWarning)

model = None
scaler = None
model_features = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, model_features
    print("Loading models and scaler...")
    try:
        model = joblib.load('model.pkl')
        scaler = joblib.load('scaler.pkl')
        model_features = joblib.load('model_features.pkl')
        print(f"Successfully loaded model expecting {len(model_features)} features.")
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        print("Please ensure you have run train.py first to generate the models.")
    yield
    print("Shutting down IDS API...")

app = FastAPI(title="Live IDS Inference API", lifespan=lifespan)

@app.post("/predict")
async def predict(request: Request):
    """
    Endpoint for receiving live network flow data.
    Takes a dynamic JSON payload to handle CicFlowMeter output directly.
    """
    global model, scaler, model_features
    
    if model is None or scaler is None or model_features is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Ensure train.py has been run.")

    try:
        # Accept raw JSON from CicFlowMeter
        payload = await request.json()
        
        # If the payload is a single dictionary, wrap it in a list
        if isinstance(payload, dict):
            payload = [payload]
            
        df = pd.DataFrame(payload)

        # Ensure all required features are present
        missing_cols = set(model_features) - set(df.columns)
        if missing_cols:
            # If there are missing columns, fill them with 0 to prevent crashes
            for col in missing_cols:
                df[col] = 0.0
        
        # Reorder and filter columns to match training EXACTLY
        X_live = df[model_features]

        # Handle any possible NaNs or Infs that CICFlowMeter might have sent
        X_live = X_live.replace([float('inf'), float('-inf')], 0.0).fillna(0.0)
        
        # Scale the data
        X_live_scaled = scaler.transform(X_live)
        
        # Run prediction
        predictions = model.predict(X_live_scaled)
        
        results = []
        for pred in predictions:
            # Assume any label other than "BENIGN" or "Normal" is an intrusion
            is_anomaly = str(pred).upper() not in ["BENIGN", "NORMAL"]
            
            if is_anomaly:
                print(f"\n\033[91m🚨 INTRUSION DETECTED: [{pred}]\033[0m\n")
            else:
                print(f"\n\033[92m✅ Normal Traffic\033[0m\n")
                
            results.append({"prediction": str(pred), "is_anomaly": is_anomaly})
            
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
