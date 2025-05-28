# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .model import load_model, predict

app = FastAPI()
model = load_model("app/model.pt")

class InputData(BaseModel):
    data: list  # flattened 28x28 image

@app.post("/predict")
async def predict_digit(input_data: InputData):
    try:
        prediction = predict(model, input_data.data)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
