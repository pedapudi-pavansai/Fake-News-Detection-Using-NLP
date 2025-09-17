from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PredictRequest, PredictResponse
from .model_utils import predict_text

app = FastAPI(title="Fake News Detection API")

# Allow CORS from frontend dev server (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "message": "Fake News Detection API"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")
    label, proba = predict_text(text)
    # Normalize label to uppercase REAL/FAKE
    label = label.upper()
    return {"label": label, "probability": round(proba, 4)}
