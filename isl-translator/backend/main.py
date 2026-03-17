from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import os

# Gesture labels (must match frontend & training scripts)
GESTURE_LABELS = [
    "HELLO", "WATER", "HELP", "YES", "NO",
    "I", "YOU", "EAT", "DRINK", "FOOD",
    "PLEASE", "THANK_YOU", "WHERE", "HOSPITAL",
    "NEED", "WANT", "GO", "COME", "MORE", "STOP",
]

MODEL_PATH = os.getenv("MODEL_PATH", "../ai-model/saved_model/isl_model.h5")

# Global model reference
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model on startup, cleanup on shutdown."""
    global model
    try:
        import tensorflow as tf
        if os.path.exists(MODEL_PATH):
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print(f"No model found at {MODEL_PATH} — running with heuristic predictions.")
    except ImportError:
        print("TensorFlow not available — running with heuristic predictions.")
    except Exception as e:
        print(f"Warning: Could not load model. {e}")

    yield  # App runs here

    model = None


app = FastAPI(
    title="ISL Translator API",
    version="1.0",
    lifespan=lifespan,
)

# CORS — allow frontend origin
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ──────────────────────────────────────────

class SequenceData(BaseModel):
    sequence: list[list[float]]  # 30 frames × 63 values each


class User(BaseModel):
    name: str
    email: str
    role: str


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"message": "ISL Translator API is running"}


@app.post("/predict")
async def predict_gesture(data: SequenceData):
    """Predict gesture from a 30-frame sequence of hand landmarks."""
    if len(data.sequence) != 30:
        raise HTTPException(status_code=400, detail="Expected a sequence of 30 frames")

    for i, frame in enumerate(data.sequence):
        if len(frame) != 63:
            raise HTTPException(status_code=400, detail=f"Frame {i} does not have 63 values")

    try:
        input_data = np.array(data.sequence).reshape(1, 30, 63)

        if model is not None:
            prediction = model.predict(input_data)
            gesture = GESTURE_LABELS[np.argmax(prediction)]
            confidence = float(np.max(prediction))
        else:
            gesture = _heuristic_predict(data.sequence[-1])
            confidence = 0.95

        return {"gesture": gesture, "confidence": confidence}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _heuristic_predict(frame: list[float]) -> str:
    """Fallback prediction using finger-up heuristics when no ML model is loaded."""
    def get_y(index: int) -> float:
        return frame[index * 3 + 1]

    # Finger states: tip.y < pip.y means finger is up
    thumb_up = get_y(4) < get_y(3)
    idx_up = get_y(8) < get_y(6)
    middle_up = get_y(12) < get_y(10)
    ring_up = get_y(16) < get_y(14)
    pinky_up = get_y(20) < get_y(18)

    if idx_up and middle_up and ring_up and pinky_up:
        return "HELLO"
    if thumb_up and idx_up and not middle_up and not ring_up and pinky_up:
        return "HELP"
    if idx_up and middle_up and not ring_up and not pinky_up:
        return "WATER"
    if idx_up and not middle_up and not ring_up and not pinky_up:
        return "YES"
    if not thumb_up and not idx_up and not middle_up and not ring_up and not pinky_up:
        return "NO"

    return "HELLO"


@app.post("/users")
async def create_user(user: User):
    # TODO: Connect to PostgreSQL via SQLAlchemy or asyncpg
    return {"status": "success", "message": f"User {user.name} created", "user": user}


@app.post("/logs")
async def log_gesture(gesture: str, timestamp: str):
    # TODO: Log usage statistics to DB
    return {"status": "logged"}


@app.get("/stats")
async def get_stats():
    # TODO: Return real platform statistics from DB
    return {"total_users": 0, "gestures_translated": 0}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
