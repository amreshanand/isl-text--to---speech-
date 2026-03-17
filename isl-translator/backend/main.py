from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

# Phase 4: Backend API creation
app = FastAPI(title="ISL Translator API", version="1.0")

# Allow requests from frontend (Next.js runs on 3000 by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*"], # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Placeholder for the loaded TensorFlow/Keras model
model = None
gesture_labels = [
    "HELLO", "WATER", "HELP", "YES", "NO",
    "I", "YOU", "EAT", "DRINK", "FOOD",
    "PLEASE", "THANK_YOU", "WHERE", "HOSPITAL",
    "NEED", "WANT", "GO", "COME", "MORE", "STOP"
]

# Load the model on startup
@app.on_event("startup")
async def load_model():
    global model
    try:
        # In a real scenario, uncomment the following line
        # import tensorflow as tf
        # model = tf.keras.models.load_model('../ai-model/saved_model/isl_model.h5')
        print("Model loaded successfully (Placeholder).")
    except Exception as e:
        print(f"Warning: Could not load model. {e}")

# Data structure expected from frontend
class SequenceData(BaseModel):
    sequence: list[list[float]]  # Expecting 30 frames, each with 63 values

@app.get("/")
def read_root():
    return {"message": "ISL Translator API is running"}

@app.post("/predict")
async def predict_gesture(data: SequenceData):
    """
    Phase 4: Predict gesture from sequence of landmarks
    Receives 30 frames, each with 63 landmark coordinates from frontend.
    """
    if len(data.sequence) != 30:
        raise HTTPException(status_code=400, detail="Expected a sequence of 30 frames")
    
    for i, frame in enumerate(data.sequence):
        if len(frame) != 63:
            raise HTTPException(status_code=400, detail=f"Frame {i} does not have 63 values")

    try:
        # Convert to numpy array and reshape for model input (Batch, Timesteps, Features)
        input_data = np.array(data.sequence).reshape(1, 30, 63)

        if model is not None:
            # Re-enable in production:
            # prediction = model.predict(input_data)
            # gesture = gesture_labels[np.argmax(prediction)]
            gesture = "HELLO"
        else:
            # MOCK PREDICTION for sequence
            # Use the last frame of the sequence for heuristic detection
            last_frame = data.sequence[-1]
            
            def get_y(index):
                return last_frame[index * 3 + 1] # y is the second coordinate
                
            # Finger states (tip.y < pip.y means finger is up)
            idx_up = get_y(8) < get_y(6)
            middle_up = get_y(12) < get_y(10)
            ring_up = get_y(16) < get_y(14)
            pinky_up = get_y(20) < get_y(18)
            thumb_up = get_y(4) < get_y(3) # approximate for thumb
            
            if idx_up and middle_up and ring_up and pinky_up:
                gesture = "HELLO"
            elif thumb_up and idx_up and not middle_up and not ring_up and pinky_up:
                gesture = "HELP"
            elif idx_up and middle_up and not ring_up and not pinky_up:
                gesture = "WATER"
            elif idx_up and not middle_up and not ring_up and not pinky_up:
                gesture = "YES"
            elif not thumb_up and not idx_up and not middle_up and not ring_up and not pinky_up:
                gesture = "NO"
            else:
                # If it doesn't match perfectly, fallback to heuristics
                if not idx_up and not middle_up and not ring_up:
                    gesture = "NO"
                elif idx_up and middle_up and not ring_up:
                    gesture = "WATER"
                elif idx_up and not middle_up and not pinky_up:
                    gesture = "YES"
                else:
                    gesture = "HELLO (SEQ)"
            
        return {"gesture": gesture, "confidence": 0.95}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Phase 8: Database Endpoints Structure
class User(BaseModel):
    name: str
    email: str
    role: str

@app.post("/users")
async def create_user(user: User):
    # In production, connect to PostgreSQL via SQLAlchemy or asyncpg
    return {"status": "success", "message": f"User {user.name} created", "user": user}

@app.post("/logs")
async def log_gesture(gesture: str, timestamp: str):
    # Log usage statistics to DB
    return {"status": "logged"}

@app.get("/stats")
async def get_stats():
    # Return platform statistics
    return {"total_users": 100, "gestures_translated": 5000}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
