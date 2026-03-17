import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Phase 3: AI Model Training Script
# Contains Data Collection, Landmark Extraction, and Keras Deep Learning Model

# Paths and configuration
DATASET_PATH = 'dataset/videos'
DATA_ARRAYS_PATH = 'dataset/arrays'
MODEL_SAVE_PATH = 'saved_model/isl_model.h5'

# Gestures matching the backend mapping
gestures = np.array(['HELLO', 'WATER', 'HELP', 'YES', 'NO'])

# Initialization of MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def process_image(image, hands_model):
    """
    Given an image/frame, process it to extract 21 x 3 (63 values) coordinates.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_model.process(image_rgb)
    landmarks = []
    
    if results.multi_hand_landmarks:
        # For simplicity, extract the first hand found
        hand_landmarks = results.multi_hand_landmarks[0]
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        # Return 63 zeros if no hand detected
        landmarks = [0.0] * 63
        
    return np.array(landmarks)

def generate_synthetic_dataset():
    """
    In a real project, read from videos. Since we don't have video files,
    this function synthesizes fake landmark data for training demonstration.
    Replace this with real video capturing logic.
    """
    if not os.path.exists(DATA_ARRAYS_PATH):
        os.makedirs(DATA_ARRAYS_PATH)
        
    for index, action in enumerate(gestures):
        # Create folder for each action
        action_path = os.path.join(DATA_ARRAYS_PATH, action)
        os.makedirs(action_path, exist_ok=True)
        
        # Suppose we record 30 videos per gesture, 30 frames each
        for sequence in range(30):
            # For each frame, generate a faux array (63 items)
            # You would replace the random logic with `process_image` over frames
            fake_landmarks = np.random.rand(63) 
            # In a real pipeline: np.save(os.path.join(action_path, f"{sequence}.npy"), actual_landmarks)
            
def load_data():
    """
    Load data from the directory structure for X (landmarks) and Y (labels).
    """
    # Assuming synthetic data generation was run:
    X, Y = [], []
    # Mocking loading for the snippet:
    for index, action in enumerate(gestures):
        for seq in range(500): # Mocking 500 samples per class
            # Add some slight variation around the 'class' index to make model learnable
            base = np.random.rand(63) * 0.1
            base[index % 63] += 0.8  # Make a specific node activate distinctly for each class
            X.append(base)
            Y.append(index)
            
    # Convert to categorical labels and numpy arrays
    X = np.array(X)
    Y = tf.keras.utils.to_categorical(Y, num_classes=len(gestures))
    return X, Y

def build_model():
    """
    Phase 3: Deep Learning Model Architecture
    Input layer expects 63 values (1D Array per frame mapping)
    """
    model = tf.keras.models.Sequential([
        # Dense network approach for basic hand orientation
        tf.keras.layers.Dense(128, activation='relu', input_shape=(63,)),
        tf.keras.layers.Dropout(0.2), # Prevent overfitting
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        
        # Softmax outputs for multi-class prediction
        tf.keras.layers.Dense(len(gestures), activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['categorical_accuracy'])
    return model

if __name__ == '__main__':
    print("Starting Model Training Pipeline...")
    
    # 1. Dataset collection or generation (Replace with video capture script)
    # generate_synthetic_dataset()
    
    # 2. Extract Data
    X, Y = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
    
    # 3. Build & Compile Model
    model = build_model()
    model.summary()
    
    # 4. Train Model
    print("Training model ...")
    model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test))
    
    # 5. Save Model
    os.makedirs('saved_model', exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved successfully to {MODEL_SAVE_PATH}")
    print("Training complete! Model is ready for inference in the FastAPI layer.")
