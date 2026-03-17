import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Configuration
# Gestures matching the dataset collection output
gestures = [
    "HELLO", "WATER", "HELP", "YES", "NO",
    "I", "YOU", "EAT", "DRINK", "FOOD",
    "PLEASE", "THANK_YOU", "WHERE", "HOSPITAL",
    "NEED", "WANT", "GO", "COME", "MORE", "STOP"
]
dataset_path = os.path.join('..', 'dataset')
model_save_path = 'model.h5'

sequence_length = 30
num_features = 63
num_classes = len(gestures)

# Create label dictionary
label_map = {label: num for num, label in enumerate(gestures)}

def load_dataset():
    """
    Loads numpy files from the dataset directory and creates X and Y arrays.
    """
    X = []
    y = []
    
    print(f"Loading dataset from: {os.path.abspath(dataset_path)}")
    
    # Ensure dataset directory exists
    if not os.path.exists(dataset_path):
        print(f"Dataset path {dataset_path} does not exist. Please run collect_data.py first.")
        # Return empty arrays to allow the script to compile without crashing
        return np.array([]), np.array([])

    for gesture in gestures:
        gesture_dir = os.path.join(dataset_path, gesture)
        
        if not os.path.exists(gesture_dir):
            print(f"Warning: Directory '{gesture_dir}' not found.")
            continue
            
        for file in os.listdir(gesture_dir):
            if file.endswith('.npy'):
                npy_path = os.path.join(gesture_dir, file)
                
                # Load the sequence
                sequence = np.load(npy_path)
                
                # Verify shape before adding
                if sequence.shape == (sequence_length, num_features):
                    X.append(sequence)
                    y.append(label_map[gesture])
                else:
                    print(f"Skipping {file}: Incorrect shape {sequence.shape}")
                    
    X = np.array(X)
    y = to_categorical(y, num_classes=num_classes)
    
    return X, y

def build_model():
    """
    Builds the TensorFlow LSTM architecture.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, num_features)),
        Dropout(0.2),
        
        LSTM(64, activation='relu'),
        Dropout(0.2),
        
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    # 1. Load Dataset
    print("--- Loading Dataset ---")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("No valid data sequences found. Aborting training.")
        return
        
    print(f"Total sequences loaded: {X.shape[0]}")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {y.shape}")
    
    # Split data to ensure random shuffling if needed, though validation_split handles this in fit
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # 2. Build Model
    print("\n--- Building LSTM Model ---")
    model = build_model()
    model.summary()
    
    # 3. Train Model
    print("\n--- Starting Training ---")
    history = model.fit(
        X_train, y_train, 
        epochs=50, 
        batch_size=32, 
        validation_split=0.2, # Uses 20% of training data for validation
        verbose=1
    )
    
    # 4. Save Model
    print("\n--- Saving Model ---")
    model.save(model_save_path)
    print(f"Model successfully saved to {os.path.abspath(model_save_path)}")
    
    # 5. Simple Evaluation
    print("\n--- Model Evaluation ---")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Test on a specific sample sequence (first test sequence)
    print("\n--- Simple Prediction Test ---")
    sample_sequence = X_test[0]
    sample_sequence_reshaped = sample_sequence.reshape(1, sequence_length, num_features)
    
    prediction = model.predict(sample_sequence_reshaped)
    predicted_class_index = np.argmax(prediction)
    predicted_gesture = gestures[predicted_class_index]
    
    actual_class_index = np.argmax(y_test[0])
    actual_gesture = gestures[actual_class_index]
    
    print(f"Sample prediction probabilities: {prediction[0]}")
    print(f"Predicted Gesture: {predicted_gesture}")
    print(f"Actual Gesture: {actual_gesture}")
    
    if predicted_gesture == actual_gesture:
        print("✅ Prediction matches actual label.")
    else:
        print("❌ Prediction does not match actual label.")

if __name__ == "__main__":
    main()
