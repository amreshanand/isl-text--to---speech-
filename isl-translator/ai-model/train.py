import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Configuration
GESTURES = [
    "HELLO", "WATER", "HELP", "YES", "NO",
    "I", "YOU", "EAT", "DRINK", "FOOD",
    "PLEASE", "THANK_YOU", "WHERE", "HOSPITAL",
    "NEED", "WANT", "GO", "COME", "MORE", "STOP",
]

DATASET_PATH = os.path.join("..", "dataset")
MODEL_SAVE_PATH = os.path.join("saved_model", "isl_model.h5")
NUM_FEATURES = 63
NUM_CLASSES = len(GESTURES)


def load_data():
    """Load landmark arrays from dataset/ for training a Dense network (single frame)."""
    X, Y = [], []

    if not os.path.exists(DATASET_PATH):
        print(f"Dataset path '{DATASET_PATH}' not found.")
        print("Run collect_data.py first, or provide .npy files.")
        return np.array([]), np.array([])

    for index, gesture in enumerate(GESTURES):
        gesture_dir = os.path.join(DATASET_PATH, gesture)
        if not os.path.exists(gesture_dir):
            print(f"Warning: '{gesture_dir}' not found, skipping.")
            continue

        for file in os.listdir(gesture_dir):
            if not file.endswith(".npy"):
                continue
            sequence = np.load(os.path.join(gesture_dir, file))
            # Use each frame independently for the Dense model
            for frame in sequence:
                if frame.shape == (NUM_FEATURES,):
                    X.append(frame)
                    Y.append(index)

    X = np.array(X)
    Y = tf.keras.utils.to_categorical(Y, num_classes=NUM_CLASSES)
    return X, Y


def build_model():
    """Dense neural network for single-frame gesture classification."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(NUM_FEATURES,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy"],
    )
    return model


if __name__ == "__main__":
    print("=== Dense Model Training Pipeline ===\n")

    X, Y = load_data()
    if len(X) == 0:
        print("No data found. Aborting.")
        exit(1)

    print(f"Loaded {len(X)} samples across {NUM_CLASSES} classes.")

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

    model = build_model()
    model.summary()

    print("\nTraining...")
    model.fit(X_train, Y_train, epochs=100, batch_size=32, validation_data=(X_test, Y_test))

    os.makedirs("saved_model", exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")
