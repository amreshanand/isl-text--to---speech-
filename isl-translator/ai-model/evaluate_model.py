"""
Model Evaluation & Testing Script
===================================
Test the trained model with live predictions or batch evaluation.

Usage:
    python evaluate_model.py                    # Evaluate on test set
    python evaluate_model.py --live             # Live webcam prediction
    python evaluate_model.py --model model.h5   # Use a specific model file
"""

import os
import argparse
import numpy as np
import tensorflow as tf

GESTURES = [
    "HELLO", "WATER", "HELP", "YES", "NO",
    "I", "YOU", "EAT", "DRINK", "FOOD",
    "PLEASE", "THANK_YOU", "WHERE", "HOSPITAL",
    "NEED", "WANT", "GO", "COME", "MORE", "STOP",
]

DATASET_PATH = os.path.join("..", "dataset")
DEFAULT_MODEL = os.path.join("saved_model", "isl_model.h5")
SEQUENCE_LENGTH = 30
NUM_FEATURES = 63
NUM_CLASSES = len(GESTURES)


def load_test_data():
    """Load a subset of the dataset for evaluation."""
    X, y = [], []
    label_map = {label: idx for idx, label in enumerate(GESTURES)}

    for gesture in GESTURES:
        gesture_dir = os.path.join(DATASET_PATH, gesture)
        if not os.path.exists(gesture_dir):
            continue

        files = sorted([f for f in os.listdir(gesture_dir) if f.endswith(".npy")])
        # Use last 20% as test set
        test_files = files[int(len(files) * 0.8):]

        for file in test_files:
            seq = np.load(os.path.join(gesture_dir, file))
            if seq.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                X.append(seq)
                y.append(label_map[gesture])

    return np.array(X), np.array(y)


def batch_evaluate(model, X_test, y_test):
    """Run batch evaluation on the test set."""
    y_cat = tf.keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

    loss, accuracy = model.evaluate(X_test, y_cat, verbose=0)
    print(f"\n  Test Loss:     {loss:.4f}")
    print(f"  Test Accuracy: {accuracy*100:.1f}%\n")

    # Per-class breakdown
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("  Gesture          Correct / Total   Accuracy")
    print("  " + "─" * 48)

    for idx, gesture in enumerate(GESTURES):
        mask = y_test == idx
        if mask.sum() == 0:
            continue
        correct = (y_pred_classes[mask] == idx).sum()
        total = mask.sum()
        acc = correct / total
        bar = "█" * int(acc * 15)
        print(f"  {gesture:14s}   {correct:3d} / {total:3d}       {acc*100:5.1f}%  {bar}")

    return accuracy


def predict_single(model, sequence):
    """Predict a single sequence and show top-3 results."""
    input_data = sequence.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
    prediction = model.predict(input_data, verbose=0)[0]

    # Top 3
    top_indices = np.argsort(prediction)[::-1][:3]

    print("\n  Top Predictions:")
    for rank, idx in enumerate(top_indices, 1):
        confidence = prediction[idx]
        bar = "█" * int(confidence * 30)
        print(f"    #{rank}  {GESTURES[idx]:12s}  {confidence*100:5.1f}%  {bar}")

    return GESTURES[top_indices[0]], prediction[top_indices[0]]


def live_webcam_predict(model):
    """Run live prediction using the webcam and MediaPipe."""
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        print("❌ opencv-python and mediapipe are required for live mode.")
        print("   pip install opencv-python mediapipe")
        return

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    sequence_buffer = []
    current_gesture = "..."
    confidence = 0.0

    print("\nLive prediction started. Press 'q' to quit.\n")

    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=1,
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                # Extract landmarks
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
                sequence_buffer.append(landmarks)

                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    seq = np.array(sequence_buffer)
                    input_data = seq.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
                    prediction = model.predict(input_data, verbose=0)[0]

                    pred_idx = np.argmax(prediction)
                    current_gesture = GESTURES[pred_idx]
                    confidence = prediction[pred_idx]
                    sequence_buffer = []

                    print(f"  Predicted: {current_gesture} ({confidence*100:.1f}%)")

            # Display
            cv2.putText(frame, f"Gesture: {current_gesture}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence*100:.1f}%", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
            cv2.putText(frame, f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("ISL Live Prediction", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\nLive prediction stopped.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ISL gesture model")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="Path to the trained model file")
    parser.add_argument("--live", action="store_true",
                        help="Run live webcam prediction")
    args = parser.parse_args()

    # Load model
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("   Train a model first: python train_lstm.py")
        return

    print(f"Loading model: {args.model}")
    model = tf.keras.models.load_model(args.model)

    if args.live:
        live_webcam_predict(model)
    else:
        print("\n── Loading Test Data ──")
        X_test, y_test = load_test_data()

        if len(X_test) == 0:
            print("❌ No test data found.")
            return

        print(f"  Test samples: {len(X_test)}")
        batch_evaluate(model, X_test, y_test)

        # Show a few sample predictions
        print("\n── Sample Predictions ──")
        indices = np.random.choice(len(X_test), size=min(5, len(X_test)), replace=False)
        for idx in indices:
            actual = GESTURES[y_test[idx]]
            predicted, conf = predict_single(model, X_test[idx])
            match = "✅" if predicted == actual else "❌"
            print(f"    {match} Actual: {actual:12s} | Predicted: {predicted:12s} ({conf*100:.1f}%)")


if __name__ == "__main__":
    main()
