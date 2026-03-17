import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands_module
from mediapipe.python.solutions import drawing_utils as mp_drawing_module

# Configuration
GESTURES = [
    "HELLO", "WATER", "HELP", "YES", "NO",
    "I", "YOU", "EAT", "DRINK", "FOOD",
    "PLEASE", "THANK_YOU", "WHERE", "HOSPITAL",
    "NEED", "WANT", "GO", "COME", "MORE", "STOP",
]

NUM_SEQUENCES = 20       # recordings per gesture
SEQUENCE_LENGTH = 30     # frames per recording
DATASET_PATH = os.path.join("..", "dataset")


def extract_landmarks(results) -> np.ndarray:
    """Extract 63 values (21 landmarks × 3 coords) from the first detected hand."""
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        return np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
    return np.zeros(63)


def run_collection():
    # Create dataset folders
    for gesture in GESTURES:
        os.makedirs(os.path.join(DATASET_PATH, gesture), exist_ok=True)

    cap = cv2.VideoCapture(0)

    with mp_hands_module.Hands(
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=1,
    ) as hands:

        print("Data collection started.")
        print("  Press 's' to START recording each sequence.")
        print("  Press 'q' at any time to QUIT.\n")

        for gesture in GESTURES:
            for seq_num in range(NUM_SEQUENCES):

                # Wait for user to press 's'
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame = cv2.flip(frame, 1)
                    cv2.putText(
                        frame,
                        f"Ready: {gesture} | Press 's' for Seq {seq_num + 1}/{NUM_SEQUENCES}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
                    )
                    cv2.imshow("Data Collection", frame)

                    key = cv2.waitKey(10) & 0xFF
                    if key == ord("s"):
                        break
                    elif key == ord("q"):
                        print("Quit.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                # Record sequence
                sequence_data = []
                for frame_num in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb)

                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing_module.draw_landmarks(
                                frame, hand_landmarks, mp_hands_module.HAND_CONNECTIONS,
                            )

                    sequence_data.append(extract_landmarks(results))

                    cv2.putText(
                        frame,
                        f"Recording {gesture} | Seq {seq_num + 1}/{NUM_SEQUENCES} | Frame {frame_num + 1}/{SEQUENCE_LENGTH}",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                    )
                    cv2.imshow("Data Collection", frame)

                    if cv2.waitKey(10) & 0xFF == ord("q"):
                        print("Quit.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                # Save the 30-frame sequence as .npy
                npy_path = os.path.join(DATASET_PATH, gesture, f"seq_{seq_num + 1}.npy")
                np.save(npy_path, np.array(sequence_data))
                print(f"Saved: {npy_path}")

        print("\n✅ Dataset collection complete!")
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_collection()
