import cv2
import numpy as np
import os
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Configuration
gestures = ["HELLO", "WATER", "HELP", "YES", "NO"]
num_sequences = 20
sequence_length = 30
dataset_path = os.path.join('..', 'dataset')

# Make sure dataset folders exist
for gesture in gestures:
    os.makedirs(os.path.join(dataset_path, gesture), exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(results):
    """
    Extracts the (x, y, z) landmarks of the first detected hand.
    Returns: a flattened numpy array of size 63 (21 landmarks * 3 coordinates).
    If no hand is detected, returns an array of 63 zeros.
    """
    if results.multi_hand_landmarks:
        # Only extract the first hand to match the frontend behavior
        hand = results.multi_hand_landmarks[0]
        landmarks = np.array([[res.x, res.y, res.z] for res in hand.landmark]).flatten()
        return landmarks
    return np.zeros(63)

def run_collection():
    cap = cv2.VideoCapture(0)
    
    # Set up mediapipe
    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        max_num_hands=1
    ) as hands:

        print("Data collection started. Follow the on-screen instructions.")
        print("Press 's' to START recording each sequence.")
        print("Press 'q' at any time to QUIT.")
        
        for gesture in gestures:
            for sequence in range(num_sequences):
                
                # ── Wait Screen ──
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                        
                    # Mirror the frame
                    frame = cv2.flip(frame, 1)
                    
                    # Instruction text
                    cv2.putText(
                        frame, 
                        f"Wait for {gesture}. Press 's' to start Seq {sequence+1}/{num_sequences}", 
                        (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 0, 255),  # Red text
                        2
                    )
                    
                    cv2.imshow('Data Collection', frame)
                    
                    # Keyboard check
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('s'):
                        break
                    elif key == ord('q'):
                        print("Quitting data collection.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                
                # ── Recording Screen ──
                sequence_data = []
                
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        continue
                        
                    frame = cv2.flip(frame, 1)
                    
                    # Convert BGR to RGB for MediaPipe processing
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(rgb_frame)
                    
                    # Draw visual landmarks
                    if results.multi_hand_landmarks:
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            
                    # Extract 63-value array
                    landmarks = extract_landmarks(results)
                    sequence_data.append(landmarks)
                    
                    # Show recording progress
                    cv2.putText(
                        frame, 
                        f"Recording {gesture} : sequence {sequence+1}/{num_sequences} | Frame {frame_num+1}/{sequence_length}", 
                        (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0, 255, 0),  # Green text
                        2
                    )
                    
                    cv2.imshow('Data Collection', frame)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        print("Quitting data collection.")
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                        
                # ── Save the 30-frame sequence ──
                # Shape is (30, 63)
                npy_path = os.path.join(dataset_path, gesture, f"seq_{sequence+1}.npy")
                np.save(npy_path, np.array(sequence_data))
                print(f"Saved: {npy_path}")

        print("✅ Dataset collection complete!")
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_collection()
