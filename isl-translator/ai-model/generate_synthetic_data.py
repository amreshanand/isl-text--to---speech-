"""
Synthetic Dataset Generator for ISL Gesture Training
=====================================================
Generates realistic hand landmark sequences for all 20 gestures.

Each gesture is defined by which fingers are UP/DOWN, and the generator
creates anatomically-plausible 21-landmark hand poses with natural
jitter, rotation, and wrist position variation.

Usage:
    python generate_synthetic_data.py
    python generate_synthetic_data.py --sequences 50 --augmentation 3
"""

import os
import argparse
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────

GESTURES = [
    "HELLO", "WATER", "HELP", "YES", "NO",
    "I", "YOU", "EAT", "DRINK", "FOOD",
    "PLEASE", "THANK_YOU", "WHERE", "HOSPITAL",
    "NEED", "WANT", "GO", "COME", "MORE", "STOP",
]

DATASET_PATH = os.path.join("..", "dataset")
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3  # 63

# MediaPipe hand landmark indices
# 0: WRIST
# 1-4: THUMB (CMC, MCP, IP, TIP)
# 5-8: INDEX (MCP, PIP, DIP, TIP)
# 9-12: MIDDLE (MCP, PIP, DIP, TIP)
# 13-16: RING (MCP, PIP, DIP, TIP)
# 17-20: PINKY (MCP, PIP, DIP, TIP)

# ── Gesture Definitions ───────────────────────────────────────────────
# Each gesture defines which fingers are UP (True) or DOWN (False)
# Order: [thumb, index, middle, ring, pinky]

GESTURE_DEFINITIONS = {
    "HELLO":     [True,  True,  True,  True,  True],    # Open palm
    "WATER":     [False, True,  True,  False, False],   # Peace sign
    "HELP":      [True,  True,  False, False, True],    # ILY sign
    "YES":       [False, True,  False, False, False],   # Pointing up
    "NO":        [False, False, False, False, False],   # Closed fist
    "I":         [False, False, False, False, True],    # Pinky up
    "YOU":       [False, True,  False, False, False],   # Pointing forward
    "EAT":       [True,  True,  True,  False, False],   # Three fingers
    "DRINK":     [True,  False, False, False, False],   # Thumb up
    "FOOD":      [True,  True,  True,  True,  False],   # Four fingers
    "PLEASE":    [False, False, False, False, False],   # Flat hand (prayer)
    "THANK_YOU": [True,  True,  True,  True,  True],    # Open hand forward
    "WHERE":     [False, True,  False, False, False],   # Pointing with wag
    "HOSPITAL":  [False, True,  True,  False, False],   # Cross motion
    "NEED":      [False, False, False, False, False],   # Grasping
    "WANT":      [True,  True,  True,  True,  True],    # Reaching hand
    "GO":        [False, True,  False, False, False],   # Pointing direction
    "COME":      [False, True,  True,  True,  True],    # Beckoning
    "MORE":      [True,  True,  True,  False, False],   # Pinching
    "STOP":      [True,  True,  True,  True,  True],    # Flat palm
}


def generate_base_hand(finger_states, hand_scale=1.0):
    """
    Generate a single anatomically-plausible hand pose.

    Returns: np.array of shape (21, 3) with x, y, z values in [0, 1] range.

    Hand anatomy layout (normalized coordinates):
    - Wrist at bottom center (~0.5, 0.8)
    - Fingers extend upward (lower y = higher on screen)
    - x spreads across [0.3, 0.7] for finger tips
    """
    landmarks = np.zeros((21, 3))

    # Wrist position (with variation)
    wrist_x = 0.5 + np.random.normal(0, 0.03)
    wrist_y = 0.75 + np.random.normal(0, 0.04)
    wrist_z = 0.0 + np.random.normal(0, 0.01)
    landmarks[0] = [wrist_x, wrist_y, wrist_z]

    # Finger base positions (MCP joints, at the knuckle line)
    finger_bases_x = [
        wrist_x - 0.12 * hand_scale,   # Thumb CMC (far left)
        wrist_x - 0.06 * hand_scale,   # Index MCP
        wrist_x - 0.01 * hand_scale,   # Middle MCP
        wrist_x + 0.04 * hand_scale,   # Ring MCP
        wrist_x + 0.09 * hand_scale,   # Pinky MCP
    ]
    finger_bases_y = [
        wrist_y - 0.08 * hand_scale,   # Thumb
        wrist_y - 0.18 * hand_scale,   # Index
        wrist_y - 0.20 * hand_scale,   # Middle (longest)
        wrist_y - 0.18 * hand_scale,   # Ring
        wrist_y - 0.15 * hand_scale,   # Pinky
    ]

    # Finger segment lengths (proportional)
    segment_lengths = [
        [0.04, 0.03, 0.025, 0.02],   # Thumb (shorter)
        [0.02, 0.04, 0.03, 0.025],   # Index
        [0.02, 0.045, 0.035, 0.03],  # Middle (longest)
        [0.02, 0.04, 0.03, 0.025],   # Ring
        [0.02, 0.035, 0.025, 0.02],  # Pinky (shortest)
    ]

    thumb_up, index_up, middle_up, ring_up, pinky_up = finger_states
    states = [thumb_up, index_up, middle_up, ring_up, pinky_up]

    for finger_idx in range(5):
        is_up = states[finger_idx]
        base_x = finger_bases_x[finger_idx]
        base_y = finger_bases_y[finger_idx]
        segs = segment_lengths[finger_idx]

        # Landmark indices for this finger
        if finger_idx == 0:
            indices = [1, 2, 3, 4]  # Thumb: CMC, MCP, IP, TIP
        else:
            indices = [
                5 + (finger_idx - 1) * 4,
                6 + (finger_idx - 1) * 4,
                7 + (finger_idx - 1) * 4,
                8 + (finger_idx - 1) * 4,
            ]

        current_x = base_x
        current_y = base_y
        current_z = wrist_z

        for joint_idx, seg_len in enumerate(segs):
            seg_scaled = seg_len * hand_scale

            if is_up:
                # Finger extended upward
                if finger_idx == 0:
                    # Thumb extends outward and slightly up
                    current_x -= seg_scaled * 0.7 + np.random.normal(0, 0.003)
                    current_y -= seg_scaled * 0.5 + np.random.normal(0, 0.003)
                else:
                    # Other fingers extend upward
                    current_y -= seg_scaled + np.random.normal(0, 0.004)
                    current_x += np.random.normal(0, 0.002)
            else:
                # Finger curled inward
                if finger_idx == 0:
                    # Thumb curls inward
                    current_x += seg_scaled * 0.3 + np.random.normal(0, 0.003)
                    current_y -= seg_scaled * 0.2 + np.random.normal(0, 0.003)
                else:
                    # Other fingers curl downward (tip.y > pip.y when curled)
                    if joint_idx < 2:
                        current_y -= seg_scaled * 0.3 + np.random.normal(0, 0.003)
                    else:
                        current_y += seg_scaled * 0.6 + np.random.normal(0, 0.003)
                    current_x += np.random.normal(0, 0.003)

            current_z += np.random.normal(0, 0.005)
            landmarks[indices[joint_idx]] = [current_x, current_y, current_z]

    return landmarks


def add_natural_jitter(landmarks, jitter_scale=0.005):
    """Add small random noise to simulate natural hand tremor."""
    noise = np.random.normal(0, jitter_scale, landmarks.shape)
    return landmarks + noise


def apply_rotation(landmarks, angle_deg):
    """Rotate hand landmarks around the wrist by a given angle (degrees)."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    wrist = landmarks[0].copy()
    rotated = landmarks.copy()

    for i in range(1, len(landmarks)):
        dx = landmarks[i, 0] - wrist[0]
        dy = landmarks[i, 1] - wrist[1]
        rotated[i, 0] = wrist[0] + dx * cos_a - dy * sin_a
        rotated[i, 1] = wrist[1] + dx * sin_a + dy * cos_a

    return rotated


def apply_scale(landmarks, scale_factor):
    """Scale hand from wrist point."""
    wrist = landmarks[0].copy()
    scaled = landmarks.copy()
    for i in range(1, len(landmarks)):
        scaled[i] = wrist + (landmarks[i] - wrist) * scale_factor
    return scaled


def apply_translation(landmarks, dx, dy):
    """Shift the entire hand."""
    translated = landmarks.copy()
    translated[:, 0] += dx
    translated[:, 1] += dy
    return translated


def generate_sequence(finger_states, with_motion=True):
    """
    Generate a 30-frame sequence of hand landmarks for a gesture.
    Adds subtle frame-to-frame motion to simulate real video capture.
    """
    frames = []

    # Base hand pose
    hand_scale = np.random.uniform(0.85, 1.15)
    base_landmarks = generate_base_hand(finger_states, hand_scale)

    # Motion parameters (slow drift during sequence)
    drift_x = np.random.normal(0, 0.001)
    drift_y = np.random.normal(0, 0.001)
    rotation_drift = np.random.normal(0, 0.3)  # degrees per frame

    for frame_idx in range(SEQUENCE_LENGTH):
        frame_landmarks = base_landmarks.copy()

        if with_motion:
            # Progressive drift
            frame_landmarks = apply_translation(
                frame_landmarks,
                drift_x * frame_idx,
                drift_y * frame_idx,
            )
            # Slight rotation over time
            frame_landmarks = apply_rotation(
                frame_landmarks,
                rotation_drift * frame_idx,
            )

        # Natural hand tremor
        frame_landmarks = add_natural_jitter(frame_landmarks, jitter_scale=0.004)

        # Flatten to 63 values
        frames.append(frame_landmarks.flatten())

    return np.array(frames)


def augment_sequence(sequence, num_augmentations=2):
    """Create augmented versions of a sequence with transformations."""
    augmented = []

    for _ in range(num_augmentations):
        aug_frames = []
        # Random global transforms
        scale = np.random.uniform(0.8, 1.2)
        rotation = np.random.uniform(-15, 15)
        shift_x = np.random.uniform(-0.08, 0.08)
        shift_y = np.random.uniform(-0.08, 0.08)
        noise_scale = np.random.uniform(0.002, 0.008)

        for frame in sequence:
            landmarks = frame.reshape(21, 3)
            landmarks = apply_scale(landmarks, scale)
            landmarks = apply_rotation(landmarks, rotation)
            landmarks = apply_translation(landmarks, shift_x, shift_y)
            landmarks = add_natural_jitter(landmarks, noise_scale)
            aug_frames.append(landmarks.flatten())

        augmented.append(np.array(aug_frames))

    return augmented


def generate_dataset(num_sequences=30, num_augmentations=3):
    """
    Generate the full synthetic dataset for all gestures.

    Args:
        num_sequences: Base sequences per gesture
        num_augmentations: Extra augmented copies per sequence

    Total samples per gesture = num_sequences × (1 + num_augmentations)
    """
    total_per_gesture = num_sequences * (1 + num_augmentations)
    total_samples = total_per_gesture * len(GESTURES)

    print(f"Generating synthetic dataset:")
    print(f"  Gestures:        {len(GESTURES)}")
    print(f"  Base sequences:  {num_sequences} per gesture")
    print(f"  Augmentations:   {num_augmentations} per sequence")
    print(f"  Total per class: {total_per_gesture}")
    print(f"  Total samples:   {total_samples}")
    print(f"  Sequence shape:  ({SEQUENCE_LENGTH}, {NUM_FEATURES})")
    print()

    for gesture in GESTURES:
        gesture_dir = os.path.join(DATASET_PATH, gesture)
        os.makedirs(gesture_dir, exist_ok=True)

        finger_states = GESTURE_DEFINITIONS[gesture]
        seq_count = 0

        for base_idx in range(num_sequences):
            # Generate base sequence
            sequence = generate_sequence(finger_states, with_motion=True)

            # Save base
            seq_count += 1
            npy_path = os.path.join(gesture_dir, f"seq_{seq_count}.npy")
            np.save(npy_path, sequence)

            # Generate and save augmented versions
            augmented_sequences = augment_sequence(sequence, num_augmentations)
            for aug_seq in augmented_sequences:
                seq_count += 1
                npy_path = os.path.join(gesture_dir, f"seq_{seq_count}.npy")
                np.save(npy_path, aug_seq)

        print(f"  ✅ {gesture:12s} — {seq_count} sequences saved")

    print(f"\n✅ Dataset saved to: {os.path.abspath(DATASET_PATH)}")
    print(f"   Total .npy files: {total_samples}")


def verify_dataset():
    """Quick verification of the generated dataset."""
    print("\n── Dataset Verification ──")
    total = 0
    for gesture in GESTURES:
        gesture_dir = os.path.join(DATASET_PATH, gesture)
        if not os.path.exists(gesture_dir):
            print(f"  ❌ {gesture}: directory missing")
            continue

        files = [f for f in os.listdir(gesture_dir) if f.endswith(".npy")]
        count = len(files)
        total += count

        # Verify shape of first file
        if count > 0:
            sample = np.load(os.path.join(gesture_dir, files[0]))
            shape_ok = "✅" if sample.shape == (SEQUENCE_LENGTH, NUM_FEATURES) else "❌"
            print(f"  {shape_ok} {gesture:12s} — {count:3d} sequences, shape: {sample.shape}")
        else:
            print(f"  ⚠️  {gesture:12s} — 0 sequences")

    print(f"\n  Total sequences: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic ISL gesture dataset")
    parser.add_argument("--sequences", type=int, default=30,
                        help="Base sequences per gesture (default: 30)")
    parser.add_argument("--augmentation", type=int, default=3,
                        help="Augmented copies per sequence (default: 3)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing dataset without generating")
    args = parser.parse_args()

    if args.verify_only:
        verify_dataset()
    else:
        generate_dataset(
            num_sequences=args.sequences,
            num_augmentations=args.augmentation,
        )
        verify_dataset()
