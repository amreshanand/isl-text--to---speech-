"""
Enhanced LSTM Training Pipeline for ISL Gesture Recognition
============================================================
Features:
  - LSTM + Dense architecture for sequence classification
  - Data augmentation on-the-fly
  - Early stopping + learning rate reduction
  - Training history plots (saved as PNG)
  - Per-class accuracy report
  - Confusion matrix visualization

Usage:
    python train_lstm.py
    python train_lstm.py --epochs 100 --batch-size 64
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Bidirectional,
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ── Configuration ──────────────────────────────────────────────────────

GESTURES = [
    "HELLO", "WATER", "HELP", "YES", "NO",
    "I", "YOU", "EAT", "DRINK", "FOOD",
    "PLEASE", "THANK_YOU", "WHERE", "HOSPITAL",
    "NEED", "WANT", "GO", "COME", "MORE", "STOP",
]

DATASET_PATH = os.path.join("..", "dataset")
MODEL_DIR = "saved_model"
MODEL_PATH = os.path.join(MODEL_DIR, "isl_model.h5")
TFLITE_PATH = os.path.join(MODEL_DIR, "isl_model.tflite")
HISTORY_PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")

SEQUENCE_LENGTH = 30
NUM_FEATURES = 63
NUM_CLASSES = len(GESTURES)

label_map = {label: idx for idx, label in enumerate(GESTURES)}


# ── Data Loading ───────────────────────────────────────────────────────

def load_dataset():
    """Load all .npy sequence files from the dataset directory."""
    X, y = [], []

    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at: {os.path.abspath(DATASET_PATH)}")
        print("   Run: python generate_synthetic_data.py")
        return np.array([]), np.array([])

    for gesture in GESTURES:
        gesture_dir = os.path.join(DATASET_PATH, gesture)
        if not os.path.exists(gesture_dir):
            print(f"  ⚠️  Missing: {gesture}")
            continue

        count = 0
        for file in sorted(os.listdir(gesture_dir)):
            if not file.endswith(".npy"):
                continue
            sequence = np.load(os.path.join(gesture_dir, file))
            if sequence.shape == (SEQUENCE_LENGTH, NUM_FEATURES):
                X.append(sequence)
                y.append(label_map[gesture])
                count += 1
            else:
                print(f"  ⚠️  Skipped {file}: shape {sequence.shape}")

        print(f"  ✅ {gesture:12s}: {count} sequences")

    return np.array(X), np.array(y)


def augment_batch(X_batch, noise_range=0.008):
    """Apply random noise augmentation to a batch during training."""
    noise = np.random.normal(0, noise_range, X_batch.shape)
    return X_batch + noise


# ── Model Architecture ────────────────────────────────────────────────

def build_model(model_type="bidirectional"):
    """
    Build the gesture recognition model.

    Args:
        model_type: "simple", "stacked", or "bidirectional"
    """
    model = Sequential(name=f"ISL_{model_type}_model")

    if model_type == "simple":
        model.add(LSTM(64, activation="relu",
                       input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation="relu"))

    elif model_type == "stacked":
        model.add(LSTM(128, return_sequences=True, activation="relu",
                       input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(LSTM(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation="relu"))

    elif model_type == "bidirectional":
        model.add(Bidirectional(
            LSTM(128, return_sequences=True, activation="relu"),
            input_shape=(SEQUENCE_LENGTH, NUM_FEATURES),
        ))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Bidirectional(LSTM(64, activation="relu")))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

    model.add(Dense(NUM_CLASSES, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── Training Callbacks ─────────────────────────────────────────────────

def get_callbacks():
    """Return training callbacks for better convergence."""
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, "best_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]


# ── Evaluation ─────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation with per-class metrics."""
    # Overall metrics
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*50}")
    print(f"  TEST RESULTS")
    print(f"{'='*50}")
    print(f"  Loss:     {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"{'='*50}\n")

    # Per-class report
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    print("── Per-Class Classification Report ──\n")
    report = classification_report(
        y_true_classes, y_pred_classes,
        target_names=GESTURES,
        digits=3,
        zero_division=0,
    )
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    print("── Confusion Matrix ──\n")

    # Header
    header = "          " + " ".join(f"{g[:4]:>4}" for g in GESTURES)
    print(header)
    for i, gesture in enumerate(GESTURES):
        row = f"{gesture[:8]:>8}  " + " ".join(f"{cm[i, j]:4d}" for j in range(len(GESTURES)))
        print(row)

    # Per-class accuracy
    print("\n── Per-Class Accuracy ──\n")
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    for i, gesture in enumerate(GESTURES):
        bar = "█" * int(per_class_acc[i] * 20)
        print(f"  {gesture:12s} {per_class_acc[i]*100:5.1f}% {bar}")

    return accuracy


def save_training_plots(history):
    """Save training history plots as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy plot
        axes[0].plot(history.history["accuracy"], label="Train", linewidth=2)
        axes[0].plot(history.history["val_accuracy"], label="Validation", linewidth=2)
        axes[0].set_title("Model Accuracy", fontsize=14, fontweight="bold")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Loss plot
        axes[1].plot(history.history["loss"], label="Train", linewidth=2)
        axes[1].plot(history.history["val_loss"], label="Validation", linewidth=2)
        axes[1].set_title("Model Loss", fontsize=14, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(HISTORY_PLOT_PATH, dpi=150, bbox_inches="tight")
        print(f"  📊 Training plots saved to: {HISTORY_PLOT_PATH}")
        plt.close()

    except ImportError:
        print("  ⚠️  matplotlib not installed — skipping training plots")


def convert_to_tflite(model):
    """Convert Keras model to TFLite for mobile/edge deployment."""
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # Enable Select TF ops to support LSTM TensorListReserve ops
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter._experimental_lower_tensor_list_ops = False
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

        with open(TFLITE_PATH, "wb") as f:
            f.write(tflite_model)

        size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
        print(f"  📱 TFLite model saved: {TFLITE_PATH} ({size_mb:.2f} MB)")

    except Exception as e:
        print(f"  ⚠️  TFLite conversion failed: {e}")


# ── Main ───────────────────────────────────────────────────────────────

def main(args):
    print("=" * 55)
    print("  ISL Gesture Recognition — LSTM Training Pipeline")
    print("=" * 55)
    print()

    # 1. Load dataset
    print("── Loading Dataset ──\n")
    X, y = load_dataset()

    if len(X) == 0:
        print("\n❌ No data found. Run this first:")
        print("   python generate_synthetic_data.py")
        return

    y_cat = to_categorical(y, num_classes=NUM_CLASSES)
    print(f"\n  Total: {X.shape[0]} sequences")
    print(f"  X shape: {X.shape}")
    print(f"  Classes: {NUM_CLASSES}")

    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.15, random_state=42, stratify=y,
    )
    print(f"\n  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # 3. Apply training augmentation
    if args.augment:
        print(f"\n  Applying noise augmentation to training data...")
        X_train_aug = augment_batch(X_train.copy())
        X_train = np.concatenate([X_train, X_train_aug])
        y_train = np.concatenate([y_train, y_train])
        print(f"  Augmented train size: {X_train.shape[0]}")

    # 4. Build model
    print(f"\n── Building {args.model_type.upper()} Model ──\n")
    os.makedirs(MODEL_DIR, exist_ok=True)

    model = build_model(model_type=args.model_type)
    model.summary()

    # 5. Train
    print(f"\n── Training ({args.epochs} epochs, batch size {args.batch_size}) ──\n")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        callbacks=get_callbacks(),
        verbose=1,
    )

    # 6. Evaluate
    accuracy = evaluate_model(model, X_test, y_test)

    # 7. Save model
    print("\n── Saving Model ──\n")
    model.save(MODEL_PATH)
    model_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"  💾 Keras model saved: {MODEL_PATH} ({model_size:.2f} MB)")

    # 8. Save plots
    save_training_plots(history)

    # 9. Convert to TFLite (optional)
    if args.tflite:
        convert_to_tflite(model)

    # 10. Summary
    print(f"\n{'='*55}")
    print(f"  ✅ TRAINING COMPLETE")
    print(f"     Accuracy: {accuracy*100:.1f}%")
    print(f"     Model:    {MODEL_PATH}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ISL gesture LSTM model")
    parser.add_argument("--epochs", type=int, default=80, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--model-type", choices=["simple", "stacked", "bidirectional"],
                        default="bidirectional", help="Model architecture")
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Apply data augmentation")
    parser.add_argument("--no-augment", action="store_false", dest="augment",
                        help="Disable data augmentation")
    parser.add_argument("--tflite", action="store_true",
                        help="Also export TFLite model for mobile")
    args = parser.parse_args()

    main(args)
