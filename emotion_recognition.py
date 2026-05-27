"""
CodeAlpha Internship - Task 2: Emotion Recognition from Speech
Objective: Recognize human emotions from speech audio using MFCCs + deep learning.
Author: [Your Name]

Supported Datasets: RAVDESS, TESS, EMO-DB
To use a real dataset, set DATA_DIR to your extracted folder and run.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ─── Optional heavy imports (graceful fallback for environments without audio libs) ───
try:
    import librosa
    import librosa.display
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False
    print("[!] librosa not found. Install with: pip install librosa")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (Dense, Dropout, Conv1D, MaxPooling1D,
                                          LSTM, BatchNormalization, Flatten, GlobalAveragePooling1D)
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    TF_OK = True
except ImportError:
    TF_OK = False
    print("[!] TensorFlow not found. Install with: pip install tensorflow")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
DATA_DIR   = "./data/RAVDESS"   # ← change to your dataset path
EMOTIONS   = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
SAMPLE_RATE = 22050
DURATION    = 3          # seconds per clip
N_MFCC      = 40
BATCH_SIZE  = 32
EPOCHS      = 80
MODEL_PATH  = "emotion_model.h5"

# RAVDESS filename emotion map (3rd token, 1-indexed)
RAVDESS_EMOTION_MAP = {
    '01': 'neutral',  '02': 'calm',    '03': 'happy',
    '04': 'sad',      '05': 'angry',   '06': 'fearful',
    '07': 'disgust',  '08': 'surprised'
}

# ──────────────────────────────────────────────
# 1. FEATURE EXTRACTION
# ──────────────────────────────────────────────

def extract_features(file_path: str, sr: int = SAMPLE_RATE, duration: int = DURATION) -> np.ndarray | None:
    """
    Extract audio features:
      - MFCCs (40)                 → timbre / spectral shape
      - Delta MFCCs (40)           → first derivative
      - Delta-Delta MFCCs (40)     → second derivative
      - Chroma (12)                → pitch class energy
      - Mel Spectrogram (128 mean) → perceptual frequency bands
      - Spectral Contrast (7)      → peak vs valley in spectrum
      - Tonnetz (6)                → tonal centroid features
    Total: 273-dim vector per file
    """
    if not LIBROSA_OK:
        return None
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration)
        y, _ = librosa.effects.trim(y)

        # Pad / truncate to fixed length
        max_len = sr * duration
        if len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)))
        else:
            y = y[:max_len]

        mfcc        = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        delta       = librosa.feature.delta(mfcc)
        delta2      = librosa.feature.delta(mfcc, order=2)
        chroma      = librosa.feature.chroma_stft(y=y, sr=sr)
        mel         = librosa.feature.melspectrogram(y=y, sr=sr)
        contrast    = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz     = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)

        features = np.hstack([
            np.mean(mfcc,     axis=1),
            np.mean(delta,    axis=1),
            np.mean(delta2,   axis=1),
            np.mean(chroma,   axis=1),
            np.mean(mel,      axis=1),
            np.mean(contrast, axis=1),
            np.mean(tonnetz,  axis=1),
        ])
        return features
    except Exception as e:
        print(f"  [!] Error processing {file_path}: {e}")
        return None


def load_ravdess(data_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load RAVDESS dataset from directory (Actor_* subfolders)."""
    X, y = [], []
    actors = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    for actor in sorted(actors):
        actor_path = os.path.join(data_dir, actor)
        for fname in os.listdir(actor_path):
            if not fname.endswith('.wav'):
                continue
            parts = fname.split('-')
            emotion_code = parts[2]
            emotion = RAVDESS_EMOTION_MAP.get(emotion_code)
            if emotion is None:
                continue
            fpath = os.path.join(actor_path, fname)
            feat = extract_features(fpath)
            if feat is not None:
                X.append(feat)
                y.append(emotion)
    return np.array(X), np.array(y)


def generate_dummy_data(n_samples: int = 800, n_features: int = 273) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic fallback when real audio data is unavailable.
    Each class gets its own Gaussian cluster.
    """
    np.random.seed(42)
    X_parts, y_parts = [], []
    for i, emotion in enumerate(EMOTIONS):
        n = n_samples // len(EMOTIONS)
        centre = np.random.randn(n_features) * (i + 1)
        data   = centre + np.random.randn(n, n_features) * 0.5
        X_parts.append(data)
        y_parts.extend([emotion] * n)
    return np.vstack(X_parts), np.array(y_parts)


# ──────────────────────────────────────────────
# 2. BUILD MODEL
# ──────────────────────────────────────────────

def build_cnn_lstm_model(input_shape: tuple, n_classes: int) -> 'tf.keras.Model':
    """
    CNN + LSTM hybrid:
      Conv1D layers capture local temporal patterns in the feature sequence.
      LSTM captures long-range temporal dependencies.
    """
    model = Sequential([
        # CNN block
        Conv1D(128, 5, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),

        Conv1D(64, 3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        Dropout(0.3),

        # LSTM block
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dropout(0.3),

        # Classification head
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ──────────────────────────────────────────────
# 3. VISUALISATIONS
# ──────────────────────────────────────────────

def plot_training(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Emotion Recognition — Training History', fontsize=14, fontweight='bold')

    ax1.plot(history.history['accuracy'],     label='Train Accuracy',  linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy',    linewidth=2, linestyle='--')
    ax1.set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'],     label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss',   linewidth=2, linestyle='--')
    ax2.set(xlabel='Epoch', ylabel='Loss', title='Loss')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('emotion_training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=classes, yticklabels=classes)
    plt.title('Emotion Recognition — Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('emotion_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_mfcc_sample(file_path: str):
    """Visualise MFCC of a single audio file."""
    if not LIBROSA_OK:
        return
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'MFCC — {os.path.basename(file_path)}')
    plt.tight_layout()
    plt.savefig('sample_mfcc.png', dpi=150, bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────
# 4. MAIN
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  CodeAlpha — Task 2: Emotion Recognition from Speech")
    print("=" * 60)

    # Load data
    use_real = LIBROSA_OK and os.path.isdir(DATA_DIR)
    if use_real:
        print(f"\n[✓] Loading RAVDESS dataset from: {DATA_DIR}")
        X, y_raw = load_ravdess(DATA_DIR)
    else:
        print("\n[!] Real data not found → using synthetic demo data.")
        print("    Set DATA_DIR to your RAVDESS folder for real results.")
        X, y_raw = generate_dummy_data()

    print(f"[✓] Samples : {len(X)}")
    print(f"[✓] Features: {X.shape[1]}")
    print(f"[✓] Emotions: {np.unique(y_raw)}")

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for CNN-LSTM (samples, timesteps, features)
    X_3d = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_3d, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    y_train_cat = to_categorical(y_train, n_classes)
    y_test_cat  = to_categorical(y_test,  n_classes)

    if not TF_OK:
        print("\n[!] TensorFlow not available. Cannot train model.")
        sys.exit(0)

    # Build & train
    model = build_cnn_lstm_model(input_shape=(X_3d.shape[1], 1), n_classes=n_classes)
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6),
    ]

    print("\n[~] Training model…")
    history = model.fit(
        X_train, y_train_cat,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    y_pred_prob = model.predict(X_test)
    y_pred      = np.argmax(y_pred_prob, axis=1)

    print("\n" + "=" * 60)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"  Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("=" * 60)

    # Save model
    model.save(MODEL_PATH)
    print(f"\n[✓] Model saved → {MODEL_PATH}")

    # Plots
    plot_training(history)
    plot_confusion(y_test, y_pred, le.classes_)
