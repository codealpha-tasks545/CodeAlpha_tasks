"""
CodeAlpha Internship - Task 3: Handwritten Character Recognition
Objective: Identify handwritten digits (MNIST) and characters (EMNIST).
Approach : Convolutional Neural Networks (CNN) with data augmentation.
Author   : [Your Name]
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout,
                                          Flatten, BatchNormalization, Input,
                                          GlobalAveragePooling2D)
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_OK = True
except ImportError:
    TF_OK = False
    print("[!] TensorFlow not found. Install: pip install tensorflow")

from sklearn.metrics import classification_report, confusion_matrix

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
DATASET     = 'mnist'   # 'mnist' or 'emnist'
IMG_SIZE    = 28
BATCH_SIZE  = 128
EPOCHS      = 30
MODEL_PATH  = 'handwriting_model.h5'

MNIST_CLASSES   = [str(i) for i in range(10)]
EMNIST_CLASSES  = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')

# ──────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────

def load_mnist():
    """Load standard MNIST digits dataset."""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32')  / 255.0
    X_train = X_train[..., np.newaxis]
    X_test  = X_test[...,  np.newaxis]
    n_classes = 10
    class_names = MNIST_CLASSES
    return X_train, X_test, y_train, y_test, n_classes, class_names


def load_emnist():
    """
    Load EMNIST letters dataset via tensorflow_datasets.
    Install: pip install tensorflow-datasets
    Falls back to MNIST if unavailable.
    """
    try:
        import tensorflow_datasets as tfds
        ds_train, ds_info = tfds.load('emnist/letters', split='train', with_info=True, as_supervised=True)
        ds_test  = tfds.load('emnist/letters', split='test',  as_supervised=True)

        def preprocess(img, label):
            img = tf.cast(img, tf.float32) / 255.0
            img = tf.image.transpose(img)   # EMNIST images are transposed
            return img, label - 1           # labels are 1-indexed

        ds_train = ds_train.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        ds_test  = ds_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        n_classes = 26
        class_names = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        return ds_train, ds_test, None, None, n_classes, class_names, True
    except Exception as e:
        print(f"[!] EMNIST load failed ({e}). Falling back to MNIST.")
        X_train, X_test, y_train, y_test, n_classes, class_names = load_mnist()
        return X_train, X_test, y_train, y_test, n_classes, class_names, False


# ──────────────────────────────────────────────
# 2. CNN MODEL
# ──────────────────────────────────────────────

def build_model(n_classes: int, img_size: int = IMG_SIZE) -> 'tf.keras.Model':
    """
    Deep CNN with:
      - 3 convolutional blocks (Conv → BN → ReLU → Pool → Dropout)
      - Global Average Pooling (fewer parameters vs Flatten)
      - Two fully-connected layers
    """
    inputs = Input(shape=(img_size, img_size, 1))

    # Block 1
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 3
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Classification head
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ──────────────────────────────────────────────
# 3. DATA AUGMENTATION
# ──────────────────────────────────────────────

def get_augmentor():
    return ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.1,
        fill_mode='nearest',
    )


# ──────────────────────────────────────────────
# 4. VISUALISATIONS
# ──────────────────────────────────────────────

def plot_samples(X, y, class_names, n=20):
    """Show a random grid of training samples."""
    indices = np.random.choice(len(X), n, replace=False)
    cols = 5
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.5))
    fig.suptitle('Sample Training Images', fontsize=14, fontweight='bold')
    for ax, idx in zip(axes.flatten(), indices):
        ax.imshow(X[idx].squeeze(), cmap='gray_r')
        ax.set_title(class_names[y[idx]], fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_training(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Handwriting Recognition — Training History', fontsize=14, fontweight='bold')

    ax1.plot(history.history['accuracy'],     label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val',   linewidth=2, linestyle='--')
    ax1.set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'],     label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val',   linewidth=2, linestyle='--')
    ax2.set(xlabel='Epoch', ylabel='Loss', title='Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('handwriting_training.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    fig_size = max(8, len(class_names))
    plt.figure(figsize=(fig_size, fig_size - 1))
    sns.heatmap(cm, annot=(len(class_names) <= 12), fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Handwriting Recognition — Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted'); plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('handwriting_confusion.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_predictions(model, X_test, y_test, class_names, n=25):
    """Show predictions with green (correct) / red (wrong) borders."""
    indices = np.random.choice(len(X_test), n, replace=False)
    X_sample = X_test[indices]
    y_sample = y_test[indices]
    y_pred   = np.argmax(model.predict(X_sample, verbose=0), axis=1)

    cols = 5
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, rows * 2.5))
    fig.suptitle('Model Predictions  (✓ green = correct  ✗ red = wrong)', fontsize=13, fontweight='bold')
    for ax, img, true, pred in zip(axes.flatten(), X_sample, y_sample, y_pred):
        ax.imshow(img.squeeze(), cmap='gray_r')
        color = 'green' if true == pred else 'red'
        ax.set_title(f'T:{class_names[true]}  P:{class_names[pred]}', fontsize=9, color=color)
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(3)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('handwriting_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()


# ──────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  CodeAlpha — Task 3: Handwritten Character Recognition")
    print("=" * 60)

    if not TF_OK:
        import sys; sys.exit(1)

    # Load data
    print(f"\n[~] Loading {'EMNIST' if DATASET == 'emnist' else 'MNIST'} dataset…")
    X_train, X_test, y_train, y_test, n_classes, class_names = load_mnist()
    print(f"[✓] Train : {X_train.shape}   Test : {X_test.shape}")
    print(f"[✓] Classes ({n_classes}): {class_names}")

    # Visualise samples
    plot_samples(X_train, y_train, class_names)

    # Build model
    model = build_model(n_classes)
    model.summary()
    print(f"\n[✓] Total parameters: {model.count_params():,}")

    # Data augmentation
    augmentor = get_augmentor()
    train_gen = augmentor.flow(X_train, y_train, batch_size=BATCH_SIZE)

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=0),
    ]

    print("\n[~] Training CNN…")
    history = model.fit(
        train_gen,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print("\n" + "=" * 60)
    print(classification_report(y_test, y_pred, target_names=class_names))
    print(f"  Test Accuracy : {test_acc:.4f}")
    print(f"  Test Loss     : {test_loss:.4f}")
    print("=" * 60)

    # Plots
    plot_training(history)
    plot_confusion(y_test, y_pred, class_names)
    plot_predictions(model, X_test, y_test, class_names)

    print(f"\n[✓] Model saved → {MODEL_PATH}")
