# Task 3: Handwritten Character Recognition ✍️

## Objective
Identify handwritten **digits (0–9)** using MNIST and optionally **letters (A–Z)** using EMNIST.

## Architecture — Deep CNN
```
Input (28×28×1)
  │
  ├─ Conv Block 1: Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
  ├─ Conv Block 2: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
  ├─ Conv Block 3: Conv2D(128) → BN → MaxPool → Dropout(0.25)
  │
  ├─ GlobalAveragePooling2D
  ├─ Dense(256) → BN → Dropout(0.5)
  ├─ Dense(128) → ReLU
  └─ Dense(n_classes) → Softmax
```

## Data Augmentation
- Rotation ±10°
- Width/height shift ±10%
- Zoom ±10%
- Shear ±10%

## Datasets
| Dataset | Classes | Samples | Auto-download |
|---------|---------|---------|---------------|
| MNIST   | 10 digits | 70,000 | ✅ via Keras |
| EMNIST Letters | 26 chars | 145,600 | ✅ via `tensorflow-datasets` |

## How to Run

### MNIST (default)
```bash
pip install -r requirements.txt
python handwriting_recognition.py
```

### EMNIST (change config at top of script)
```bash
pip install tensorflow-datasets
# Set DATASET = 'emnist' in the script
python handwriting_recognition.py
```

## Expected Results (MNIST)
| Metric | Value |
|--------|-------|
| Test Accuracy | ~99.4% |
| Test Loss     | ~0.02  |

## Outputs
- `sample_images.png` — random training samples grid
- `handwriting_training.png` — accuracy & loss curves
- `handwriting_confusion.png` — confusion matrix
- `handwriting_predictions.png` — example predictions (green=correct, red=wrong)
- `handwriting_model.h5` — saved model
