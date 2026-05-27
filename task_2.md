# Task 2: Emotion Recognition from Speech 🎙️

## Objective
Recognize human emotions (**neutral, calm, happy, sad, angry, fearful, disgust, surprised**) from raw speech audio.

## Approach
Speech Signal Processing → Feature Extraction → CNN + LSTM Deep Learning Model

## Architecture
```
Input Audio (.wav)
      ↓
Preprocessing (trim, pad/truncate to 3s)
      ↓
Feature Extraction:
  • MFCCs (40) + Δ + ΔΔ   → timbre / spectral shape
  • Chroma (12)             → pitch class energy
  • Mel Spectrogram (128)   → perceptual frequency
  • Spectral Contrast (7)   → peak/valley in spectrum
  • Tonnetz (6)             → tonal centroid
      ↓
CNN-LSTM Model
  Conv1D → BatchNorm → MaxPool → Dropout ×2
  LSTM(128) → LSTM(64)
  Dense(128) → Dense(n_classes) softmax
      ↓
Emotion Label
```

## Dataset Setup (RAVDESS)
1. Download RAVDESS from [Zenodo](https://zenodo.org/record/1188976)
2. Extract to `./data/RAVDESS/`
3. Structure should be: `Actor_01/`, `Actor_02/`, …

## How to Run
```bash
pip install -r requirements.txt
python emotion_recognition.py
```
> Without the dataset the script runs in **demo mode** with synthetic data.

## Outputs
- `emotion_training_history.png` — accuracy & loss curves
- `emotion_confusion_matrix.png` — per-emotion confusion matrix
- `emotion_model.h5` — saved trained model
