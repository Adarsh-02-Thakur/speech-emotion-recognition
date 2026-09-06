# 🎙️ Speech Emotion Recognition (Task 2)

Recognize human emotions (happy, angry, sad, etc.) from speech audio using
MFCC features and a CNN + LSTM deep learning model. Fully working, end to
end: feature extraction → training → evaluation → inference → a browser demo.

## Project structure

```
speech_emotion_recognition/
├── data/                     # Put your dataset here (RAVDESS / TESS / EMO-DB)
├── models/                   # Trained model + label encoder get saved here
├── static/uploads/           # Temp storage for files uploaded via the web demo
├── templates/index.html      # Web demo front-end
├── utils/
│   ├── feature_extraction.py # MFCC extraction + filename label parsing
│   └── model_builder.py      # CNN+LSTM and simple-CNN architectures
├── train.py                  # Training pipeline
├── predict.py                # CLI inference on a single .wav file
├── app.py                    # Flask web demo
└── requirements.txt
```

## 1. Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get a dataset

Pick one (RAVDESS is the easiest to start with):

| Dataset | Link | Notes |
|---|---|---|
| **RAVDESS** | https://zenodo.org/record/1188976 | ~1,440 files, 8 emotions, `Actor_01..24` folders |
| **TESS** | https://tspace.library.utoronto.ca/handle/1807/24487 | 2,800 files, emotion word is in the filename |
| **EMO-DB** | http://emodb.bilderbar.info/download/ | German dataset, ~535 files |

Download and unzip so you end up with something like:

```
data/RAVDESS/Actor_01/03-01-05-01-01-01-01.wav
data/RAVDESS/Actor_02/...
...
```

You don't need to reorganize anything — `train.py` walks the folder
recursively and parses the emotion label straight from each filename.

## 3. Train the model

```bash
python train.py --data_dir data/RAVDESS --dataset ravdess --epochs 50
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | — | Root folder of the dataset (required) |
| `--dataset` | `ravdess` | `ravdess`, `tess`, or `emodb` — controls how labels are parsed |
| `--architecture` | `cnn_lstm` | `cnn_lstm` (more accurate) or `simple_cnn` (faster) |
| `--epochs` | 50 | Training epochs (early stopping is enabled) |
| `--batch_size` | 32 | Batch size |
| `--n_mfcc` | 40 | Number of MFCC coefficients extracted per frame |
| `--max_pad_len` | 174 | Fixed time-axis length each clip is padded/truncated to |
| `--model_dir` | `models` | Where to save the trained model + encoder |

This will print live training progress, then a final test-set accuracy, and save:

- `models/emotion_model.h5` — the trained model
- `models/best_model.h5` — best checkpoint by validation accuracy
- `models/label_encoder.pkl` — maps class indices back to emotion names
- `models/norm_stats.pkl` — normalization stats needed for inference
- `models/training_curves.png` — accuracy/loss plots

**Typical accuracy:** on RAVDESS with the CNN+LSTM model and ~50 epochs,
expect roughly 65-80% test accuracy (varies with random seed and how long
you train — this is a hard, small-data task even in published research).

## 4. Predict on a new audio clip

```bash
python predict.py --audio path/to/some_clip.wav --model_dir models
```

Output:
```
Predicted emotion: HAPPY

Full probability breakdown:
  happy         71.42%
  surprised     12.05%
  neutral        6.33%
  ...
```

## 5. Try the web demo

```bash
python app.py --model_dir models
```

Open `http://127.0.0.1:5000`, upload a `.wav` file, and see the predicted
emotion with a probability breakdown.

## How it works

1. **Feature extraction (`utils/feature_extraction.py`)** — each audio clip
   is loaded, silence-trimmed, and converted into MFCCs (Mel-Frequency
   Cepstral Coefficients), which capture the shape of the vocal-tract
   frequency response — the acoustic signature most tied to emotional
   tone. Each clip becomes a `(40, 174)` matrix (40 coefficients × time frames).
2. **Model (`utils/model_builder.py`)** — a CNN treats the MFCC matrix like
   an image and learns local spectral-temporal patterns; its output is
   reshaped into a sequence and fed to an LSTM, which models how those
   patterns evolve over the length of the utterance. A dense softmax head
   then classifies into emotion classes.
3. **Training (`train.py`)** — normalizes features, one-hot encodes labels,
   does a stratified 70/15/15 train/val/test split, and trains with early
   stopping + learning-rate decay to avoid overfitting on the modest dataset sizes.
4. **Inference (`predict.py`, `app.py`)** — applies the exact same feature
   extraction + normalization to a new clip and runs it through the saved model.

## Troubleshooting

- **"No audio samples were found/parsed"** — double-check `--dataset` matches
  the naming convention of the files in `--data_dir` (RAVDESS/TESS/EMO-DB
  each encode emotion differently in the filename).
- **Low accuracy** — try more epochs, the `cnn_lstm` architecture, or combine
  multiple datasets (you'll need to run feature extraction dataset-by-dataset
  since label parsing differs, then merge the arrays before training — this
  is a straightforward extension of `train.py` if you want it).
- **Slow training on CPU** — use `--architecture simple_cnn` for a lighter model.
