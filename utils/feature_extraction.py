"""
feature_extraction.py
----------------------
Utility functions for loading speech audio and extracting MFCC
(Mel-Frequency Cepstral Coefficient) features used by the emotion
recognition model.

Works with datasets such as:
  - RAVDESS   (filename encodes emotion, e.g. 03-01-05-01-02-01-12.wav)
  - TESS      (filename contains emotion word, e.g. OAF_back_angry.wav)
  - EMO-DB    (filename encodes emotion with a German letter code)
"""

import os
import re
import numpy as np
import librosa

# ---------------------------------------------------------------------
# Emotion label maps for each supported dataset
# ---------------------------------------------------------------------

# RAVDESS: 3rd field of filename = emotion code
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# EMO-DB: single letter embedded in filename encodes emotion
EMODB_EMOTION_MAP = {
    "W": "angry",
    "L": "boredom",
    "E": "disgust",
    "A": "fearful",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}

# TESS: emotion word appears directly in the filename
TESS_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "ps", "surprise"]


def parse_label_from_filename(filepath, dataset="ravdess"):
    """
    Extract the emotion label from a filename according to the
    naming convention of the given dataset.
    """
    filename = os.path.basename(filepath)
    dataset = dataset.lower()

    if dataset == "ravdess":
        parts = filename.split("-")
        if len(parts) < 3:
            return None
        code = parts[2]
        return RAVDESS_EMOTION_MAP.get(code)

    elif dataset == "emodb":
        # emotion letter is typically the 6th character, e.g. 03a01Fa.wav -> 'F'
        match = re.search(r"[a-zA-Z]{2}\d{2}([A-Za-z])", filename)
        if match:
            return EMODB_EMOTION_MAP.get(match.group(1).upper())
        return None

    elif dataset == "tess":
        name = filename.lower()
        for emo in TESS_EMOTIONS:
            if emo in name:
                return "surprised" if emo == "ps" else emo
        return None

    else:
        raise ValueError(f"Unknown dataset type: {dataset}")


def extract_mfcc(file_path, n_mfcc=40, max_pad_len=174, sr=22050):
    """
    Load an audio file and extract a fixed-size MFCC feature matrix.

    Returns
    -------
    np.ndarray of shape (n_mfcc, max_pad_len)
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
    except Exception as e:
        print(f"[WARN] Could not load {file_path}: {e}")
        return None

    # Trim leading/trailing silence for cleaner features
    audio, _ = librosa.effects.trim(audio)

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)

    # Pad or truncate along the time axis to a fixed length
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc


def extract_features_from_directory(root_dir, dataset="ravdess", n_mfcc=40, max_pad_len=174):
    """
    Walk a dataset directory, extract MFCCs for every .wav file, and
    collect labels parsed from filenames.

    Returns
    -------
    X : np.ndarray, shape (num_samples, n_mfcc, max_pad_len)
    y : list[str] of emotion labels (same length as X)
    """
    X, y = [], []

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith(".wav"):
                continue

            fpath = os.path.join(dirpath, fname)
            label = parse_label_from_filename(fpath, dataset=dataset)
            if label is None:
                continue

            mfcc = extract_mfcc(fpath, n_mfcc=n_mfcc, max_pad_len=max_pad_len)
            if mfcc is None:
                continue

            X.append(mfcc)
            y.append(label)

    X = np.array(X)
    return X, y
