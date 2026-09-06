"""
predict.py
----------
Run inference on a single .wav file using the trained emotion
recognition model.

Usage
-----
    python predict.py --audio path/to/sample.wav --model_dir models
"""

import os
import argparse
import pickle
import numpy as np
from tensorflow.keras.models import load_model

from utils.feature_extraction import extract_mfcc


def load_artifacts(model_dir):
    model = load_model(os.path.join(model_dir, "emotion_model.h5"))

    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)

    with open(os.path.join(model_dir, "norm_stats.pkl"), "rb") as f:
        norm_stats = pickle.load(f)

    return model, encoder, norm_stats


def predict_emotion(audio_path, model, encoder, norm_stats):
    mfcc = extract_mfcc(
        audio_path,
        n_mfcc=norm_stats["n_mfcc"],
        max_pad_len=norm_stats["max_pad_len"],
    )
    if mfcc is None:
        raise ValueError(f"Could not process audio file: {audio_path}")

    mfcc = (mfcc - norm_stats["mean"]) / (norm_stats["std"] + 1e-8)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # (1, n_mfcc, time, 1)

    probs = model.predict(mfcc, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = encoder.inverse_transform([pred_idx])[0]

    # Build a sorted dict of {emotion: probability} for the full breakdown
    all_probs = {
        encoder.inverse_transform([i])[0]: float(probs[i])
        for i in range(len(probs))
    }
    all_probs = dict(sorted(all_probs.items(), key=lambda kv: kv[1], reverse=True))

    return pred_label, all_probs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict emotion from a speech audio file.")
    parser.add_argument("--audio", type=str, required=True, help="Path to a .wav file.")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory with trained model artifacts.")
    args = parser.parse_args()

    model, encoder, norm_stats = load_artifacts(args.model_dir)
    label, probs = predict_emotion(args.audio, model, encoder, norm_stats)

    print(f"\nPredicted emotion: {label.upper()}\n")
    print("Full probability breakdown:")
    for emo, p in probs.items():
        print(f"  {emo:<12s} {p * 100:5.2f}%")
