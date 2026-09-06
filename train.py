"""
train.py
--------
End-to-end training pipeline for the Speech Emotion Recognition project.

Usage
-----
    python train.py --data_dir data/RAVDESS --dataset ravdess --epochs 50

This will:
  1. Walk the dataset directory and extract MFCC features from every .wav file
  2. Encode emotion labels
  3. Split into train/validation/test sets
  4. Build and train a CNN+LSTM model
  5. Save the trained model, label encoder, and training curves
"""

import os
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from utils.feature_extraction import extract_features_from_directory
from utils.model_builder import build_cnn_lstm_model, build_simple_cnn_model


def main(args):
    os.makedirs(args.model_dir, exist_ok=True)

    print(f"[INFO] Extracting MFCC features from '{args.data_dir}' "
          f"(dataset type: {args.dataset}) ...")
    X, y = extract_features_from_directory(
        args.data_dir,
        dataset=args.dataset,
        n_mfcc=args.n_mfcc,
        max_pad_len=args.max_pad_len,
    )

    if len(X) == 0:
        raise RuntimeError(
            "No audio samples were found/parsed. Check --data_dir and --dataset "
            "match the folder structure and filename convention of your dataset."
        )

    print(f"[INFO] Extracted {len(X)} samples across {len(set(y))} emotion classes: {sorted(set(y))}")

    # Normalize features (zero mean, unit variance) computed over the whole dataset
    mean, std = X.mean(), X.std()
    X = (X - mean) / (std + 1e-8)

    # Add channel dimension -> (samples, n_mfcc, time, 1)
    X = X[..., np.newaxis]

    # Encode string labels -> integers -> one-hot
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_onehot = to_categorical(y_encoded)

    # Train / val / test split (70 / 15 / 15).
    # Stratification keeps class balance across splits, but fails if any
    # class has too few samples for the requested split sizes — fall back
    # to a plain random split in that case (this mainly affects tiny/toy
    # datasets; real datasets like RAVDESS/TESS/EMO-DB have plenty per class).
    def safe_split(X_in, y_in, test_size):
        try:
            return train_test_split(X_in, y_in, test_size=test_size, random_state=42, stratify=y_in)
        except ValueError as e:
            print(f"[WARN] Stratified split failed ({e}); falling back to a random split.")
            return train_test_split(X_in, y_in, test_size=test_size, random_state=42)

    X_train, X_temp, y_train, y_temp = safe_split(X, y_onehot, test_size=0.3)
    X_val, X_test, y_val, y_test = safe_split(X_temp, y_temp, test_size=0.5)

    print(f"[INFO] Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

    input_shape = X_train.shape[1:]
    num_classes = y_onehot.shape[1]

    if args.architecture == "cnn_lstm":
        model = build_cnn_lstm_model(input_shape, num_classes)
    else:
        model = build_simple_cnn_model(input_shape, num_classes)

    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(
            os.path.join(args.model_dir, "best_model.h5"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # --- Evaluate on held-out test set ---
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[RESULT] Test accuracy: {test_acc * 100:.2f}%  |  Test loss: {test_loss:.4f}")

    # --- Save artifacts needed for inference ---
    model.save(os.path.join(args.model_dir, "emotion_model.h5"))

    with open(os.path.join(args.model_dir, "label_encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    with open(os.path.join(args.model_dir, "norm_stats.pkl"), "wb") as f:
        pickle.dump({"mean": mean, "std": std, "n_mfcc": args.n_mfcc, "max_pad_len": args.max_pad_len}, f)

    # --- Plot training curves ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(args.model_dir, "training_curves.png"))
    print(f"[INFO] Saved model + encoder + training curves to '{args.model_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a speech emotion recognition model.")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Root directory containing the dataset's .wav files (can be nested in subfolders).")
    parser.add_argument("--dataset", type=str, default="ravdess", choices=["ravdess", "tess", "emodb"],
                         help="Which dataset naming convention to parse labels from.")
    parser.add_argument("--model_dir", type=str, default="models",
                         help="Directory to save the trained model and related artifacts.")
    parser.add_argument("--architecture", type=str, default="cnn_lstm", choices=["cnn_lstm", "simple_cnn"])
    parser.add_argument("--n_mfcc", type=int, default=40)
    parser.add_argument("--max_pad_len", type=int, default=174)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    main(args)
