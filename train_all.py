"""
train_all.py
------------
Trains a per-user KeystrokeLSTM verifier for each subject listed in
SUBJECTS_TO_TRAIN, and saves the model weights + fitted scaler to
the models/ folder.

What this produces for each subject (e.g. s002):
    models/s002_model.pt   -> PyTorch state_dict
    models/s002_scaler.pkl -> Fitted StandardScaler
    models/s002_meta.json  -> FAR/FRR at several thresholds + held-out
                              test rows (used by the Streamlit app)

Why this script exists:
    The notebook trains a single model in-memory and throws it away
    when the kernel dies. To deploy on Streamlit, we need the trained
    artefacts on disk so the app can load them without retraining.

Run:
    python train_all.py
"""

import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import KeystrokeLSTM, build_genuine_impostor_split, reshape_for_lstm

# -------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
DATA_PATH = "DSL-StrongPasswordData.csv"
MODEL_DIR = "models"
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
IMPOSTOR_SAMPLE = 200

# Subjects you want available in the Streamlit demo. Pick any 5–10
# from the 51 in the CMU dataset (s002 to s057, with gaps).
SUBJECTS_TO_TRAIN = ["s002", "s003", "s004", "s005", "s007", "s008", "s010", "s011"]

# Thresholds reported in the per-subject meta file
THRESHOLDS = [0.3, 0.5, 0.7, 0.9]


def train_one_subject(df: pd.DataFrame, subject: str) -> dict:
    """
    Train, evaluate, and persist one per-user model.

    Returns a small dict of evaluation metrics so the caller can print
    a summary across all trained subjects.
    """
    print(f"\n=== Training verifier for {subject} ===")

    # Build the genuine/impostor dataset for this user
    X, y = build_genuine_impostor_split(df, target_subject=subject,
                                        impostor_sample=IMPOSTOR_SAMPLE, seed=SEED)

    # Stratified train/test split — preserve genuine:impostor ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Fit scaler on TRAIN ONLY, apply to test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape for LSTM
    X_train_seq = reshape_for_lstm(X_train)
    X_test_seq = reshape_for_lstm(X_test)

    # Tensors and DataLoaders
    X_train_t = torch.from_numpy(X_train_seq).float()
    X_test_t = torch.from_numpy(X_test_seq).float()
    y_train_t = torch.from_numpy(y_train).float().unsqueeze(1)
    y_test_t = torch.from_numpy(y_test).float().unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=BATCH_SIZE, shuffle=True)

    # Model, loss, optimizer
    model = KeystrokeLSTM(input_size=3, hidden_size=32)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:>2}/{EPOCHS}  loss={total_loss / len(train_loader):.4f}")

    # Evaluation on the held-out test set
    model.eval()
    with torch.no_grad():
        probs = model(X_test_t).numpy().ravel()
    labels = y_test_t.numpy().ravel()

    # FAR / FRR at multiple thresholds
    far_frr = {}
    for thr in THRESHOLDS:
        preds = (probs > thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        far = fp / (fp + tn) if (fp + tn) else 0.0   # impostor accepted
        frr = fn / (fn + tp) if (fn + tp) else 0.0   # genuine rejected
        far_frr[str(thr)] = {"FAR": float(far), "FRR": float(frr)}

    # Persist artefacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{subject}_model.pt"))
    with open(os.path.join(MODEL_DIR, f"{subject}_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Save a small bundle of the held-out test rows. The Streamlit app
    # uses this to show "pick a genuine sample" / "pick an impostor
    # sample" without having to ship the full CSV with every demo.
    meta = {
        "subject": subject,
        "far_frr": far_frr,
        "test_rows_unscaled": X_test_seq.tolist(),     # already reshaped (n, 10, 3)
        "test_labels": labels.tolist(),
        "test_probs_at_train_time": probs.tolist(),    # sanity-check the loaded model later
    }
    # NB: test_rows_unscaled is misnamed — these rows ARE scaled. We
    # save the scaled tensor because that's what the model expects.
    with open(os.path.join(MODEL_DIR, f"{subject}_meta.json"), "w") as f:
        json.dump(meta, f)

    return {"subject": subject, "far_frr": far_frr}


def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            f"Could not find {DATA_PATH}. Download the CMU Keystroke Dataset and "
            f"place DSL-StrongPasswordData.csv in the project root."
        )

    df = pd.read_csv(DATA_PATH)

    # Drop H.Return so we get 30 features cleanly reshapeable to (10, 3)
    if "H.Return" in df.columns:
        df = df.drop("H.Return", axis=1)

    summary = []
    for subj in SUBJECTS_TO_TRAIN:
        if subj not in df["subject"].unique():
            print(f"  [skip] {subj} not in dataset")
            continue
        summary.append(train_one_subject(df, subj))

    # Final summary table
    print("\n=== Summary (FAR / FRR at threshold = 0.7) ===")
    print(f"{'Subject':<8} {'FAR':>8} {'FRR':>8}")
    for row in summary:
        ff = row["far_frr"]["0.7"]
        print(f"{row['subject']:<8} {ff['FAR']:>7.2%} {ff['FRR']:>7.2%}")


if __name__ == "__main__":
    main()
