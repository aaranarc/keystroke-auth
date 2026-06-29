"""
model.py
--------
Defines the KeystrokeLSTM model and shared preprocessing utilities.

Why this file is separate from the notebook:
    Streamlit (app.py) and the training script (train_all.py) both need
    access to the SAME model class and the SAME preprocessing logic.
    If we keep the class definition only inside the notebook, neither
    can import it. Factoring it out into model.py is what makes the
    project deployable.
"""

import numpy as np
import torch
import torch.nn as nn


# -------------------------------------------------------------------
# Model definition
# -------------------------------------------------------------------
class KeystrokeLSTM(nn.Module):
    """
    Single-layer LSTM followed by dropout and a linear classifier head.

    Input shape : (batch, seq_len=10, features=3)
        - 10 keystroke transitions
        - 3 timing features per transition: H, DD, UD

    Output shape: (batch, 1)
        - Sigmoid probability that the sample is genuine (label = 1)
    """

    def __init__(self, input_size: int = 3, hidden_size: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # lstm_out: (batch, seq_len, hidden)
        # h_n:      (num_layers=1, batch, hidden)
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]            # (batch, hidden)
        dropped = self.dropout(last_hidden)
        logits = self.fc(dropped)        # (batch, 1)
        return torch.sigmoid(logits)


# -------------------------------------------------------------------
# Preprocessing helpers (used identically by training and inference)
# -------------------------------------------------------------------
# The CMU dataset has 31 timing columns. We drop H.Return so we are
# left with 30 features, which reshape cleanly into (10 transitions
# x 3 features per transition).
DROP_COLUMN = "H.Return"
META_COLUMNS = ["subject", "sessionIndex", "rep"]


def feature_columns(df) -> list:
    """Return the 30 feature columns in the order the model expects."""
    return [c for c in df.columns if c not in META_COLUMNS + [DROP_COLUMN]]


def reshape_for_lstm(x: np.ndarray) -> np.ndarray:
    """
    Reshape (n_samples, 30) into (n_samples, 10, 3) for the LSTM.

    The 30 flat features are grouped into 10 time steps of 3 features
    each — this is the same reshape used during training and MUST be
    applied identically at inference time.
    """
    return x.reshape(x.shape[0], 10, 3)


def build_genuine_impostor_split(df, target_subject: str, impostor_sample: int = 200, seed: int = 42):
    """
    For a chosen target subject, return:
        - X: (n, 30) feature matrix containing genuine + impostor rows
        - y: (n,) labels — 1 for the target subject, 0 for others

    Parameters
    ----------
    df : pandas.DataFrame
        The full CMU dataset.
    target_subject : str
        e.g. 's002' — the user whose identity we are training to verify.
    impostor_sample : int
        How many random rows from OTHER subjects to use as impostors.
    seed : int
        For reproducibility of the impostor sample.
    """
    feats = feature_columns(df)

    genuine = df[df["subject"] == target_subject]
    impostor = df[df["subject"] != target_subject].sample(
        n=impostor_sample, random_state=seed
    )

    X = np.vstack([genuine[feats].values, impostor[feats].values])
    y = np.concatenate([np.ones(len(genuine)), np.zeros(len(impostor))])
    return X, y
