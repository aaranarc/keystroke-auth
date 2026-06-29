# Keystroke Dynamics Authentication

A behavioural-biometric continuous-authentication system. An LSTM learns the typing rhythm of each user from the CMU Keystroke Dynamics Benchmark Dataset and decides whether a new typing sample belongs to the claimed user or an impostor.

**[➡  Live demo](https://keystroke-app.streamlit.app/)** 
---

## What's in here

| File | Purpose |
|---|---|
| `model.py` | `KeystrokeLSTM` class + preprocessing utilities — imported by both the training script and the Streamlit app |
| `train_all.py` | Trains one verifier per subject; saves model weights, scaler, and held-out test rows to `models/` |
| `app.py` | Streamlit app — pick a user, pick a sample, see the verdict + FAR/FRR curves |
| `notebooks/keystroke_lstm_project.ipynb` | Original exploratory notebook |
| `requirements.txt` | Pinned dependencies for Streamlit Cloud |

## Dataset

CMU Keystroke Dynamics Benchmark — Killourhy & Maynard (2009). 51 subjects, each typing the password `.tie5Roanl` 400 times across 8 sessions. Each row provides three timing features per keystroke: **H** (hold), **DD** (down-down), **UD** (up-down / flight).

Download `DSL-StrongPasswordData.csv` from the [CMU site](https://www.cs.cmu.edu/~keystroke/) and place it in the project root before running `train_all.py`.

## Architecture

```
Input  (batch, seq_len=10, features=3)
   |
   v
LSTM (hidden=32, batch_first=True)
   |
   v
Dropout (p=0.2)
   |
   v
Linear (32 -> 1)
   |
   v
Sigmoid
   |
   v
Output (batch, 1)  -> probability sample is genuine
```

The 30 raw timing features (after dropping `H.Return`) are reshaped into 10 transitions × 3 features per transition so the LSTM can model the password as a temporal sequence rather than a flat vector.

## Per-user verifier design

Identity verification is a **per-user binary classification** problem:

- Class 1 (genuine): all 400 samples from the target user
- Class 0 (impostor): 200 randomly sampled rows from the other 50 users

One LSTM is trained per user. Production deployment would store one model per enrolled user; this demo ships verifiers for 8 subjects.

## Run it locally

```bash
# 1. Clone
git clone https://github.com/YOUR-USERNAME/keystroke-auth.git
cd keystroke-auth

# 2. Install
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Download DSL-StrongPasswordData.csv into the project root

# 4. Train (creates models/<subject>_model.pt, _scaler.pkl, _meta.json)
python train_all.py

# 5. Launch the demo
streamlit run app.py
```

## Evaluation metrics

The app reports both error rates a verifier is judged on:

- **FAR** — False Acceptance Rate — fraction of impostors mistakenly accepted (security failure)
- **FRR** — False Rejection Rate — fraction of genuine users mistakenly rejected (usability failure)

These trade off against each other through the decision threshold. The app's threshold slider lets you see the trade-off live, and the FAR/FRR curve plot shows the full sweep.

## Possible extensions

- **Live keystroke capture** in the browser via a `streamlit-keyup` component, so users can type the password themselves rather than picking saved samples.
- **One-class formulation** (one-class SVM or autoencoder) — better suited to the data-starved regime (~400 samples per user).
- **Triplet / Siamese network** — learn an embedding space where same-user samples cluster, removing the need for one model per user.
- **Cross-session evaluation** — train on early sessions, test on later ones, to measure typing-drift robustness.

## Author

[Aarana Chaurasia]

## License

MIT — see `LICENSE`.
