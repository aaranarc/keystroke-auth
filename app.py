"""
app.py
------
Streamlit demo for the Keystroke Dynamics Authentication system.

How the demo works (v1, file-driven):
    1. User picks a TARGET subject from the sidebar — this is the
       identity we're verifying against.
    2. User picks a TEST SAMPLE — either a genuine sample from the
       target, or a random impostor sample.
    3. The app loads the matching saved model + scaler, runs the
       sample through the network, and shows:
          - the raw probability,
          - the verdict at the chosen threshold,
          - the FAR/FRR curves for this subject's model,
          - a confusion-matrix view on the full held-out test set.

This is the simplest deployable UX. A v2 would add a live keystroke
capture component (streamlit-keyup) so the user could type the
password '.tie5Roanl' in-browser. That requires a JS-level component
and is intentionally out of scope here.
"""

import json
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics import confusion_matrix

from model import KeystrokeLSTM

# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Keystroke Dynamics Authentication",
    page_icon="⌨️",
    layout="wide",
)

MODEL_DIR = "models"


# -------------------------------------------------------------------
# Cached loaders — Streamlit reruns the script on every interaction,
# so we cache the heavy work (loading the model, scaler, test rows)
# -------------------------------------------------------------------
@st.cache_data
def available_subjects() -> list:
    """Return list of subjects we have trained models for."""
    if not os.path.isdir(MODEL_DIR):
        return []
    subjects = sorted({
        f.split("_")[0]
        for f in os.listdir(MODEL_DIR)
        if f.endswith("_model.pt")
    })
    return subjects


@st.cache_resource
def load_model(subject: str):
    """Load the per-user LSTM weights into a model instance."""
    model = KeystrokeLSTM(input_size=3, hidden_size=32)
    state = torch.load(
        os.path.join(MODEL_DIR, f"{subject}_model.pt"),
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()
    return model


@st.cache_resource
def load_scaler(subject: str):
    with open(os.path.join(MODEL_DIR, f"{subject}_scaler.pkl"), "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_meta(subject: str) -> dict:
    with open(os.path.join(MODEL_DIR, f"{subject}_meta.json"), "r") as f:
        return json.load(f)


# -------------------------------------------------------------------
# Top-level checks
# -------------------------------------------------------------------
subjects = available_subjects()
if not subjects:
    st.error(
        "No trained models found in `models/`. "
        "Run `python train_all.py` first to generate them."
    )
    st.stop()

# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tab_browser, tab_login = st.tabs(["Sample Browser", "Demo Login"])


# ===================================================================
# TAB 1 — Sample Browser  (existing functionality, unchanged)
# ===================================================================
with tab_browser:
    # -------------------------------------------------------------------
    # Sidebar — controls
    # -------------------------------------------------------------------
    st.sidebar.title("⌨️  Controls")

    target = st.sidebar.selectbox(
        "Target subject (identity being verified)",
        subjects,
        index=0,
        help="Each subject has its own LSTM verifier — the model accepts "
             "this user and rejects everyone else.",
    )

    threshold = st.sidebar.slider(
        "Decision threshold",
        min_value=0.10, max_value=0.95, value=0.70, step=0.05,
        help="Probability cut-off above which we ACCEPT. Higher threshold = "
             "fewer false accepts (lower FAR) but more false rejects (higher FRR).",
    )

    sample_source = st.sidebar.radio(
        "Test sample source",
        ["Genuine (target's own typing)", "Impostor (someone else)"],
        help="Pick whose keystroke timings to feed into the model.",
    )

    # -------------------------------------------------------------------
    # Header
    # -------------------------------------------------------------------
    st.title("Keystroke Dynamics Authentication")
    st.markdown(
        "Behavioural biometric verification using an LSTM trained on the "
        "**CMU Keystroke Dynamics Benchmark Dataset** (Killourhy & Maynard). "
        "Each user has a personal verifier model that learns their unique "
        "typing rhythm for the password `.tie5Roanl`."
    )

    # -------------------------------------------------------------------
    # Load the chosen subject's artefacts and test rows
    # -------------------------------------------------------------------
    model = load_model(target)
    scaler = load_scaler(target)
    meta = load_meta(target)

    test_rows = np.array(meta["test_rows_unscaled"])   # (n, 10, 3) already scaled
    test_labels = np.array(meta["test_labels"])

    genuine_indices = np.where(test_labels == 1)[0]
    impostor_indices = np.where(test_labels == 0)[0]

    # Pick a sample to evaluate
    if "Genuine" in sample_source:
        pool = genuine_indices
        pool_label = "Genuine samples in held-out test set"
    else:
        pool = impostor_indices
        pool_label = "Impostor samples in held-out test set"

    sample_idx_pos = st.sidebar.number_input(
        f"Sample # (0 to {len(pool) - 1})",
        min_value=0, max_value=max(0, len(pool) - 1), value=0,
    )
    sample_idx = int(pool[sample_idx_pos])
    sample_row = test_rows[sample_idx]                  # (10, 3) — already scaled
    true_label = int(test_labels[sample_idx])

    # -------------------------------------------------------------------
    # Main layout — two columns
    # -------------------------------------------------------------------
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Sample being tested")
        st.caption(pool_label)

        sample_df = pd.DataFrame(
            sample_row,
            columns=["H (hold)", "DD (down-down)", "UD (up-down / flight)"],
            index=[f"transition {i + 1}" for i in range(sample_row.shape[0])],
        )
        st.dataframe(sample_df.style.format("{:+.3f}"), use_container_width=True)

        # Bar chart of the three timing channels across the 10 transitions
        fig_in, ax_in = plt.subplots(figsize=(6, 3))
        width = 0.27
        x_pos = np.arange(sample_row.shape[0])
        ax_in.bar(x_pos - width, sample_row[:, 0], width, label="H")
        ax_in.bar(x_pos, sample_row[:, 1], width, label="DD")
        ax_in.bar(x_pos + width, sample_row[:, 2], width, label="UD")
        ax_in.set_xlabel("Keystroke transition")
        ax_in.set_ylabel("Scaled timing")
        ax_in.set_title("Timing fingerprint")
        ax_in.legend()
        ax_in.grid(axis="y", alpha=0.3)
        st.pyplot(fig_in, clear_figure=True)

    with col_right:
        st.subheader("Verifier output")

        with torch.no_grad():
            x = torch.from_numpy(sample_row).float().unsqueeze(0)   # (1, 10, 3)
            prob = float(model(x).item())

        verdict_accept = prob > threshold
        verdict_text = "✅  ACCEPT — claim verified" if verdict_accept else "❌  REJECT — identity not verified"
        truth_text = "(this sample was actually GENUINE)" if true_label == 1 else "(this sample was actually IMPOSTOR)"

        st.metric("Genuine-probability score", f"{prob:.4f}")
        st.metric("Threshold", f"{threshold:.2f}")
        st.markdown(f"### {verdict_text}")
        st.caption(truth_text)

        # Outcome interpretation
        if verdict_accept and true_label == 1:
            st.success("True Accept — correct decision.")
        elif verdict_accept and true_label == 0:
            st.error("False Accept (security failure) — impostor was let through.")
        elif not verdict_accept and true_label == 1:
            st.warning("False Reject (usability failure) — genuine user was blocked.")
        else:
            st.success("True Reject — correct decision.")

    # -------------------------------------------------------------------
    # Bottom row — full-test-set metrics
    # -------------------------------------------------------------------
    st.markdown("---")
    st.subheader(f"Verifier performance for {target}  (full held-out test set)")

    # Predictions on the entire test set at the current threshold
    with torch.no_grad():
        x_all = torch.from_numpy(test_rows).float()
        probs_all = model(x_all).numpy().ravel()
    preds_all = (probs_all > threshold).astype(int)
    cm = confusion_matrix(test_labels, preds_all, labels=[0, 1])

    m1, m2 = st.columns(2)

    with m1:
        st.markdown("**Confusion matrix at current threshold**")
        cm_df = pd.DataFrame(
            cm,
            index=["actual: impostor", "actual: genuine"],
            columns=["pred: impostor", "pred: genuine"],
        )
        st.dataframe(cm_df, use_container_width=True)

        tn, fp, fn, tp = cm.ravel()
        far_now = fp / (fp + tn) if (fp + tn) else 0.0
        frr_now = fn / (fn + tp) if (fn + tp) else 0.0
        st.write(f"**FAR** (impostor accepted): `{far_now:.2%}`")
        st.write(f"**FRR** (genuine rejected): `{frr_now:.2%}`")

    with m2:
        st.markdown("**FAR / FRR across thresholds**")
        grid_thresholds = np.linspace(0.05, 0.95, 19)
        far_curve, frr_curve = [], []
        for thr in grid_thresholds:
            p = (probs_all > thr).astype(int)
            c = confusion_matrix(test_labels, p, labels=[0, 1]).ravel()
            ttn, tfp, tfn, ttp = c
            far_curve.append(tfp / (tfp + ttn) if (tfp + ttn) else 0.0)
            frr_curve.append(tfn / (tfn + ttp) if (tfn + ttp) else 0.0)

        fig_curve, ax_curve = plt.subplots(figsize=(6, 3.5))
        ax_curve.plot(grid_thresholds, far_curve, label="FAR (impostor accepted)", linewidth=2)
        ax_curve.plot(grid_thresholds, frr_curve, label="FRR (genuine rejected)", linewidth=2)
        ax_curve.axvline(threshold, color="grey", linestyle="--", label=f"current = {threshold:.2f}")
        ax_curve.set_xlabel("Threshold")
        ax_curve.set_ylabel("Error rate")
        ax_curve.set_title(f"Error trade-off — {target}")
        ax_curve.legend()
        ax_curve.grid(alpha=0.3)
        st.pyplot(fig_curve, clear_figure=True)

    # -------------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------------
    st.markdown("---")
    st.caption(
        "Architecture: single-layer LSTM (hidden=32) + dropout(0.2) + linear + sigmoid. "
        "Loss: BCE.  Optimizer: Adam (lr=1e-3).  "
        "Trained per-subject with 200 random impostor samples and stratified 80/20 split."
    )


# ===================================================================
# TAB 2 — Demo Login
# ===================================================================
with tab_login:
    st.title("🔐 Keystroke Login Demo")
    st.info(
        "**Simulated login** — This demo uses pre-recorded test samples from "
        "the CMU Keystroke Dynamics Benchmark Dataset. The typing in the "
        "password box below is purely cosmetic; verification runs against a "
        "randomly-picked test sample from the claimed user's held-out data."
    )

    # --- Login form layout ---
    login_col, spacer, info_col = st.columns([2, 0.5, 2])

    with login_col:
        st.subheader("Sign In")

        claimed_subject = st.selectbox(
            "Claimed identity",
            subjects,
            index=0,
            key="login_subject",
        )

        st.text_input(
            "Password",
            type="password",
            placeholder=".tie5Roanl",
            key="login_password",
        )

        simulate_impostor = st.toggle(
            "Simulate impostor attempt",
            value=False,
            key="login_impostor_toggle",
            help="When ON, the verifier is fed an impostor sample instead of "
                 "a genuine one.",
        )

        login_clicked = st.button("🚪 Login", use_container_width=True, type="primary")

    with info_col:
        st.subheader("How it works")
        st.markdown(
            """
            1. **Pick a subject** — each has a dedicated LSTM verifier.
            2. **Type anything** in the password field (it's cosmetic).
            3. **Toggle impostor mode** to see how the model handles
               a different person's typing rhythm.
            4. **Click Login** — a random test sample is pulled from the
               held-out set and run through the model.

            **Threshold**: `0.70` (fixed for this demo)
            """
        )

    # --- Login logic ---
    if login_clicked:
        with st.spinner("Analysing typing rhythm..."):
            time.sleep(1)

        # Reuse cached loaders
        login_model = load_model(claimed_subject)
        login_meta = load_meta(claimed_subject)

        login_test_rows = np.array(login_meta["test_rows_unscaled"])
        login_test_labels = np.array(login_meta["test_labels"])

        # Pick a random sample based on toggle
        if simulate_impostor:
            candidate_indices = np.where(login_test_labels == 0)[0]
            ground_truth_label = 0
            ground_truth_text = "Impostor"
        else:
            candidate_indices = np.where(login_test_labels == 1)[0]
            ground_truth_label = 1
            ground_truth_text = "Genuine"

        chosen_idx = random.choice(candidate_indices)
        chosen_sample = login_test_rows[chosen_idx]  # (10, 3)

        # Run through model
        with torch.no_grad():
            x_login = torch.from_numpy(chosen_sample).float().unsqueeze(0)
            login_prob = float(login_model(x_login).item())

        login_threshold = 0.70
        access_granted = login_prob > login_threshold

        # --- Verdict ---
        st.markdown("---")

        if access_granted:
            st.success(f"✅ ACCESS GRANTED — Welcome back, {claimed_subject}!")
            st.balloons()
        else:
            st.error("🚫 ACCESS DENIED — Identity could not be verified")

        # --- Supporting details ---
        st.markdown("#### Verification Details")
        detail_cols = st.columns(4)

        with detail_cols[0]:
            st.metric("Probability", f"{login_prob:.4f}")

        with detail_cols[1]:
            st.metric("Threshold", f"{login_threshold:.2f}")

        with detail_cols[2]:
            st.metric("Ground Truth", ground_truth_text)

        with detail_cols[3]:
            # Determine outcome classification
            if access_granted and ground_truth_label == 1:
                outcome = "True Accept ✅"
            elif access_granted and ground_truth_label == 0:
                outcome = "False Accept ⚠️"
            elif not access_granted and ground_truth_label == 1:
                outcome = "False Reject ⚠️"
            else:
                outcome = "True Reject ✅"
            st.metric("Outcome", outcome)
