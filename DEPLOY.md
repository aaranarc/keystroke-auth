# Deployment guide — GitHub + Streamlit Cloud

This walks through everything from "files on your laptop" to "live URL on the internet".

---

## Stage 0 — Verify locally first (do NOT skip)

Before pushing anything, make sure the full pipeline works on your machine.

```bash
# From inside the keystroke-auth/ folder
python -m venv .venv
source .venv/bin/activate         # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Make sure DSL-StrongPasswordData.csv is in the project root
ls DSL-StrongPasswordData.csv

# Train per-subject verifiers
python train_all.py
# -> populates models/ with 8 sets of _model.pt + _scaler.pkl + _meta.json

# Launch the demo
streamlit run app.py
# -> opens http://localhost:8501
```

If `streamlit run app.py` works end-to-end and you can flip between subjects and samples, you're ready to ship. **If it errors locally, it will also error on Streamlit Cloud — fix it now.**

---

## Stage 1 — Put it on GitHub

### 1.1  Create the repo on GitHub.com

1. Go to https://github.com → top-right `+` → **New repository**
2. Repository name: `keystroke-auth` (or any name you like)
3. Description: *"Behavioural biometric authentication using LSTM on the CMU Keystroke Dataset"*
4. Visibility: **Public** (required for free Streamlit Cloud)
5. Do **not** tick "Add a README" / "Add .gitignore" / "Choose a license" — we already have all three locally
6. Click **Create repository**

GitHub will show a "quick setup" page with a URL like `https://github.com/YOUR-USERNAME/keystroke-auth.git`. Copy it.

### 1.2  Initialise and push from your machine

```bash
cd keystroke-auth

# Initialise the repo
git init
git branch -M main

# Stage everything (.gitignore excludes the right things)
git add .

# First commit
git commit -m "Initial commit: keystroke dynamics LSTM authentication"

# Connect to GitHub and push
git remote add origin https://github.com/YOUR-USERNAME/keystroke-auth.git
git push -u origin main
```

If `git push` asks for a password, GitHub no longer accepts your account password over HTTPS — you need a **Personal Access Token**:
- GitHub → your profile picture → **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)** → **Generate new token (classic)**
- Scope: tick `repo`
- Copy the token (you only see it once) and paste it when `git push` asks for a password.

After the push, refresh the GitHub repo page — you should see all the files.

### 1.3  What gets committed and what doesn't

| Committed | Not committed |
|---|---|
| `model.py`, `train_all.py`, `app.py` | `.venv/`, `__pycache__/` |
| `requirements.txt`, `README.md`, `LICENSE`, `.gitignore` | (handled by `.gitignore`) |
| `models/*.pt`, `models/*.pkl`, `models/*.json`  *(few hundred KB total — fine for git)* | |
| `DSL-StrongPasswordData.csv` *(~600 KB — also fine for git)* | |
| `notebooks/keystroke_lstm_project.ipynb` | |

**Do not** commit datasets larger than ~25 MB. GitHub will warn you, and pushing beyond 100 MB will fail outright. The CMU CSV is small enough to be safe.

---

## Stage 2 — Deploy on Streamlit Community Cloud

### 2.1  Sign up

1. Go to https://streamlit.io/cloud
2. Click **Sign up** → **Continue with GitHub**
3. Authorise Streamlit to read your public repos

### 2.2  Deploy the app

1. Click **New app** (top right)
2. Fill in:
   - **Repository**: `YOUR-USERNAME/keystroke-auth`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL** (optional): pick a custom subdomain like `keystroke-auth-raj`
3. Click **Deploy**

Streamlit will:
- Clone your repo
- `pip install -r requirements.txt` (this takes ~3–5 min — PyTorch is a big install)
- Run `streamlit run app.py`

Watch the deployment logs. If the build fails, the log will tell you why — usually a missing dependency or a path issue. Fix locally, `git push`, and Streamlit auto-redeploys.

### 2.3  Once live

You'll get a URL like `https://keystroke-auth-raj.streamlit.app`. Test it:
- Switch subjects, change the threshold, flip between genuine and impostor samples
- Confirm the FAR/FRR curve renders
- Try on mobile too — Streamlit is responsive by default

### 2.4  Updating

Any `git push` to `main` triggers an automatic redeploy within ~30 seconds. No separate Streamlit action needed.

---

## Stage 3 — Polish for the placement portfolio

After it's live, do this:

1. **Put the live URL in the README** (replace the placeholder)
2. **Add a screenshot or short GIF** to the README — recruiters skim
3. **Pin the repo** on your GitHub profile (Profile → Customize your pins)
4. **Add to your CV / LinkedIn**:
   > *Keystroke Dynamics Authentication* — Built a behavioural-biometric verification system in PyTorch (per-user LSTM) on the CMU Keystroke Benchmark; deployed an interactive demo on Streamlit Cloud. Reports FAR/FRR trade-off across decision thresholds. Live: [URL]
5. **Open one or two GitHub issues yourself** describing planned enhancements (e.g., "Add live keystroke capture", "One-class SVM baseline") — shows the project is alive rather than abandoned.

---

## Common failure modes and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` on Streamlit Cloud | `torch` missing from `requirements.txt` | Add it, push |
| Build hangs >10 min installing PyTorch | Streamlit Cloud is rebuilding the wheel cache; subsequent deploys are faster | Wait it out the first time |
| `FileNotFoundError: models/s002_model.pt` | You didn't commit the `models/` folder | `git add models/ && git commit && git push` |
| `RuntimeError: Error(s) in loading state_dict` | `KeystrokeLSTM` definition in `model.py` changed after training | Retrain (`python train_all.py`) and re-commit `models/` |
| App loads but is blank | Check the right sidebar → **Manage app** → logs for the actual error | Usually a Python exception inside the script |
| Push rejected — "file too large" | A `.pt`, `.pkl`, or CSV exceeded 100 MB | Use Git LFS, or remove and download at runtime |
