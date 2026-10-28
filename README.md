# 🧠 NEUROENDO
### Emotion cues · Adaptive learning · Artificial endocrine system

<p align="center">
  <strong>A biologically inspired, non-quantum AI prototype that explores how internal state variables can modulate learning and responses.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  <img alt="Learning" src="https://img.shields.io/badge/Learning-Hebbian-BA477D?style=for-the-badge">
  <img alt="Approach" src="https://img.shields.io/badge/Approach-Classical%20%7C%20Non--Quantum-7D6BB3?style=for-the-badge">
</p>

<p align="center">
  <em>From signals to associations — with feedback shaping how the system learns.</em>
</p>

---

## ✨ What is NEUROENDO?

NEUROENDO is an AIML project prototype inspired by the idea that learning may be influenced by an internal regulatory system. It brings together **visual cues, spoken language, simple audio features, Hebbian association updates, and a simulated artificial endocrine controller** in one interactive Streamlit app.

The goal is to make the learning process visible and explainable: explore input signals, observe a response, give feedback, and inspect how the association weights change.

> **Important:** NEUROENDO is an educational demonstrator, not a validated emotion detector, mental-health tool, or model of human biology. It uses simplified engineering rules and pretrained models.

## 🌸 Explore the app

| Workspace | What you can do |
|---|---|
| **Emotion Studio** | Explore demo signals or capture a still image and a short audio clip, view the available emotion cues, and teach the model from feedback. |
| **Model Insights** | Inspect the Hebbian weight matrix, animated feature-to-response network, learning history, and reward/stress trends. |
| **A/B Experiment** | Compare ordinary Hebbian updates with endocrine-modulated updates on the same reproducible synthetic dataset. |

The interface uses a soft pink-and-peach visual theme, separate workspace navigation, responsive cards, and visual summaries to make the project easier to demonstrate.

## 🧩 How it works

```text
Camera + microphone / simulated demo signals
                    │
                    ▼
      Classical feature extraction
       ├─ Visual cue from image
       ├─ Acoustic energy / texture
       └─ Speech transcription → text emotion cues
                    │
                    ▼
       Feature vector x = [V, A, U]
                    │
                    ▼
       Hebbian association network  ◄──── Feedback target y
                    ▲                           │
                    │                           ▼
       Endocrine modulation g(H)  ◄──── Reward D / Stress S
                    │
                    ▼
        Response + analytics dashboard
```

### The learning idea

The network uses a small association matrix to connect three input features to three response categories: **Supportive**, **Neutral**, and **Upbeat**.

Hebbian-style update with simulated endocrine modulation:

```text
ΔW = η · g(H) · outer(x, y)
g(H) = clip(1 + D − S, 0.2, 1.5)
```

The reward-like state (D) and stress-like state (S) are simple controller variables. They are not actual hormone measurements.

```text
D_next = 0.9 · D + 0.1 · reward_feedback
S_next = 0.9 · S + 0.1 · stress_feedback
```

This design lets you explore how changing feedback and controller state changes the **magnitude of learning updates**. It does not imply that endocrine modulation must improve accuracy.

## 🎛️ Features

- **Two input modes:** demo/simulated signals for a reliable presentation, or camera + microphone capture.
- **Automatic speech transcription:** recorded audio is passed to the pretrained `openai/whisper-tiny.en` model; no manual transcript text box is required.
- **Text emotion cues:** the transcript is analyzed with `j-hartmann/emotion-english-distilroberta-base`, with an explainable keyword fallback if model loading fails.
- **Facial-expression cues:** attempts to use `trpakov/vit-face-expression` to estimate visible expression labels.
- **Audio feature extraction:** rough energy and zero-crossing features provide a limited acoustic-arousal proxy.
- **Feedback-based Hebbian learning:** select a target response and observe association weights update.
- **Artificial endocrine controller:** adjust reward-like and stress-like states and inspect the resulting learning modulation factor.
- **Model Insights:** inspect the weight matrix, animated network diagram, interaction history, and state charts.
- **A/B learning experiment:** compare ordinary and endocrine-modulated updates on identical synthetic examples with a repeatable seed.
- **Pink/peach UI:** distinct workspaces, styled cards, responsive layout, and reduced-motion support.
- **Optional shared-password gate:** configure `APP_PASSWORD` in Streamlit secrets before deployment.

## 🚀 Run it locally

### 1. Clone the repository

```bash
git clone https://github.com/sumanbhatbtech25-beep/Emotional-Intelligence.git
cd Emotional-Intelligence
```

### 2. Create and activate a virtual environment

**Windows PowerShell**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies and launch

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`). Open it in your browser.

## ☁️ Deploy with Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create an app from this repository and select `app.py` as the entry point.
4. Configure the app secrets described below if using the password gate.
5. Deploy and review the app logs if dependency installation or model loading fails.

**First-run note:** the speech and emotion models may download model weights the first time they are used. This needs network access and sufficient memory; initial model loading may take time. Use **Demo / simulated signals** first to explain the learning logic without waiting for model downloads.

## 🔐 Optional app password

The app includes a **shared-password gate**. It is not individual accounts, OAuth, or role-based authorization.

### Streamlit Community Cloud

In the app's **Settings → Secrets**, add:

```toml
APP_PASSWORD = "replace-with-a-long-unique-password"
```

Save the secret and allow the app to restart.

### Local development

Create `.streamlit/secrets.toml` in the project directory:

```toml
APP_PASSWORD = "replace-with-a-long-unique-password"
```

**Never commit this file or expose your password in source code.** If `APP_PASSWORD` is missing, the app remains locked and shows setup instructions. Anyone who knows the shared password can sign in; use platform-level access controls for sensitive deployments.

## 🧪 Suggested project demo

A simple presentation flow:

1. Open **Emotion Studio** and select **Demo / simulated signals**.
2. Adjust visual, audio-arousal, and uncertainty inputs.
3. Observe the response and emotion-cue visualizations.
4. Choose a feedback target and click **Learn from this interaction**.
5. Change the reward-like and stress-like states in the sidebar; compare learning modulation.
6. Open **Model Insights** to show the weight matrix and network connections.
7. Open **A/B Experiment**, choose a sample count, and run the comparison.
8. Explain the limitations: synthetic results are illustrative, and emotion estimates are uncertain.

## 📦 Project structure

```text
Emotional-Intelligence/
├── app.py            # Streamlit app, models, learning logic, and UI
├── requirements.txt  # Python dependencies
└── README.md         # Setup, architecture, and demo guide
```

## 🛡️ Privacy, limitations & responsible use

- Camera input is a **still capture** and microphone input is a **recorded clip**. Streamlit's built-in controls do not provide continuous always-on webcam or microphone streaming.
- Do not record other people without their informed consent.
- Model outputs and scores are estimates, not calibrated probabilities or diagnoses. A facial expression or spoken phrase cannot prove someone's internal emotional state.
- Audio loudness/texture is not a reliable way to identify a specific emotion.
- This repository integrates pretrained models and fallback heuristics; it is **not a newly fine-tuned model trained on a custom labeled dataset**.
- The A/B experiment uses synthetic data with a hand-designed target rule. It demonstrates update behavior, not real-world emotional understanding or guaranteed performance improvement.
- Initial model downloads may require network access, RAM, and startup time. If a model cannot load, inspect the Streamlit logs and use demo mode for the algorithm walkthrough.

## 🧭 Future directions

- Evaluate on a carefully consented, representative, labeled dataset.
- Report per-class metrics and uncertainty rather than relying on one accuracy number.
- Add user-controlled privacy options and clearer data-handling documentation.
- Compare learning variants over multiple seeds and report variability.
- Improve accessibility and test the UI across screen sizes and browsers.

## 📚 Project title

**NEUROENDO: A Non-Quantum Multimodal Hebbian Learning Model with an Artificial Endocrine System**

---

<p align="center">
  <strong>Built as an AIML learning prototype</strong><br>
  <sub>Explore the signals. Inspect the weights. Question the results.</sub>
</p>
