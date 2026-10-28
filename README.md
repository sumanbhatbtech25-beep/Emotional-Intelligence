# 🧠 NEUROENDO

<p align="center">
  <strong>Emotion cues · Adaptive learning · Artificial endocrine system</strong><br>
  A pink-and-peach, non-quantum AIML prototype exploring how feedback influences a small learning network.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Approach" src="https://img.shields.io/badge/Approach-Classical%20%7C%20Non--Quantum-7D6BB3">
</p>

## 🌸 What it does

NEUROENDO combines webcam frames, microphone audio, speech transcription, and visible-expression cues with a small **Hebbian association network**. Give feedback to adjust its weights, then explore the result in the dashboard.

| Workspace | Explore |
|---|---|
| **♡ Emotion Studio** | Simulated inputs or direct webcam + microphone recording |
| **✧ Model Insights** | Learned weights, network visualization, and reward/stress history |
| **⇄ A/B Experiment** | Compare ordinary and endocrine-modulated learning on synthetic data |

## 🎥 Try the recording workflow

1. Choose **Record with camera + mic**.
2. Allow camera and microphone permissions and press **START**.
3. Speak naturally for a few seconds, then press **STOP**.
4. Select **Analyze recorded inputs** to inspect the cues and transcript.
5. Choose a feedback target and select **Learn from this interaction**.

The live stream uses a low-resolution, reduced-frame-rate setting to help performance. The app samples a limited number of frames and caps audio capture at 30 seconds. The first model run may be slower while pretrained models download and load.

## 🧩 Project structure

```text
.
├── app.py                    # Streamlit UI and page layouts
├── neuroendo/
│   ├── __init__.py            # Python package marker
│   ├── learning.py            # Hebbian weights, endocrine rule, A/B experiment
│   ├── emotion.py             # Speech, text, facial and audio cue analysis
│   └── media.py               # Webcam/microphone capture and media helpers
├── requirements.txt           # Runtime dependencies
└── README.md                  # Setup and project guide
```

The entry point is intentionally small and imports helpers from the `neuroendo` package. Keep UI layout in `app.py`; put learning logic in `learning.py`, emotion inference in `emotion.py`, and capture/decoding code in `media.py`.

## 🧠 How learning works

```text
Webcam + microphone / demo signals
                ↓
      Feature extraction + cues
                ↓
         Feature vector x
                ↓
    Hebbian network ← feedback y
                ↑
    Reward (D) / stress (S)
                ↓
        Response + insights
```

The weight update is:

```text
ΔW = η · g(H) · outer(x, y)
g(H) = clip(1 + D − S, 0.2, 1.5)
```

The reward and stress values are **simulated controller variables**, not measured hormones. Learning updates the small association matrix during the current Streamlit session; it does **not** fine-tune the pretrained emotion models. Session state is not permanent storage, so learned weights may reset after a new session or app restart.

## 🚀 Run locally

```bash
git clone https://github.com/sumanbhatbtech25-beep/Emotional-Intelligence.git
cd Emotional-Intelligence
python -m venv .venv
```

Activate the environment:

- **Windows PowerShell:** `.\.venv\Scripts\Activate.ps1`
- **macOS/Linux:** `source .venv/bin/activate`

Then install and launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Create an app from this repository and select `app.py`.
3. Wait for dependencies and models to install; inspect the app logs if startup fails.

The browser will request camera and microphone permissions. Live recording may require HTTPS, which Streamlit Community Cloud provides. Model downloads can make the first analysis slow.

### Optional shared-password gate

In Streamlit app **Settings → Secrets**, add:

```toml
APP_PASSWORD = "replace-with-a-long-unique-password"
```

This is a shared-password gate, not individual accounts or OAuth. Never commit secrets or passwords to GitHub.

## 🛡️ Limitations and privacy

- Emotion outputs are uncertain cues, not diagnoses or proof of someone's internal feelings.
- Audio energy is only a rough arousal proxy; visible-expression cues can be unreliable.
- Pretrained models and fallback heuristics are used; this is not a custom fine-tuned emotion model.
- The A/B experiment uses a hand-designed synthetic dataset and is illustrative, not proof of real-world improvement.
- Record only yourself or people who have given informed consent. Captured media is held in the active app session for analysis; do not treat the app as secure long-term media storage.

---

<p align="center"><strong>Explore the signals ✦ Inspect the weights ✦ Question the results</strong></p>
