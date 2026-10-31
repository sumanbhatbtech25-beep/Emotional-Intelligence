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

NEUROENDO reads your face, voice and words, turns them into **emotion scores** (sadness, anger, fear, joy, surprise, neutral), and uses those scores to decide **how to reply**: supportive, de-escalating, calming, upbeat, curious or neutral. The reply strategy is chosen by a small **Hebbian association network**. A simulated **artificial endocrine system** reacts to your emotional state and to whether its replies land. It biases the system toward careful replies when stress is high and controls how quickly the network re-learns from your feedback. Everything is classical (non-quantum).

| Workspace | Explore |
|---|---|
| **💬 Chat** (front page) | Live camera on one side, conversation on the other. Talk or type; NEUROENDO senses your emotion and replies |
| **✧ Model Insights** (sidebar) | Learned weights (incl. negative anti-Hebbian links), reward/stress/plasticity history, acceptance rate |
| **⇄ A/B Experiment** (sidebar) | Plain vs endocrine-modulated learning with simulated users whose preferences change |

## 💬 Using the chat

1. Press **START** on the camera and allow camera + microphone access.
2. Tell NEUROENDO about your day (e.g. *"I passed my test today!"*), then press **STOP**.
3. Press **Send what I said**. Your words appear in the chat with the emotion it sensed, followed by a reply, e.g. *"Hurray! 🎉 You passed your test — you've done great!"*
4. Press **👍 Nice reply**, or **🙈 Reply differently** to pick a better tone. It answers again in that tone and learns your preference.

You can also just type in the message box. The learning details live in **Model Insights** and **A/B Experiment** in the sidebar (open it with the `»` arrow at the top left).

Replies respond to life events NEUROENDO recognises: passing or failing a test, getting a job or promotion, winning, birthdays, rejection, being scolded, break-ups, loss, illness, nerves before an exam, and arguments. Everything else gets a reply in the chosen tone.

The live stream uses a low-resolution, reduced-frame-rate setting to help performance. The app samples a limited number of frames and caps audio capture at 30 seconds. The first model run may be slower while pretrained models download and load.

## 🧩 Project structure

```text
.
├── app.py                    # Streamlit UI and page layouts
├── neuroendo/
│   ├── __init__.py            # Python package marker
│   ├── learning.py            # Emotion features, Hebbian network, endocrine system, replies, A/B test
│   ├── emotion.py             # Speech, text, facial and audio analysis, fusion, uncertainty
│   ├── replies.py             # Life-event detection and reply wording per strategy
│   └── media.py               # Webcam/microphone capture and media helpers
├── requirements.txt           # Runtime dependencies
└── README.md                  # Setup and project guide
```

The entry point is intentionally small and imports helpers from the `neuroendo` package. Keep UI layout in `app.py`; put learning logic in `learning.py`, emotion inference in `emotion.py`, and capture/decoding code in `media.py`.

## 🧠 How it works

```text
Webcam frames ──► face-expression model ─┐
Microphone ─────► Whisper speech-to-text ─► text-emotion model ─┤──► fused emotion scores
            └───► acoustic arousal ──────────────────────────────┘        + arousal + uncertainty
                                                                              = feature vector x (8)
                                                                                     │
                              Artificial endocrine system                            ▼
                          Reward D ◄─ accepted replies, positive emotion      x·W + stress bias
                          Stress S ◄─ corrected replies, negative/aroused     ──► reply strategy + reply
                                │                                                    │
                     g(H) = plasticity ──────────► ΔW = η·g(H)·x(y − p)ᵀ ◄── your feedback y
```

**1. Emotion recognition.** Sampled webcam frames go to a pretrained facial-expression model (`trpakov/vit-face-expression`). Speech is transcribed with Whisper Tiny and classified by a pretrained text-emotion model (`j-hartmann/emotion-english-distilroberta-base`), with a transparent keyword fallback if that model can't load. Scores are fused (65% text, 35% face, plus a small arousal nudge). Uncertainty rises when a signal is missing, when face and words disagree, or when no emotion clearly leads.

**2. Choosing a reply.** The feature vector `x = [6 emotion scores, arousal, uncertainty]` is multiplied by the weight matrix `W` (8 inputs × 6 reply strategies). `W` starts from a weak default pairing (sadness→Supportive, anger→De-escalating, fear→Calming, joy→Upbeat, surprise→Curious, neutral/uncertain→Neutral), so replies make sense before any training. The winning strategy is turned into words by `replies.py`. If you mention a recognised life event, the reply is about that event, in the chosen tone. A recognised event also nudges the emotion scores toward its usual emotion (e.g. "I passed" → joy), in case the text model reads it as neutral.

**3. Learning from feedback** — a three-factor Hebbian rule with an anti-Hebbian term:

```text
ΔW = η · g(H) · x (y − p)ᵀ        η = 0.25,  p = softmax(3 · x·W)
```

`x yᵀ` strengthens the link to the reply you wanted. `−x pᵀ` weakens links to replies the network over-favours, so wrong associations fade and weights can become negative.

**4. Artificial endocrine system.** Two leaky hormone-like states:

```text
D ← 0.75·D + 0.25·(0.7·accepted  + 0.3·positive-emotion drive)
S ← 0.75·S + 0.25·(0.7·corrected + 0.3·negative/high-arousal drive)
g(H) = clip(1 + 2S − D, 0.2, 2.0)     # plasticity
bias = +0.15·S on Supportive / Calming / De-escalating
```

Stress speeds re-learning when replies keep missing, and sustained reward consolidates what works. A distressed user also nudges the system toward careful replies.

D and S are **simulated controller variables**, not measured hormones. The network learns during the current Streamlit session only and does **not** fine-tune the pretrained emotion models.

### What the A/B experiment shows

With 20 simulated users × 240 interactions, preferences that differ from the default, 10% noisy feedback, and a preference change half-way:

| Variant | Accuracy | After preference change |
|---|---|---|
| A · plain Hebbian (η = 0.25) | 69.2% | 45.2% |
| **B · endocrine-modulated (η = 0.25)** | **71.5%** (better for 20/20 users) | **48.6%** |
| Plain, best η chosen with hindsight (0.5) | 72.0% | — |

The endocrine system beats plain learning at the same base rate, mostly by re-learning faster after the change. A plain learner whose rate is hand-tuned in hindsight reaches about the same accuracy. The benefit is **self-tuning plasticity**, not extra information. The app shows this comparison directly.

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

Without a password the app is open.

In Streamlit app **Settings → Secrets**, add:

```toml
APP_PASSWORD = "replace-with-a-long-unique-password"
```

This is a shared-password gate, not individual accounts or OAuth. Never commit secrets or passwords to GitHub.

## 🛡️ Limitations and privacy

- Emotion outputs are uncertain cues, not diagnoses or proof of someone's internal feelings.
- Audio energy is only a rough arousal proxy; visible-expression cues can be unreliable.
- Pretrained models and fallback heuristics are used; this is not a custom fine-tuned emotion model.
- Replies are templates per strategy and life event, not generated by a language model, so unusual messages get a general reply.
- The A/B experiment uses simulated users and is illustrative, not proof of real-world improvement.
- Record only yourself or people who have given informed consent. Captured media is held in the active app session for analysis; do not treat the app as secure long-term media storage.

---

<p align="center"><strong>Explore the signals ✦ Inspect the weights ✦ Question the results</strong></p>
