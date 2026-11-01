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
Two buttons in the top-right corner switch between the views:

**💬 Chat** (front page): live camera on one side, conversation on the other. Talk or type; NEUROENDO senses your emotion and replies.

**🧠 Emotional Learning**: the full non-quantum learning site, with these workspaces in its sidebar:

| Workspace | Explore |
|---|---|
| **♡ Emotion Studio** | Type a message or record with camera + mic, see the emotion snapshot, reply strategy scores and architecture, then accept or correct the reply |
| **☰ Last 5 Conversations** | A short title for each recent conversation (e.g. "Passed a test"), the emotion sensed → reply style, your feedback, and which weights went up or down |
| **✧ Model Insights** | Learned weights (incl. negative anti-Hebbian links), reward/stress/plasticity history, acceptance rate |
| **⇄ A/B Experiment** | Plain vs endocrine-modulated learning with simulated users whose preferences change |

## 💬 Using the chat

1. Press **START** on the camera and allow camera + microphone access.
2. Tell NEUROENDO about your day (e.g. *"I passed my test today!"*), then press **STOP**.
3. Analysis starts automatically. Your words appear in the chat with the emotion it sensed, followed by a reply, e.g. *"Hurray! 🎉 You passed your test — you've done great!"*
4. Press **👍 Nice reply**, or **👎 Wrong reply** to pick a better tone. It answers again in that tone and learns your preference.

You can also just type in the message box. Chat and Emotional Learning share the same network and endocrine state, so feedback in either one shows up in Model Insights.

Your latest Chat recording is analysed in **Emotional Learning → Emotion Studio** too (input mode **Use my Chat recording**). It shows the video frames that were analysed, the words heard, what the face and the voice each suggested, and the fused result, so you don't need to record twice.

Replies respond to life events NEUROENDO recognises: passing or failing a test, getting a job or promotion, winning, birthdays, rejection, being scolded, break-ups, loss, illness, nerves before an exam, and arguments. Everything else gets a reply in the chosen tone.

**Speech recognition.** Browser audio arrives as interleaved stereo; it is de-interleaved to mono, cleaned up (hum filter, resampled to 16 kHz, silence trimmed, volume evened out), then transcribed with Whisper `base.en`. For better accuracy on a more powerful server, set `WHISPER_MODEL = "openai/whisper-small.en"` in Streamlit secrets. The `.en` models expect English speech.

The live stream uses a low-resolution, reduced-frame-rate setting to help performance. The app samples a limited number of frames and caps audio capture at 30 seconds. The first model run may be slower while pretrained models download and load.

## 🧩 Project structure

```text
.
├── app.py                    # Streamlit UI and page layouts
├── neuroendo/
│   ├── __init__.py            # Python package marker
│   ├── learning.py            # Emotion features, Hebbian network, endocrine system, replies, A/B test
│   ├── emotion.py             # Speech, text, facial and audio analysis, fusion, uncertainty
│   ├── replies.py             # Life-event detection (negation-aware) and reply wording per strategy
│   ├── security.py            # Escaping, data wiping, idle timeout, login throttling
│   └── media.py               # Webcam/microphone capture and media helpers
├── .streamlit/config.toml     # Security-focused server settings
├── .streamlit/secrets.toml.example  # Template for APP_PASSWORD (real secrets.toml is git-ignored)
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

**3. Learning from feedback** — a three-factor Hebbian rule with an anti-Hebbian term, plus an explicit penalty for wrong replies:

```text
ΔW = η · g(H) · x (y − p)ᵀ                          η = 0.25,  p = softmax(3 · x·W)
ΔW[:, wrong] −= η · g(H) · 0.5 · x                  only when you reject a reply
```

`x yᵀ` strengthens the link to the reply you wanted. `−x pᵀ` weakens links to replies the network over-favours, so wrong associations fade. A rejected reply also gets the direct penalty, so the links that chose it go down, and below zero if they had little support. Only replies you give feedback on are learned from; messages with no feedback change nothing.

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
| A · plain Hebbian (η = 0.25) | 70.5% | 47.5% |
| **B · endocrine-modulated (η = 0.25)** | **72.1%** (better for 18/20 users) | **52.4%** |
| Plain, best η chosen with hindsight (0.5) | 71.2% | — |

The endocrine system beats plain learning at the same base rate, mostly by re-learning faster after the change. With the wrong-reply penalty it also edges out a plain learner whose rate was hand-tuned in hindsight. The margin is small, though: the benefit is **self-tuning plasticity**, not extra information. The app shows this comparison directly.

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

**Make the live camera work (important).** Most home, college and mobile networks block direct video between your browser and the cloud server, so the camera hangs on *"Connection is taking longer than expected"*. The fix is a relay (TURN) server, and a free one comes with a Hugging Face account:

1. Sign in at [huggingface.co](https://huggingface.co) → **Settings → Access Tokens → Create new token** (type **Read**).
2. In Streamlit Cloud, open your app → **Settings → Secrets** and add `HF_TOKEN = "hf_..."`.
3. Save. The app restarts, and the camera connects through the relay.

Cloudflare or Twilio TURN credentials work too (see `.streamlit/secrets.toml.example`). Locally on `localhost` you usually don't need any of this.

The browser will request camera and microphone permissions. Live recording may require HTTPS, which Streamlit Community Cloud provides. Model downloads can make the first analysis slow.

### Optional shared-password gate

Without a password the app is open.

In Streamlit app **Settings → Secrets**, add:

```toml
APP_PASSWORD = "replace-with-a-long-unique-password"
```

This is a shared-password gate, not individual accounts or OAuth. **Recommended for any public deployment**: without it, anyone with the link can use your server's compute. Use a long passphrase (16+ characters). A template is in `.streamlit/secrets.toml.example`; the real `secrets.toml` is git-ignored, so it can't be committed by accident.

## 🔒 Security and privacy

**Recordings are deleted right after use.**
- Camera frames and microphone audio are held **in memory only**, never written to disk.
- When you press STOP, the recording is analysed and then **wiped immediately**, even if the analysis fails. Raw audio is overwritten before it's released.
- Only derived results are kept for the session: emotion scores, the transcript, and the number of frames analysed. No images are kept, so none can be shown or downloaded later.
- **🗑️ Delete my data** (Chat page) erases your messages, transcripts, analysis results and any recording in progress.
- After **30 minutes of inactivity**, your conversation is deleted automatically. Closing the tab ends the session and frees everything.
- Each browser session is isolated: no visitor can see another visitor's data. Only the AI models are shared between sessions, and they hold no user data.

**Sign-in (when `APP_PASSWORD` is set)**
- The password is compared in constant time, and failed attempts are slowed down.
- After **5 wrong passwords in 15 minutes**, sign-in is locked for 15 minutes for that client IP and browser session.
- A sign-in lasts at most **8 hours**. Logging out, or an expired sign-in, wipes the session's data.

**Web hardening** (`.streamlit/config.toml`)
- Error details and stack traces are never shown to visitors (`showErrorDetails = "none"`); errors are logged on the server without user content.
- The developer menu is hidden (`toolbarMode = "viewer"`), XSRF and CORS protection are on, uploads are effectively disabled, and usage statistics aren't sent to Streamlit.
- Everything a user types or says is shown as literal text, so it can't inject links, images or HTML. Messages are capped at 1,000 characters, and per-session history is bounded.

**What this does not cover**
- The WebRTC connection uses Google's public STUN server to set up the video link, which sees your IP address (standard for browser video calls). The media itself is encrypted (DTLS-SRTP) between your browser and the server.
- Run the app over **HTTPS** (Streamlit Community Cloud does this). Browsers block the camera on plain HTTP anyway, except on localhost.
- The AI models are downloaded from Hugging Face on first use. For stricter supply-chain control, pin model revisions or host the models yourself.
- The shared password is one secret for everyone. For per-person accounts, put the app behind a proper login provider (for example, Streamlit's built-in OIDC sign-in).

## 🛡️ Limitations and privacy

- Emotion outputs are uncertain cues, not diagnoses or proof of someone's internal feelings.
- Audio energy is only a rough arousal proxy; visible-expression cues can be unreliable.
- Pretrained models and fallback heuristics are used; this is not a custom fine-tuned emotion model.
- Replies are templates per strategy and life event, not generated by a language model, so unusual messages get a general reply.
- The A/B experiment uses simulated users and is illustrative, not proof of real-world improvement.
- Record only yourself or people who have given informed consent.

---

<p align="center"><strong>Explore the signals ✦ Inspect the weights ✦ Question the results</strong></p>
