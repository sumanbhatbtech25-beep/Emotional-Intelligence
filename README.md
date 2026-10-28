# 🧠 NEUROENDO

<p align="center">
  <strong>Emotion cues · Adaptive learning · Artificial endocrine system</strong><br>
  A pink-and-peach, non-quantum AIML prototype exploring how feedback can influence learning.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="Approach" src="https://img.shields.io/badge/Approach-Classical%20%7C%20Non--Quantum-7D6BB3">
</p>

## 🌸 What it does

NEUROENDO combines visual cues, spoken language, simple audio features, a small **Hebbian association network**, and a simulated artificial endocrine controller. Explore inputs, see a response, provide feedback, and inspect how the model's weights change.

| Workspace | Explore |
|---|---|
| **♡ Emotion Studio** | Demo inputs or camera/audio capture, emotion cues, and feedback-based learning |
| **✧ Model Insights** | Learned weights, network visualization, and reward/stress history |
| **⇄ A/B Experiment** | Compare standard and endocrine-modulated Hebbian updates on synthetic data |

## 🧩 How it works

```text
Camera + microphone / demo inputs
              ↓
 Visual cues · Speech-to-text · Audio features
              ↓
       Feature vector x
              ↓
 Hebbian network ← Feedback target y
              ↑
 Simulated reward (D) / stress (S)
              ↓
       Response + insights
```

The learning update is:

```text
ΔW = η · g(H) · outer(x, y)
g(H) = clip(1 + D − S, 0.2, 1.5)
```

Reward and stress are **simulated controller variables**, not measured hormones. The A/B experiment uses a hand-designed synthetic dataset and is illustrative—not proof of real-world improvement.

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
2. Create an app from this GitHub repository and choose `app.py`.
3. Deploy and check the app logs if dependencies or models fail to load.

Speech and emotion models may download on first use, so startup can take time. Try **Demo / simulated signals** first for a quick presentation.

### Optional shared-password gate

In Streamlit app **Settings → Secrets**, add:

```toml
APP_PASSWORD = "replace-with-a-long-unique-password"
```

This is a shared-password gate, not individual accounts or OAuth. Never commit your secrets file or password to GitHub.

## 🛡️ Important limitations

- Camera input is a **still image** and microphone input is a **recorded clip**—not continuous, always-on streaming.
- Emotion estimates are uncertain cues, not diagnoses or proof of someone's internal feelings.
- Audio energy/texture is only a rough proxy; facial-expression cues can be unreliable.
- The app integrates pretrained models and fallback heuristics; it is not a custom fine-tuned emotion model.
- Do not record other people without their informed consent.

## 📁 Project files

- `app.py` — Streamlit interface, signal processing, learning logic, and experiments
- `requirements.txt` — Python dependencies
- `README.md` — Setup and project overview

---

<p align="center"><strong>Explore the signals ✦ Inspect the weights ✦ Question the results</strong></p>
