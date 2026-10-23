# NEUROENDO — AIML Project Prototype

A non-quantum, biologically inspired multimodal learning demonstrator combining:
- Visual input from a camera capture
- Audio input from a microphone recording
- A small Hebbian association matrix
- A simulated artificial endocrine controller
- A Streamlit dashboard

## Run online

### Easiest option: Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py` and `requirements.txt`.
2. Open https://share.streamlit.io/ and deploy the repository.
3. Grant browser camera/microphone permissions when prompted.
4. Use **Demo / simulated signals** first, then try **Camera + microphone**.

Note: browser audio capture may be unavailable on some browsers or deployments. The demo mode always works and is useful for explaining the algorithm.

## Run locally

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## How to demonstrate it
1. Select Demo mode.
2. Change visual, audio, and uncertainty sliders.
3. Observe the selected response and association scores.
4. Select a feedback target and click **Learn from this interaction**.
5. Repeat with different reward/stress states.
6. Show the changing Hebbian weight matrix and endocrine-state plots.
7. Optionally switch to Camera + microphone and capture a still image and short audio clip.

## Model equations

Hebbian update:
`ΔW = η * g(H) * outer(x, y)`

Endocrine modulation:
`g(H) = clip(1 + D - S, 0.2, 1.5)`

Leaky states:
`D_next = 0.9*D + 0.1*reward_feedback`
`S_next = 0.9*S + 0.1*stress_feedback`

This is a deliberately simple prototype. The camera feature is a basic smile-like cue and audio features are rough acoustic proxies. These are not reliable measures of emotion and should not be presented as such. The project demonstrates a proposed way to regulate classical Hebbian learning using internal state variables.

## Suggested project title
**NEUROENDO: A Non-Quantum Multimodal Hebbian Learning Model with an Artificial Endocrine System**

## App access protection

The app now includes an optional shared-password sign-in gate. Configure it before using the app:

### Streamlit Community Cloud
1. Open your app in Streamlit Community Cloud.
2. Open **Settings → Secrets**.
3. Add a strong, unique password:

```toml
APP_PASSWORD = "replace-with-a-long-unique-password"
```

4. Save the secrets and restart/redeploy the app if needed.

### Local development
Create `.streamlit/secrets.toml` in the project directory with the same `APP_PASSWORD` entry, then run `streamlit run app.py`. Do not commit `.streamlit/secrets.toml` to GitHub.

If `APP_PASSWORD` is not configured, the app remains locked and displays setup instructions. The sign-in is a **shared-password gate**, not individual accounts, OAuth, or role-based authorization. Anyone who knows the shared password can sign in. Use platform-level access restrictions for sensitive/private deployments, and do not treat this alone as a full identity system.
