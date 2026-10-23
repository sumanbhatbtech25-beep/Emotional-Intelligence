import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="NEUROENDO", page_icon="🧠", layout="wide")

# Optional shared-password gate. Configure APP_PASSWORD in Streamlit secrets.
# This is a simple access gate, not individual user accounts or OAuth.
import hmac

def require_app_password():
    try:
        configured_password = st.secrets.get("APP_PASSWORD", "")
    except Exception:
        configured_password = ""

    if not configured_password:
        st.error("App access is not configured yet.")
        st.info(
            "For Streamlit Community Cloud, open your app's Settings → Secrets "
            "and add APP_PASSWORD = \"a-long-unique-password\". "
            "For local use, add the same entry to .streamlit/secrets.toml. "
            "Do not commit secrets.toml or put the password in source code."
        )
        st.stop()

    if st.session_state.get("authenticated", False):
        with st.sidebar:
            if st.button("Log out"):
                st.session_state["authenticated"] = False
                st.rerun()
        return

    st.title("🧠 NEUROENDO — Sign in")
    st.caption("Enter the shared app password to continue.")
    with st.form("app_login_form", clear_on_submit=True):
        password = st.text_input("App password", type="password")
        submitted = st.form_submit_button("Sign in", type="primary")

    if submitted:
        if hmac.compare_digest(str(password), str(configured_password)):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

require_app_password()


RESPONSES = ["Supportive", "Neutral", "Upbeat"]
FEATURES = ["Visual positive cue", "Audio arousal cue", "Uncertainty cue"]

def init_state():
    if "W" not in st.session_state:
        # Small, bounded association matrix: features x response categories
        st.session_state.W = np.zeros((3, 3), dtype=float)
    if "reward" not in st.session_state:
        st.session_state.reward = 0.20
    if "stress" not in st.session_state:
        st.session_state.stress = 0.15
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_x" not in st.session_state:
        st.session_state.last_x = None
    if "last_features" not in st.session_state:
        st.session_state.last_features = None
    if "last_pred" not in st.session_state:
        st.session_state.last_pred = None
    if "last_notes" not in st.session_state:
        st.session_state.last_notes = []
    if "ab_results" not in st.session_state:
        st.session_state.ab_results = None
    if "emotion_scores" not in st.session_state:
        st.session_state.emotion_scores = None
    if "emotion_notes" not in st.session_state:
        st.session_state.emotion_notes = []

def clip(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))

def endocrine_modulation(reward, stress):
    # Proposed engineering rule, not a biological law.
    return clip(1.0 + reward - stress, 0.2, 1.5)

def run_ab_experiment(seed=42, samples=120):
    """Compare ordinary and endocrine-modulated Hebbian updates on identical data."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, size=(samples, 3))
    # Deterministic teaching rule provides repeatable labels for this demonstration.
    target_scores = np.column_stack([
        0.65 * X[:, 2] + 0.20 * X[:, 1],
        0.45 * (1.0 - X[:, 2]) + 0.20 * (1.0 - X[:, 0]),
        0.60 * X[:, 0] + 0.25 * X[:, 1]
    ])
    y_idx = np.argmax(target_scores, axis=1)
    split = int(samples * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_idx[:split], y_idx[split:]

    W_plain = np.zeros((3, 3), dtype=float)
    W_endo = np.zeros((3, 3), dtype=float)
    d_state, s_state = 0.20, 0.15
    eta = 0.10

    for x_i, target_i in zip(X_train, y_train):
        y = np.zeros(3, dtype=float)
        y[target_i] = 1.0
        # A: ordinary Hebbian learning, no endocrine modulation.
        W_plain = np.clip(W_plain + eta * np.outer(x_i, y), -2.0, 2.0)
        # B: same sample and target, with the simulated endocrine factor.
        factor = endocrine_modulation(d_state, s_state)
        W_endo = np.clip(W_endo + eta * factor * np.outer(x_i, y), -2.0, 2.0)
        d_state, s_state = update_endocrine(d_state, s_state, 0.8, 0.2)

    def evaluate(W):
        scores = X_test @ W
        pred = np.argmax(scores, axis=1)
        return float(np.mean(pred == y_test)), pred

    acc_plain, pred_plain = evaluate(W_plain)
    acc_endo, pred_endo = evaluate(W_endo)
    return {
        "samples": samples,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "plain_accuracy": acc_plain,
        "endo_accuracy": acc_endo,
        "plain_weights": W_plain,
        "endo_weights": W_endo,
        "plain_predictions": pred_plain,
        "endo_predictions": pred_endo,
        "test_targets": y_test,
    }


def visual_features(image_bytes):
    """Extract a coarse visual cue using OpenCV Haar cascades.
    This is NOT reliable emotion recognition; it only estimates a smile-like visual cue."""
    try:
        import cv2
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return 0.0, 1.0, ["Could not decode image; visual input marked uncertain."]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        if len(faces) == 0:
            return 0.0, 1.0, ["No face detected; visual input marked uncertain."]
        # Analyze the largest detected face only.
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        roi = gray[y + int(h * 0.45): y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi, scaleFactor=1.7, minNeighbors=20, minSize=(20, 10))
        smile_cue = 1.0 if len(smiles) > 0 else 0.0
        return smile_cue, 0.25, [
            f"Detected {len(faces)} face(s).",
            "Smile-like visual cue detected." if smile_cue else "No clear smile-like cue detected."
        ]
    except Exception as e:
        return 0.0, 1.0, [f"Visual feature extraction unavailable: {e}"]

def audio_features(audio_bytes):
    """Estimate audio arousal from simple signal features. This is not emotion recognition."""
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(io.BytesIO(audio_bytes))
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        if data.size == 0:
            return 0.0, ["Audio clip was empty."]
        peak = max(float(np.max(np.abs(data))), 1.0)
        # Normalize integer PCM to roughly [-1, 1], and handle float recordings.
        if peak > 1.5:
            data = data / peak
        else:
            data = data / max(1.0, peak)
        rms = float(np.sqrt(np.mean(data ** 2)))
        # Zero-crossing rate is a rough texture feature, not a direct emotion measure.
        zcr = float(np.mean(np.abs(np.diff(np.signbit(data)))))
        # Normalize using broad heuristic ranges; microphone gain strongly affects this.
        energy_score = clip(rms / 0.20)
        texture_score = clip(zcr / 0.25)
        arousal = clip(0.7 * energy_score + 0.3 * texture_score)
        return arousal, [
            f"Sample rate: {sr} Hz.",
            f"Relative energy feature: {energy_score:.2f}.",
            f"Zero-crossing feature: {texture_score:.2f}.",
            "Audio arousal is a rough acoustic proxy; it does not identify a person's emotion."
        ]
    except Exception as e:
        return 0.0, [
            "Could not decode audio. Please use a WAV recording or use Demo mode.",
            f"Details: {e}"
        ]

def make_response(pred, visual, audio, uncertain):
    if uncertain > 0.7:
        return "I couldn't confidently interpret the available signals. Would you like to try again or tell me how you're feeling?"
    if pred == "Supportive":
        return "I'm here to help. Would you like to talk through what's happening or work on it step by step?"
    if pred == "Upbeat":
        return "That's great! Would you like to build on that momentum?"
    return "I'm listening. What would be most useful for you right now?"

def update_endocrine(reward, stress, feedback_reward, feedback_stress):
    # Leaky state update: retain 90% of prior state, add 10% of current input.
    reward = clip(0.90 * reward + 0.10 * feedback_reward)
    stress = clip(0.90 * stress + 0.10 * feedback_stress)
    return reward, stress

init_state()

st.title("🧠 NEUROENDO")
st.caption("A non-quantum, biologically inspired multimodal learning demonstrator")
st.markdown(
    "Camera and microphone cues feed a small Hebbian association network. "
    "A simulated artificial endocrine system modulates how strongly the network learns."
)

with st.sidebar:
    st.header("Endocrine controller")
    st.write("These are simulated regulatory variables, not measured biological hormones.")
    st.session_state.reward = st.slider("Reward-like signal (D)", 0.0, 1.0, float(st.session_state.reward), 0.05)
    st.session_state.stress = st.slider("Stress-like signal (S)", 0.0, 1.0, float(st.session_state.stress), 0.05)
    mod = endocrine_modulation(st.session_state.reward, st.session_state.stress)
    st.metric("Learning modulation g(H)", f"{mod:.2f}×")
    st.caption("Rule: g(H) = clip(1 + D − S, 0.2, 1.5)")
    if st.button("Reset learned weights and history"):
        st.session_state.W = np.zeros((3, 3), dtype=float)
        st.session_state.history = []
        st.session_state.last_x = None
        st.session_state.last_features = None
        st.session_state.last_pred = None
        st.session_state.last_notes = []
        st.rerun()

mode = st.radio("Input mode", ["Demo / simulated signals", "Camera + microphone"], horizontal=True)

st.markdown("### Describe what happened (recommended)")
st.caption("Words often provide more context than a facial expression. The model estimates emotion from text and, when available, a facial-expression model.")
emotion_text = st.text_area(
    "What happened, or what are you feeling?",
    placeholder="Example: I got scolded because I was late to a meeting, and I feel embarrassed and sad.",
    height=90,
    key="emotion_text_input"
)
if st.button("Analyze my words", key="analyze_words"):
    text_scores, text_notes = infer_text_emotion(emotion_text)
    st.session_state.emotion_scores = combine_emotion_scores(text_scores=text_scores)
    st.session_state.emotion_notes = text_notes
    st.session_state.emotion_source = "text"

col1, col2 = st.columns(2)
visual_cue = None
audio_cue = None
visual_uncertainty = 0.0
notes = []

if mode == "Demo / simulated signals":
    with col1:
        visual_cue = st.slider("Visual positive cue (simulated)", 0.0, 1.0, 0.7, 0.1)
    with col2:
        audio_cue = st.slider("Audio arousal cue (simulated)", 0.0, 1.0, 0.4, 0.1)
    visual_uncertainty = st.slider("Visual uncertainty (simulated)", 0.0, 1.0, 0.2, 0.1)
    notes = ["Demo inputs are simulated so the algorithm can be demonstrated consistently."]
else:
    st.info("Capture a still image and a short audio clip. Your browser may ask for camera/microphone permission. For a live continuous stream, a separate browser frontend is usually needed.")
    with col1:
        img_file = st.camera_input("Capture visual input")
    with col2:
        audio_file = st.audio_input("Record a short audio clip")
    if st.button("Extract features from captured media"):
        if img_file is None or audio_file is None:
            st.warning("Capture both an image and an audio clip first.")
        else:
            visual_cue, visual_uncertainty, vnotes = visual_features(img_file.getvalue())
            face_scores, face_notes = infer_facial_emotion(img_file.getvalue())
            text_scores, text_notes = infer_text_emotion(emotion_text) if emotion_text.strip() else (None, [])
            audio_cue, anot = audio_features(audio_file.getvalue())
            st.session_state.emotion_scores = combine_emotion_scores(
                text_scores=text_scores, face_scores=face_scores, audio_arousal=audio_cue
            )
            st.session_state.emotion_notes = text_notes + face_notes + anot
            st.session_state.emotion_source = "text + face + acoustic proxy"
            st.session_state.last_features = {
                "visual": visual_cue, "audio": audio_cue, "uncertainty": visual_uncertainty
            }
            st.session_state.last_notes = vnotes + anot
            st.success("Features extracted. Scroll down to run the learning model.")


# Demo mode updates the current input on each rerun. Captured mode uses the last extracted feature set.
if mode == "Demo / simulated signals":
    current = {"visual": float(visual_cue), "audio": float(audio_cue), "uncertainty": float(visual_uncertainty)}
    st.session_state.last_features = current
    st.session_state.last_notes = notes
elif st.session_state.last_features is not None:
    current = st.session_state.last_features
else:
    current = None

st.divider()
with st.expander("System architecture", expanded=False):
    st.markdown("""
    ```text
    Camera / microphone or demo signals
                   |
                   v
       Classical feature extraction
                   |
                   v
        Feature vector x = [V, A, U]
                   |
                   v
       Hebbian association network <---- Feedback target y
                   ^                         |
                   |                         v
       Endocrine modulation g(H) <---- Reward D / Stress S
                   |
                   v
          Response + live dashboard
    ```
    """)
    st.caption("All processing is classical and non-quantum. D and S are simulated controller states, not measured hormones.")

st.subheader("Learning and response")

if current is None:
    st.info("Select Demo mode or capture both an image and audio clip, then extract features.")
else:
    x = np.array([current["visual"], current["audio"], current["uncertainty"]], dtype=float)
    # Bias toward a neutral response initially to avoid arbitrary argmax ties.
    W = st.session_state.W
    scores = x @ W
    if np.allclose(scores, 0):
        pred_idx = 1
    else:
        pred_idx = int(np.argmax(scores))
    pred = RESPONSES[pred_idx]
    st.session_state.last_x = x
    st.session_state.last_pred = pred

    a, b, c = st.columns(3)
    a.metric("Visual positive cue", f'{current["visual"]:.2f}')
    b.metric("Audio arousal cue", f'{current["audio"]:.2f}')
    c.metric("Uncertainty cue", f'{current["uncertainty"]:.2f}')

    if st.session_state.emotion_scores:
        emotion_scores = st.session_state.emotion_scores
        best_emotion = max(emotion_scores, key=emotion_scores.get)
        best_score = emotion_scores[best_emotion]
        st.markdown("#### Emotion estimate (separate from the response network)")
        st.metric("Most likely cue", best_emotion.title(), f"{best_score:.0%} relative score")
        emotion_df = pd.DataFrame({
            "Emotion": list(emotion_scores.keys()),
            "Score": list(emotion_scores.values())
        }).set_index("Emotion")
        st.bar_chart(emotion_df)
        for note in st.session_state.emotion_notes:
            st.write(f"- {note}")
        st.caption("Scores are model/heuristic outputs, not calibrated probabilities or a diagnosis. Check whether the estimate matches how you actually feel.")

    st.write("**Extracted feature notes**")
    for note in st.session_state.last_notes:
        st.write(f"- {note}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Neural association scores")
        score_df = pd.DataFrame({"Response": RESPONSES, "Score": scores})
        st.bar_chart(score_df.set_index("Response"))
        st.caption("Scores are learned associations, not probabilities.")
    with right:
        st.markdown("#### Current response")
        st.markdown(f"### {pred}")
        st.write(make_response(pred, current["visual"], current["audio"], current["uncertainty"]))

    st.markdown("#### Teach the model from feedback")
    st.write("Choose the response that best fits this interaction. The selected feedback label updates the association weights.")
    chosen = st.selectbox("Feedback target", RESPONSES, index=pred_idx, key="feedback_target")
    reward_feedback = st.slider("Outcome reward", 0.0, 1.0, 0.8, 0.1, key="outcome_reward")
    stress_feedback = st.slider("Challenge / stress cue", 0.0, 1.0, 0.2, 0.1, key="challenge_stress")
    if st.button("Learn from this interaction", type="primary"):
        y = np.zeros(3, dtype=float)
        y[RESPONSES.index(chosen)] = 1.0
        eta = 0.10
        modulation = endocrine_modulation(st.session_state.reward, st.session_state.stress)
        delta = eta * modulation * np.outer(x, y)
        st.session_state.W = np.clip(st.session_state.W + delta, -2.0, 2.0)
        st.session_state.reward, st.session_state.stress = update_endocrine(
            st.session_state.reward, st.session_state.stress, reward_feedback, stress_feedback
        )
        st.session_state.history.append({
            "Interaction": len(st.session_state.history) + 1,
            "Target": chosen,
            "Reward D": st.session_state.reward,
            "Stress S": st.session_state.stress,
            "Learning factor": modulation,
            "Weight change norm": float(np.linalg.norm(delta))
        })
        st.success(f"Learned association for '{chosen}'. Weight-change magnitude: {np.linalg.norm(delta):.3f}")
        st.rerun()

st.divider()
st.subheader("Inspect the model")

w_df = pd.DataFrame(st.session_state.W, index=FEATURES, columns=RESPONSES)
st.markdown("**Hebbian weight matrix** — rows are input features, columns are response neurons.")
st.dataframe(w_df.style.format("{:.3f}"), use_container_width=True)

if st.session_state.history:
    hist = pd.DataFrame(st.session_state.history)
    st.markdown("**Learning history**")
    st.dataframe(hist, use_container_width=True)
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown("Reward-like state over interactions")
        st.line_chart(hist.set_index("Interaction")[["Reward D"]])
    with chart_col2:
        st.markdown("Stress-like state over interactions")
        st.line_chart(hist.set_index("Interaction")[["Stress S"]])
else:
    st.info("No learning updates yet. Run an input, choose feedback, and click 'Learn from this interaction' to populate the weight matrix.")

with st.expander("Model equations and limitations"):
    st.markdown(r"""
**Hebbian update with endocrine modulation**

\[
\Delta W = \eta\,g(H)\,x\,y^T
\]

where \(x\) is the input feature vector, \(y\) is a one-hot feedback target, and
\(g(H)=\operatorname{clip}(1+D-S,0.2,1.5)\).

**Leaky endocrine states**

\[
D_{t+1}=0.9D_t+0.1R_t,\quad
S_{t+1}=0.9S_t+0.1Q_t
\]

These are intentionally simple engineering choices for a prototype. The camera feature is a basic smile-like cue and the audio feature is a rough acoustic-arousal proxy. Neither can reliably determine a person's internal emotional state. The model demonstrates learning dynamics and feedback, not clinical or definitive emotion detection.
""")

st.divider()
st.subheader("A/B experiment: ordinary vs endocrine-modulated Hebbian learning")
st.write(
    "Both variants receive the same reproducible synthetic inputs, labels, training/test split, "
    "and base learning rate. Variant A uses ordinary Hebbian updates; Variant B multiplies each "
    "update by the simulated endocrine factor. This is a controlled demonstration, not evidence "
    "of real-world emotional understanding."
)
ab_col1, ab_col2 = st.columns([1, 2])
with ab_col1:
    ab_samples = st.select_slider(
        "Synthetic examples", options=[60, 120, 240, 480], value=120, key="ab_samples"
    )
    if st.button("Run A/B comparison", type="primary"):
        st.session_state.ab_results = run_ab_experiment(samples=int(ab_samples))
with ab_col2:
    st.caption("The experiment uses fixed seed 42 so results are repeatable for the same sample count.")

if st.session_state.ab_results is not None:
    ab = st.session_state.ab_results
    a1, a2, a3 = st.columns(3)
    a1.metric("Ordinary Hebbian accuracy", f"{ab['plain_accuracy']:.1%}")
    a2.metric("Endocrine-modulated accuracy", f"{ab['endo_accuracy']:.1%}")
    a3.metric("Difference (B − A)", f"{(ab['endo_accuracy'] - ab['plain_accuracy']):+.1%}")
    comp_df = pd.DataFrame({
        "Variant": ["A · Ordinary Hebbian", "B · Endocrine-modulated"],
        "Test accuracy": [ab["plain_accuracy"], ab["endo_accuracy"]]
    }).set_index("Variant")
    st.bar_chart(comp_df)
    st.write(
        f"Training examples: {ab['train_samples']} · Test examples: {ab['test_samples']} · "
        "Same generated dataset and labels for both variants."
    )
    st.dataframe(pd.DataFrame({
        "Variant": ["A · Ordinary Hebbian", "B · Endocrine-modulated"],
        "Final test accuracy": [ab["plain_accuracy"], ab["endo_accuracy"]],
        "Weight matrix norm": [
            float(np.linalg.norm(ab["plain_weights"])),
            float(np.linalg.norm(ab["endo_weights"]))
        ]
    }), use_container_width=True)
    st.caption(
        "Interpret cautiously: the synthetic labels follow a hand-designed rule, and endocrine "
        "modulation changes update magnitude rather than adding new information. The comparison "
        "is illustrative and does not guarantee that Variant B performs better."
    )

st.caption("Privacy note: this prototype processes captured media in the app session. Avoid recording other people without their informed consent.")
