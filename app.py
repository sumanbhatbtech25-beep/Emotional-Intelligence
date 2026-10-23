import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="NEUROENDO", page_icon="🧠", layout="wide")

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

def clip(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))

def endocrine_modulation(reward, stress):
    # Proposed engineering rule, not a biological law.
    return clip(1.0 + reward - stress, 0.2, 1.5)

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
            audio_cue, anot = audio_features(audio_file.getvalue())
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
st.caption("Privacy note: this prototype processes captured media in the app session. Avoid recording other people without their informed consent.")
