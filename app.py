import numpy as np
import pandas as pd
import streamlit as st
from neuroendo.learning import (
    EMOTIONS, FEATURES, RESPONSES, init_state, reset_state, render_network_svg, run_ab_experiment,
    build_features, predict, hebbian_update, endocrine_modulation, endocrine_step, emotional_drive,
    make_response,
)
from neuroendo.emotion import (
    transcribe_audio, infer_text_emotion, infer_facial_emotion, combine_emotion_scores,
    visual_features, audio_features, average_emotion_scores, estimate_uncertainty,
)
from neuroendo.media import LiveCaptureBuffer, WebcamVideoProcessor, MicrophoneAudioProcessor, extract_video_frames, extract_video_audio, webrtc_streamer, WebRtcMode, RTCConfiguration


st.set_page_config(page_title="NEUROENDO", page_icon="🧠", layout="wide")

# NEUROENDO pink + peach visual system
st.markdown("""
<style>
:root {
  --neo-pink: #D94F87;
  --neo-pink-deep: #A92F65;
  --neo-pink-soft: #FCE7F0;
  --neo-peach: #FFF0E5;
  --neo-peach-deep: #F5B99D;
  --neo-ink: #432A38;
  --neo-muted: #806A76;
}
.stApp {
  background:
    radial-gradient(ellipse at 8% 0%, rgba(255, 211, 226, .52), transparent 34%),
    radial-gradient(ellipse at 96% 8%, rgba(255, 221, 194, .55), transparent 32%),
    linear-gradient(180deg, #FFFAF8 0%, #FFF5F2 55%, #FFF9F7 100%);
  color: var(--neo-ink);
}
[data-testid="stHeader"] { background: rgba(255, 250, 248, .78); }
[data-testid="stToolbar"] { right: 1rem; }
.block-container {
  max-width: 1440px;
  padding-top: 2rem;
  padding-bottom: 3.5rem;
}
h1, h2, h3 {
  color: var(--neo-ink) !important;
  letter-spacing: -.025em;
}
h1 { font-weight: 800 !important; }
h2, h3 { font-weight: 720 !important; }
p, label, .stMarkdown, [data-testid="stCaptionContainer"] {
  color: var(--neo-ink);
}
[data-testid="stCaptionContainer"] p { color: var(--neo-muted) !important; }
[data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(255,255,255,.94), rgba(252,231,240,.78));
  border: 1px solid #F4C5D8;
  padding: 1rem 1.1rem;
  border-radius: 18px;
  box-shadow: 0 7px 22px rgba(169, 47, 101, .07);
}
[data-testid="stMetricLabel"] { color: var(--neo-muted) !important; }
[data-testid="stMetricValue"] { color: var(--neo-pink-deep) !important; }
.stButton > button, .stFormSubmitButton > button {
  border: 1px solid #D94F87;
  border-radius: 12px;
  background: linear-gradient(135deg, #D94F87, #BE3D76);
  color: white;
  font-weight: 700;
  padding: .55rem 1.05rem;
  transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
  box-shadow: 0 5px 14px rgba(217, 79, 135, .18);
}
.stButton > button:hover, .stFormSubmitButton > button:hover {
  color: white;
  border-color: #A92F65;
  filter: brightness(1.04);
  transform: translateY(-1px);
  box-shadow: 0 8px 18px rgba(217, 79, 135, .25);
}
.stButton > button:focus, .stFormSubmitButton > button:focus {
  outline: 3px solid rgba(217, 79, 135, .22);
  outline-offset: 2px;
}
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div,
div[data-baseweb="select"] > div {
  background: rgba(255,255,255,.9);
  border-color: #EAC5D3;
  border-radius: 12px;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="textarea"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
  border-color: var(--neo-pink);
  box-shadow: 0 0 0 1px var(--neo-pink);
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #FCE7F0 0%, #FFF0E5 100%);
  border-right: 1px solid #F3C7D8;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--neo-pink-deep) !important; }
[data-testid="stExpander"] {
  border: 1px solid #F0C6D6;
  border-radius: 15px;
  background: rgba(255,255,255,.58);
  overflow: hidden;
}
[data-testid="stAlert"] { border-radius: 14px; }
hr { border-color: #F2D2DE !important; }
[data-testid="stDataFrame"], [data-testid="stTable"] {
  border: 1px solid #F0C6D6;
  border-radius: 14px;
  overflow: hidden;
}
[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, #E46A9B, #F2B08F);
}
/* Refined interactions and branded scrollbars */
html { scroll-behavior: smooth; scrollbar-color: #E58AAF #FFF0E5; scrollbar-width: thin; }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: #FFF0E5; border-radius: 99px; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg,#E58AAF,#F3B18F); border: 2px solid #FFF0E5; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #C7477D; }
::selection { background: #F8C9DC; color: #432A38; }
.stRadio [role="radiogroup"] { gap:.55rem; padding:.45rem; border-radius:16px; background:rgba(252,231,240,.68); border:1px solid #F1C8D8; }
.stRadio [role="radio"] { padding:.65rem .9rem; border-radius:12px; transition:all .18s ease; }
.stRadio [role="radio"]:hover { background:rgba(255,255,255,.85); }
.stSlider [data-baseweb="slider"] [role="slider"] { background:#C7477D; border-color:#C7477D; box-shadow:0 0 0 5px rgba(217,79,135,.13); }
.stSlider [data-baseweb="slider"] > div > div { background:linear-gradient(90deg,#D94F87,#F2B08F); }
[data-testid="stCheckbox"] label, [data-testid="stRadio"] label, [data-testid="stSelectbox"] label, [data-testid="stSlider"] label { font-weight:650; color:#5A3A4B; }
[data-testid="stChart"] { background:rgba(255,255,255,.68); padding:.65rem; border:1px solid #F4D5E0; border-radius:18px; }
[data-testid="stExpander"] summary { font-weight:700; color:#A92F65; padding:.85rem 1rem; }
[data-testid="stExpander"] summary:hover { background:rgba(252,231,240,.62); }
[data-testid="stAlert"] { border:1px solid #F0C6D6; box-shadow:0 5px 16px rgba(169,47,101,.045); }
.stMarkdown a { color:#A92F65 !important; font-weight:700; text-decoration-thickness:1.5px; }
.stMarkdown a:hover { color:#D94F87 !important; }
[data-testid="stMetric"] { transition:transform .18s ease, box-shadow .18s ease; }
[data-testid="stMetric"]:hover { transform:translateY(-2px); box-shadow:0 12px 25px rgba(169,47,101,.12); }
.neo-section { display:flex; align-items:center; gap:.8rem; margin:1.7rem 0 1rem; padding:.85rem 1.05rem; border-radius:16px; border:1px solid #F0C3D5; background:linear-gradient(100deg,rgba(252,225,237,.94),rgba(255,235,221,.9),rgba(255,255,255,.75)); box-shadow:0 5px 18px rgba(169,47,101,.045); }
.neo-section .neo-number { display:inline-flex; align-items:center; justify-content:center; min-width:34px; height:34px; border-radius:11px; background:linear-gradient(135deg,#D94F87,#F2B08F); color:white; font-weight:850; font-size:.8rem; }
.neo-section .neo-heading { font-size:1.12rem; font-weight:800; color:#432A38; }
.neo-section .neo-kicker { margin-left:auto; font-size:.68rem; font-weight:800; letter-spacing:.09em; text-transform:uppercase; color:#A92F65; background:rgba(255,255,255,.68); border:1px solid #F2C5D7; border-radius:999px; padding:.32rem .6rem; }
.neo-pill { display:inline-block; padding:.32rem .65rem; margin:.65rem .35rem 0 0; border-radius:999px; border:1px solid #F1C2D4; background:rgba(255,255,255,.7); color:#A92F65; font-size:.74rem; font-weight:750; }
.neo-hero { animation:neo-glow 6s ease-in-out infinite; }
@keyframes neo-glow { 0%,100% { box-shadow:0 10px 28px rgba(169,47,101,.08); } 50% { box-shadow:0 14px 34px rgba(169,47,101,.14); } }
@media (prefers-reduced-motion: reduce) { *,*::before,*::after { animation:none !important; scroll-behavior:auto !important; transition:none !important; } }
@media (max-width:700px) { .neo-section { gap:.55rem; padding:.7rem; } .neo-section .neo-kicker { display:none; } .neo-section .neo-heading { font-size:1rem; } }

/* Premium workspace, recording studio, visualization and accessibility polish */
.neo-brand { display:flex;align-items:center;gap:.7rem;padding:.35rem .2rem 1rem; }
.neo-brand-mark { display:flex;align-items:center;justify-content:center;width:43px;height:43px;border-radius:15px;background:linear-gradient(140deg,#D94F87,#F5B99D);color:#fff;font-size:1.35rem;box-shadow:0 7px 18px rgba(169,47,101,.18); }
.neo-brand-name { font-size:1.12rem;font-weight:900;letter-spacing:-.035em;color:#432A38; }
.neo-brand-sub { font-size:.68rem;color:#806A76;letter-spacing:.08em;text-transform:uppercase;font-weight:800; }
.neo-sidebar-label { margin:.5rem 0 .45rem;color:#A92F65;font-size:.68rem;font-weight:850;letter-spacing:.13em;text-transform:uppercase; }
[data-testid="stSidebar"] .stButton > button { text-align:left;min-height:3.05rem;border-radius:13px; }
[data-testid="stSidebar"] .stButton > button[kind="primary"] { background:linear-gradient(120deg,#D94F87,#C7477D); }
.neo-page-meta { display:flex;align-items:center;gap:.55rem;margin:.15rem 0 1rem;color:#806A76;font-size:.82rem; }
.neo-live-dot { width:8px;height:8px;border-radius:50%;background:#D94F87;display:inline-block;box-shadow:0 0 0 4px rgba(217,79,135,.12); }
.neo-studio { padding:1rem 1.1rem;border:1px solid #F0C3D5;border-radius:19px;background:linear-gradient(115deg,rgba(252,225,237,.92),rgba(255,235,221,.9),rgba(255,255,255,.72));margin:.5rem 0 1rem; }
.neo-step { display:inline-flex;align-items:center;gap:.35rem;padding:.36rem .62rem;margin:.25rem .25rem .1rem 0;border-radius:999px;border:1px solid #F0C6D6;background:rgba(255,255,255,.78);font-size:.74rem;font-weight:750;color:#806A76; }
.neo-step.active { background:#FCE1ED;color:#A92F65;border-color:#E7A0BD; }
.neo-wave { height:37px;display:flex;align-items:center;gap:4px;margin:.55rem 0 .1rem; }
.neo-wave span { display:block;width:5px;border-radius:9px;background:linear-gradient(180deg,#D94F87,#F5B99D);animation:neo-wave 1.05s ease-in-out infinite; }
.neo-wave span:nth-child(3n) { animation-delay:-.7s; }.neo-wave span:nth-child(3n + 1) { animation-delay:-.35s; }
@keyframes neo-wave { 0%,100% { height:9px;opacity:.58; } 50% { height:29px;opacity:1; } }
.neo-network-card { border:1px solid #F0C6D6;border-radius:20px;padding:.65rem;background:linear-gradient(145deg,rgba(255,255,255,.92),rgba(252,231,240,.6));overflow:hidden; }
.neo-node { transform-box:fill-box;transform-origin:center;animation:neo-node-pulse 2.8s ease-in-out infinite; }
.neo-node:nth-of-type(2n) { animation-delay:-1.2s; }
.neo-edge { animation:neo-edge-pulse 3.2s ease-in-out infinite; }
@keyframes neo-node-pulse { 0%,100% { opacity:.82; } 50% { opacity:1; } }
@keyframes neo-edge-pulse { 0%,100% { stroke-opacity:.3; } 50% { stroke-opacity:.8; } }
.neo-insight-card { padding:.9rem 1rem;border-radius:16px;border:1px solid #F0C6D6;background:rgba(255,255,255,.8);height:100%; }
.neo-insight-label { font-size:.72rem;color:#806A76;font-weight:750;text-transform:uppercase;letter-spacing:.08em; }
.neo-insight-value { font-size:1.45rem;font-weight:900;color:#A92F65;margin-top:.2rem; }
[data-testid="stCameraInput"],[data-testid="stAudioInput"] { border:1px solid #F0C6D6;border-radius:16px;padding:.65rem;background:rgba(255,255,255,.65); }
[data-testid="stCameraInput"] button,[data-testid="stAudioInput"] button { min-height:2.7rem; }
[data-testid="stProgress"] { border-radius:999px;overflow:hidden; }
.stButton > button:active { transform:scale(.985); }
@media (max-width: 768px) {
  .block-container { padding:1rem .8rem 2rem; }
  .neo-hero { padding:1.15rem !important;border-radius:19px !important; }
  .neo-hero div[style*="font-size:2.35rem"] { font-size:1.8rem !important; }
  .neo-section { flex-wrap:wrap; }
  .neo-section .neo-kicker { margin-left:0; }
  .neo-studio { padding:.85rem; }
  .neo-network-card { padding:.25rem; }
  [data-testid="stMetric"] { padding:.75rem; }
}
@media (prefers-reduced-motion: reduce) {
  .neo-wave span,.neo-node,.neo-edge { animation:none !important; }
}


/* Cute finishing touches: soft bubbles, sparkle accents, and friendlier controls */
.neo-hero::after {
  content: "✦  ♡  ✧";
  position: absolute;
  right: 1.4rem;
  top: 1rem;
  color: #D94F87;
  font-size: 1.35rem;
  letter-spacing: .28rem;
  opacity: .72;
  transform: rotate(7deg);
  pointer-events: none;
}
.neo-hero::before {
  content: "·  ·  ·";
  position: absolute;
  right: 1.5rem;
  bottom: 1rem;
  color: #E6A27F;
  font-size: 1.5rem;
  letter-spacing: .42rem;
  opacity: .75;
  pointer-events: none;
}
.neo-pill { box-shadow: 0 3px 9px rgba(169,47,101,.045); transition: transform .18s ease, background .18s ease; }
.neo-pill:hover { transform: translateY(-2px) rotate(-1deg); background: #fff !important; }
.neo-brand-mark { border: 2px solid rgba(255,255,255,.72); outline: 3px solid rgba(217,79,135,.09); }
[data-testid="stMetric"] { position: relative; overflow: hidden; }
[data-testid="stMetric"]::after {
  content: "✧";
  position: absolute;
  right: .75rem;
  top: .35rem;
  color: #E7A1BD;
  font-size: 1rem;
  opacity: .7;
  pointer-events: none;
}
.stButton > button, .stFormSubmitButton > button { border-radius: 999px; }
[data-testid="stSidebar"] .stButton > button { border-radius: 15px; }
.stRadio [role="radio"] { border: 1px solid transparent; }
.stRadio [role="radio"][aria-checked="true"] { background: white; border-color: #E7A0BD; box-shadow: 0 4px 12px rgba(169,47,101,.08); }
.neo-step { box-shadow: 0 2px 7px rgba(169,47,101,.035); }
.neo-insight-card { box-shadow: 0 6px 16px rgba(169,47,101,.045); transition: transform .18s ease, box-shadow .18s ease; }
.neo-insight-card:hover { transform: translateY(-2px); box-shadow: 0 10px 22px rgba(169,47,101,.09); }
[data-testid="stCameraInput"], [data-testid="stAudioInput"] { box-shadow: 0 5px 14px rgba(169,47,101,.035); }
[data-testid="stExpander"] { box-shadow: 0 4px 12px rgba(169,47,101,.025); }
@media (max-width: 700px) {
  .neo-hero::after { right: .8rem; top: .55rem; font-size: 1rem; }
  .neo-hero::before { right: .8rem; bottom: .45rem; font-size: 1.1rem; }
}

</style>
""", unsafe_allow_html=True)


# Optional shared-password gate. Configure APP_PASSWORD in Streamlit secrets.
# This is a simple access gate, not individual user accounts or OAuth.
import hmac

def require_app_password():
    try:
        configured_password = st.secrets.get("APP_PASSWORD", "")
    except Exception:
        configured_password = ""

    if not configured_password:
        # Gate is optional: with no APP_PASSWORD secret the app is open.
        with st.sidebar:
            st.caption("🔓 Open access. Set APP_PASSWORD in Streamlit secrets to require a shared password.")
        return

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


init_state()

st.markdown("""
<div class="neo-hero" style="position:relative; overflow:hidden; padding: 1.65rem 1.8rem; margin: .2rem 0 1.25rem; border-radius: 24px;
background: linear-gradient(115deg, #FCE1ED 0%, #FFEBDD 58%, #FFF7F3 100%);
border: 1px solid #F3C2D6; box-shadow: 0 10px 28px rgba(169,47,101,.08);">
  <div style="font-size:.78rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase;
  color:#A92F65; margin-bottom:.45rem;">Emotion • Learning • Adaptation</div>
  <div style="font-size:2.35rem; line-height:1.12; font-weight:850; color:#432A38;">
    🧠 NEUROENDO
  </div>
  <div style="font-size:1rem; margin-top:.65rem; color:#806A76;">
    A biologically inspired, non-quantum multimodal learning demonstrator
  </div>
  <div style="margin-top:.8rem;"><span class="neo-pill">✦ MULTIMODAL AI</span><span class="neo-pill">♡ EMOTION CUES</span><span class="neo-pill">↗ ADAPTIVE LEARNING</span></div>
</div>
""", unsafe_allow_html=True)
st.markdown(
    "Your face, voice and words are turned into emotion scores. Those scores drive a small Hebbian "
    "network that picks how to reply, and a simulated endocrine system — reacting to your emotion and "
    "to whether its replies land — shapes both the reply and how fast the network learns."
)


# Premium sidebar navigation and persistent endocrine controls.
if "active_page" not in st.session_state:
    st.session_state.active_page = "Emotion Studio"

with st.sidebar:
    st.markdown("""
    <div class="neo-brand">
      <div class="neo-brand-mark">🧠</div>
      <div><div class="neo-brand-name">NEUROENDO</div><div class="neo-brand-sub">Emotion · Learning · Adaptation</div></div>
    </div>
    <div class="neo-sidebar-label">Workspace</div>
    """, unsafe_allow_html=True)
    pages = [
        ("Emotion Studio", "♡", "Capture & learn"),
        ("Model Insights", "⌘", "Weights & history"),
        ("A/B Experiment", "⇄", "Compare learning"),
    ]
    for page_name, page_icon, page_label in pages:
        selected = st.session_state.active_page == page_name
        if st.button(
            f"{page_icon}   {page_name}  {'✓' if selected else ''}",
            key="nav_" + page_name,
            use_container_width=True,
            type="primary" if selected else "secondary",
            help=page_label,
        ):
            st.session_state.active_page = page_name
            st.rerun()
    st.caption("Choose a workspace to keep each task focused.")
    st.divider()
    st.markdown('<div class="neo-sidebar-label">Artificial endocrine controller</div>', unsafe_allow_html=True)
    st.write("Simulated regulatory variables, updated automatically from your detected emotion and your feedback. Not measured hormones.")
    endo_c1, endo_c2 = st.columns(2)
    endo_c1.metric("Reward D", f"{st.session_state.reward:.2f}")
    endo_c2.metric("Stress S", f"{st.session_state.stress:.2f}")
    mod = endocrine_modulation(st.session_state.reward, st.session_state.stress)
    st.metric("Plasticity g(H)", f"{mod:.2f}×")
    st.caption("g(H) = clip(1 + 2S − D, 0.2, 2.0): stress speeds re-learning, sustained reward consolidates.")
    if st.button("Reset network, hormones and history", use_container_width=True):
        reset_state()
        st.rerun()

st.markdown(
    f'<div class="neo-page-meta"><span class="neo-live-dot"></span>Workspace / <strong style="color:#A92F65;">{st.session_state.active_page}</strong></div>',
    unsafe_allow_html=True,
)


if st.session_state.active_page == "Emotion Studio":
    mode = st.radio("Input mode", ["Type a message", "Record with camera + mic"], horizontal=True)

    if mode == "Type a message":
        st.caption("Type how you're feeling. The text goes through the same emotion model used for transcribed speech.")
        typed_text = st.text_area("Your message", "I got scolded at work today and I feel awful.", height=80)
        typed_arousal = st.slider("How intense / energetic does this feel?", 0.0, 1.0, 0.4, 0.05,
                                  help="Stands in for the acoustic arousal cue that a voice recording would provide.")
        if st.button("Analyze message", type="primary"):
            with st.spinner("Reading the emotion in your message…"):
                text_scores, text_notes = infer_text_emotion(typed_text)
            combined = combine_emotion_scores(text_scores=text_scores, audio_arousal=typed_arousal)
            uncertainty = estimate_uncertainty(text_scores=text_scores, combined=combined, face_detected=False)
            uncertainty = max(0.0, uncertainty - 0.2)  # typed input: no face expected, don't penalise it
            st.session_state.emotion_scores = combined
            st.session_state.emotion_notes = text_notes
            st.session_state.transcript = typed_text.strip()
            st.session_state.last_features = {"emotion_scores": combined, "arousal": typed_arousal, "uncertainty": uncertainty}
            st.session_state.last_notes = [f"Typed message with intensity {typed_arousal:.2f}."]
            st.session_state.feedback_done = False
            st.session_state.reply_variant += 1
            st.rerun()

    if mode == "Record with camera + mic":
        st.markdown("""
        <div class="neo-studio">
          <div style="font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;font-weight:850;color:#A92F65;">Live recording studio</div>
          <div style="font-size:1.22rem;font-weight:850;color:#432A38;margin:.25rem 0;">Record your face + voice ✨</div>
          <div style="color:#806A76;font-size:.88rem;">Allow camera and microphone access, press START, speak naturally, then press STOP. NEUROENDO samples your video frames and analyzes the recorded microphone audio.</div>
          <span class="neo-step active">01 · Record</span><span class="neo-step">02 · Stop</span><span class="neo-step">03 · Analyze & learn</span>
        </div>
        """, unsafe_allow_html=True)
        if "live_capture_buffer" not in st.session_state:
            st.session_state.live_capture_buffer = LiveCaptureBuffer()
        capture_buffer = st.session_state.live_capture_buffer
        webrtc_ctx = webrtc_streamer(
            key="neuroendo-live-recorder",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={
                "video": {"width": {"ideal": 480, "max": 640}, "height": {"ideal": 360, "max": 480}, "frameRate": {"ideal": 10, "max": 12}},
                "audio": {"echoCancellation": True, "noiseSuppression": True, "autoGainControl": True},
            },
            video_processor_factory=lambda: WebcamVideoProcessor(capture_buffer),
            audio_processor_factory=lambda: MicrophoneAudioProcessor(capture_buffer),
            async_processing=True,
        )
        st.caption("Low-bandwidth recording mode: up to 480 px wide at 10–12 FPS. Audio capture is capped at 30 seconds; about one frame per second is sampled for expression cues.")
        if webrtc_ctx.state.playing:
            st.info("Recording is active. Speak clearly, then press STOP in the recorder above.")
        captured_frames, captured_audio = capture_buffer.snapshot()
        st.caption(f"Captured so far: {len(captured_frames)} sampled frames · {'audio ready' if captured_audio else 'waiting for microphone audio'}")
        action_col1, action_col2 = st.columns(2)
        with action_col1:
            analyze_recording = st.button("Analyze recorded inputs", type="primary", disabled=webrtc_ctx.state.playing)
        with action_col2:
            clear_recording = st.button("Clear recording")
        if clear_recording:
            capture_buffer.clear()
            st.session_state.last_features = None
            st.session_state.emotion_scores = None
            st.session_state.transcript = ""
            st.rerun()
        if analyze_recording:
            captured_frames, captured_audio = capture_buffer.snapshot()
            if not captured_frames:
                st.warning("No video frames were captured. Start recording, allow camera access, record for a few seconds, and stop.")
            else:
                with st.spinner("Analyzing recorded face and microphone inputs…"):
                    visual_results = [visual_features(frame) for frame in captured_frames]
                    face_detected = any(result[1] < 1.0 for result in visual_results)
                    vnotes = list(dict.fromkeys(note for result in visual_results for note in result[2]))
                    face_score_sets, face_notes = [], []
                    for frame in captured_frames:
                        frame_scores, frame_note = infer_facial_emotion(frame)
                        if frame_scores:
                            face_score_sets.append(frame_scores)
                        face_notes.extend(frame_note)
                    face_scores = average_emotion_scores(face_score_sets)
                    audio_cue, anot, speech_notes, text_notes = 0.0, [], [], []
                    text_scores = None
                    st.session_state.transcript = ""
                    if captured_audio:
                        audio_cue, anot = audio_features(captured_audio)
                        transcript, speech_notes = transcribe_audio(captured_audio)
                        st.session_state.transcript = transcript
                        text_scores, text_notes = infer_text_emotion(transcript) if transcript else (None, [])
                    else:
                        text_notes = ["No microphone audio was captured; speech and acoustic analysis were skipped."]
                    combined = combine_emotion_scores(text_scores=text_scores, face_scores=face_scores, audio_arousal=audio_cue)
                    uncertainty = estimate_uncertainty(text_scores=text_scores, face_scores=face_scores,
                                                       combined=combined, face_detected=face_detected)
                    st.session_state.emotion_scores = combined
                    st.session_state.emotion_notes = speech_notes + text_notes + list(dict.fromkeys(face_notes)) + anot
                    st.session_state.emotion_source = "recorded webcam frames + microphone audio"
                    st.session_state.last_features = {"emotion_scores": combined, "arousal": audio_cue, "uncertainty": uncertainty}
                    st.session_state.last_notes = vnotes + anot
                    st.session_state.feedback_done = False
                    st.session_state.reply_variant += 1
                    st.success("Recording analyzed. Review the detected cues and transcript below.")
                    st.rerun()


    current = st.session_state.last_features

    st.divider()
    with st.expander("System architecture", expanded=False):
        st.markdown("""
        ```text
        Webcam frames + microphone audio   (or a typed message)
                 |                |
          face-expression    Whisper speech-to-text -> text-emotion model
              model                     |
                 |                acoustic arousal
                 v                      v
          Fused emotion scores (sadness, anger, fear, joy, surprise, neutral)
                 + arousal + uncertainty  =  feature vector x  (8 values)
                                 |
                                 v
          Hebbian network  x @ W  + stress bias  -->  reply strategy + reply text
                 ^                                         |
                 |  dW = eta * g(H) * x (y - p)^T          v
                 +---------------  your feedback (accept / correct)
                                 |
          Artificial endocrine system:
            Reward D  <- accepted replies, positive emotion
            Stress S  <- corrected replies, negative / high-arousal emotion
            g(H) = clip(1 + 2S - D)  sets plasticity;  S biases toward careful replies
        ```
        """)
        st.caption("All processing is classical and non-quantum. D and S are simulated controller states, not measured hormones.")

    st.markdown('<div class="neo-section"><span class="neo-number">01</span><span class="neo-heading">Learning &amp; response</span><span class="neo-kicker">Live interaction</span></div>', unsafe_allow_html=True)

    if current is None:
        st.info("Type a message and click **Analyze message**, or record with your camera and microphone.")
    else:
        x = build_features(current["emotion_scores"], current["arousal"], current["uncertainty"])
        scores, probs, pred_idx = predict(x, st.session_state.W, S=st.session_state.stress)
        pred = RESPONSES[pred_idx]
        st.session_state.last_x = x
        st.session_state.last_pred = pred
        st.session_state.last_probs = probs

        a, b, c = st.columns(3)
        top_emotion = max(current["emotion_scores"], key=current["emotion_scores"].get)
        a.metric("Leading emotion", top_emotion.title())
        b.metric("Arousal cue", f'{current["arousal"]:.2f}')
        c.metric("Uncertainty", f'{current["uncertainty"]:.2f}')

        if st.session_state.transcript:
            st.markdown("#### What you said")
            st.write(st.session_state.transcript)
            st.caption("Check the transcript for speech-recognition mistakes; errors can affect the emotion estimate.")

        if st.session_state.emotion_scores:
            emotion_scores = st.session_state.emotion_scores
            best_emotion = max(emotion_scores, key=emotion_scores.get)
            best_score = float(emotion_scores[best_emotion])
            emotion_art = {"sadness":("☁️","#8E77B5"),"anger":("🔥","#D94F70"),"fear":("🌧️","#6C91B8"),"joy":("☀️","#E7A34E"),"surprise":("✨","#B17CC4"),"neutral":("🌸","#D96F9B")}
            st.markdown("#### Your emotion snapshot")
            emoji, accent = emotion_art.get(best_emotion,("💗","#D94F87"))
            st.markdown(f"""<div style="display:flex;align-items:center;gap:1rem;padding:1rem 1.1rem;border-radius:20px;border:1px solid #F0C3D5;background:linear-gradient(110deg,#FCE1ED,#FFEBDD 72%,#FFFFFF);box-shadow:0 8px 22px rgba(169,47,101,.07);margin:.35rem 0 1rem;"><div style="font-size:2.3rem;">{emoji}</div><div style="flex:1;"><div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.12em;font-weight:800;color:#A92F65;">Leading emotion cue</div><div style="font-size:1.55rem;font-weight:850;color:#432A38;">{best_emotion.title()}</div><div style="font-size:.88rem;color:#806A76;">Estimated from the available signals</div></div><div style="font-size:1.45rem;font-weight:850;color:{accent};">{best_score:.0%}</div></div>""", unsafe_allow_html=True)
            st.markdown("**Emotion mix**")
            top_emotions = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)[:3]
            emotion_cols = st.columns(len(top_emotions))
            for emotion_col, (label, score) in zip(emotion_cols, top_emotions):
                icon_emoji, icon_color = emotion_art.get(label,("💗","#D94F87"))
                with emotion_col:
                    st.markdown(f"""<div style="height:100%;padding:.8rem;border-radius:16px;border:1px solid #F2D0DD;background:rgba(255,255,255,.78);text-align:center;"><div style="font-size:1.45rem;">{icon_emoji}</div><div style="font-weight:800;color:#432A38;margin:.25rem 0;">{label.title()}</div><div style="font-size:1.15rem;font-weight:850;color:{icon_color};">{float(score):.0%}</div></div>""", unsafe_allow_html=True)
            st.markdown("**Relative signal strength**")
            st.progress(max(0.0, min(1.0, best_score)))
            st.caption("This is a relative model score, not a calibrated probability or diagnosis. Visible expressions and spoken words do not prove how someone feels.")
            st.markdown("**Signal distribution**")
            emotion_chart_df = pd.DataFrame({
                "Relative score": emotion_scores
            }).sort_values("Relative score", ascending=True)
            st.bar_chart(emotion_chart_df, horizontal=True)
            st.caption("Scores summarize model outputs; they should be interpreted as uncertain cues, not ground truth.")
            with st.expander("How this estimate was formed"):
                for note in st.session_state.emotion_notes:
                    st.write(f"- {note}")

        if st.session_state.last_notes:
            st.write("**Extracted feature notes**")
            for note in st.session_state.last_notes:
                st.write(f"- {note}")

        left, right = st.columns(2)
        with left:
            st.markdown("#### Reply strategy scores")
            score_df = pd.DataFrame({"Response": RESPONSES, "Score": scores})
            st.bar_chart(score_df.set_index("Response"))
            st.caption("Scores = learned associations (x · W) plus the stress bias. Higher wins.")
        with right:
            st.markdown("#### NEUROENDO's reply")
            st.markdown(f"**Strategy:** {pred}")
            reply = make_response(pred, current["emotion_scores"], current["uncertainty"],
                                  st.session_state.reply_variant)
            st.markdown(f'<div class="neo-insight-card" style="font-size:1.02rem;line-height:1.5;">{reply}</div>',
                        unsafe_allow_html=True)
            r_emo, s_emo = emotional_drive(x)
            st.caption(f"Your detected state pushes reward by {r_emo:.2f} and stress by {s_emo:.2f}.")

        st.markdown("#### Was that the right kind of reply?")
        if st.session_state.get("feedback_done"):
            st.success("Thanks — feedback learned. Send a new message or recording to continue.")
        else:
            st.write("Accept it, or pick the reply style you'd have preferred. The network and the endocrine system both learn from this.")
            fb_left, fb_right = st.columns([1, 2])
            with fb_left:
                accept = st.button("👍 Good reply", type="primary", use_container_width=True)
            with fb_right:
                alternatives = [r for r in RESPONSES if r != pred]
                chosen_alt = st.selectbox("I'd have preferred…", alternatives, key="feedback_target")
                correct = st.button("Teach this instead", use_container_width=True)
            if accept or correct:
                target = pred if accept else chosen_alt
                accepted = bool(accept)
                modulation = endocrine_modulation(st.session_state.reward, st.session_state.stress)
                st.session_state.W, delta = hebbian_update(
                    st.session_state.W, x, RESPONSES.index(target), probs, factor=modulation
                )
                st.session_state.reward, st.session_state.stress = endocrine_step(
                    st.session_state.reward, st.session_state.stress, x, accepted
                )
                st.session_state.history.append({
                    "Interaction": len(st.session_state.history) + 1,
                    "Emotion": top_emotion,
                    "Reply": pred,
                    "Target": target,
                    "Accepted": accepted,
                    "Reward D": st.session_state.reward,
                    "Stress S": st.session_state.stress,
                    "Plasticity g": modulation,
                    "Weight change norm": float(np.linalg.norm(delta)),
                })
                st.session_state.feedback_done = True
                st.rerun()


if st.session_state.active_page == "Model Insights":
    st.divider()
    st.markdown('<div class="neo-section"><span class="neo-number">02</span><span class="neo-heading">Inside the model</span><span class="neo-kicker">Weights &amp; history</span></div>', unsafe_allow_html=True)

    w_df = pd.DataFrame(st.session_state.W, index=FEATURES, columns=RESPONSES)
    st.markdown("**Learned association network**")
    st.markdown(render_network_svg(st.session_state.W), unsafe_allow_html=True)
    st.caption("The diagram is animated for visual clarity. Connection thickness reflects the current learned weight magnitude.")
    st.markdown("**Weight matrix W** — rows are input features (emotion scores, arousal, uncertainty), columns are reply strategies. It starts from a weak default pairing (sadness→Supportive, anger→De-escalating, fear→Calming, joy→Upbeat, surprise→Curious) and adapts to your feedback.")
    st.dataframe(w_df.style.format("{:.3f}"), use_container_width=True)

    if st.session_state.history:
        hist = pd.DataFrame(st.session_state.history)
        hist["Acceptance (rolling 5)"] = hist["Accepted"].astype(float).rolling(5, min_periods=1).mean()
        st.markdown("**Learning history**")
        st.dataframe(hist, use_container_width=True)
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("Endocrine state over interactions")
            st.line_chart(hist.set_index("Interaction")[["Reward D", "Stress S", "Plasticity g"]])
        with chart_col2:
            st.markdown("How often replies were accepted")
            st.line_chart(hist.set_index("Interaction")[["Acceptance (rolling 5)"]])
    else:
        st.info("No learning updates yet. Analyze a message or recording, then accept or correct the reply.")

    with st.expander("Model equations and limitations"):
        st.markdown(r"""
**Input.** \(x = [p_{sad}, p_{anger}, p_{fear}, p_{joy}, p_{surprise}, p_{neutral}, a, u]\) — fused emotion scores, acoustic arousal \(a\), uncertainty \(u\).

**Reply choice.** \(s = xW + b(S)\), where \(b\) adds \(0.15S\) to Supportive, Calming and De-escalating. Strategy \(=\arg\max s\); \(p=\mathrm{softmax}(3s)\).

**Three-factor Hebbian update with anti-Hebbian term.**
\[ \Delta W = \eta\, g(H)\, x\,(y - p)^T,\qquad \eta = 0.25 \]
\(x y^T\) strengthens the link to the reply you wanted; \(-x p^T\) weakens links to replies the network over-favours, so weights can go negative.

**Endocrine system.** Leaky states (rate 0.25):
\[ D \leftarrow 0.75D + 0.25\,(0.7\,\text{accepted} + 0.3\,r_{emo}),\quad S \leftarrow 0.75S + 0.25\,(0.7\,\text{corrected} + 0.3\,s_{emo}) \]
\(r_{emo}\) comes from joy/surprise; \(s_{emo}\) from sadness, anger, fear and arousal.
\[ g(H) = \mathrm{clip}(1 + 2S - D,\ 0.2,\ 2.0) \]

**Limitations.** Emotion scores come from pretrained models (or a keyword fallback for text) and are uncertain cues, not ground truth. Replies are template-based per strategy, not generated by a language model. D and S are engineering variables inspired by neuromodulation, not a biological model.
""")


if st.session_state.active_page == "A/B Experiment":
    st.divider()
    st.markdown('<div class="neo-section"><span class="neo-number">03</span><span class="neo-heading">A/B learning experiment</span><span class="neo-kicker">Compare variants</span></div>', unsafe_allow_html=True)
    st.write(
        "Both variants interact online with the same simulated users: noisy emotion scores in, a reply out, "
        "then feedback. The simulated users prefer some non-default replies, give 10% random feedback, and "
        "change their preferences half-way through. Both start from the same prior and use the same "
        "Hebbian/anti-Hebbian rule and base learning rate. Variant B adds the endocrine system."
    )
    ab_col1, ab_col2 = st.columns([1, 2])
    with ab_col1:
        ab_samples = st.select_slider("Interactions per user", options=[120, 240, 480], value=240, key="ab_samples")
        ab_seeds = st.select_slider("Simulated users", options=[10, 20, 40], value=20, key="ab_seeds")
        if st.button("Run A/B comparison", type="primary"):
            with st.spinner("Simulating users…"):
                st.session_state.ab_results = run_ab_experiment(seeds=int(ab_seeds), samples=int(ab_samples))
    with ab_col2:
        st.caption("Accuracy = how often the reply chosen *before* feedback matched what the user wanted. Seeds 0…N−1, so results are repeatable.")

    if st.session_state.ab_results is not None:
        ab = st.session_state.ab_results
        diff = ab["endo_accuracy"] - ab["plain_accuracy"]
        a1, a2, a3 = st.columns(3)
        a1.metric("A · Plain Hebbian", f"{ab['plain_accuracy']:.1%}", help=f"± {ab['plain_std']:.1%} across users")
        a2.metric("B · Endocrine", f"{ab['endo_accuracy']:.1%}", delta=f"{diff:+.1%}", help=f"± {ab['endo_std']:.1%} across users")
        a3.metric("B wins", f"{ab['endo_wins']} / {ab['seeds']} users")
        b1, b2, b3 = st.columns(3)
        b1.metric("After preference shift · A", f"{ab['plain_post_shift']:.1%}", help="First 40 interactions after the shift")
        b2.metric("After preference shift · B", f"{ab['endo_post_shift']:.1%}",
                  delta=f"{ab['endo_post_shift'] - ab['plain_post_shift']:+.1%}")
        b3.metric(f"Plain, best hand-tuned η={ab['best_fixed_eta']:g}", f"{ab['best_fixed_accuracy']:.1%}")

        start = ab["curve_window"]
        curve_df = pd.DataFrame({
            "Interaction": np.arange(start, start + len(ab["curve_plain"])),
            "A · Plain": ab["curve_plain"],
            "B · Endocrine": ab["curve_endo"],
        }).set_index("Interaction")
        st.markdown(f"**Rolling accuracy ({ab['curve_window']}-interaction window, averaged over users)** — preferences change at interaction {ab['shift_at']}.")
        st.line_chart(curve_df)

        if ab["best_fixed_accuracy"] >= ab["endo_accuracy"] - 0.005:
            st.info(
                "Reading the result honestly: at the same base learning rate the endocrine system helps, mainly by "
                "re-learning faster after the preference shift. A plain learner whose rate is hand-tuned with hindsight "
                "reaches about the same accuracy, so the benefit is self-tuning plasticity, not extra information."
            )
        else:
            st.success("The endocrine variant beat both the same-rate and the best hand-tuned plain learner on this run.")
        st.caption("Synthetic users only — this illustrates the learning dynamics and is not evidence of real-world emotional understanding.")

    st.caption("Privacy note: this prototype processes captured media in the app session. Avoid recording other people without their informed consent.")
