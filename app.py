import io
import math
import os
import re
import subprocess
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


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
FEATURES = ["Visual positive cue", "Audio cue", "Uncertainty cue"]

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
    if "transcript" not in st.session_state:
        st.session_state.transcript = ""

def clip(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))

def endocrine_modulation(reward, stress):
    # Proposed engineering rule, not a biological law.
    return clip(1.0 + reward - stress, 0.2, 1.5)


def render_network_svg(weights):
    """Render a lightweight animated SVG of feature-to-response associations."""
    input_nodes = [(78, 70), (78, 145), (78, 220)]
    output_nodes = [(322, 70), (322, 145), (322, 220)]
    feature_labels = ["Visual", "Audio", "Uncertainty"]
    response_labels = ["Supportive", "Neutral", "Upbeat"]
    parts = [
        '<div class="neo-network-card">',
        '<svg viewBox="0 0 400 280" role="img" aria-label="Animated neural association network" style="width:100%;height:auto;display:block;">',
        '<text x="78" y="25" text-anchor="middle" font-size="11" font-weight="800" fill="#A92F65">INPUT FEATURES</text>',
        '<text x="322" y="25" text-anchor="middle" font-size="11" font-weight="800" fill="#A92F65">RESPONSE NODES</text>',
    ]
    for i, (x1, y1) in enumerate(input_nodes):
        for j, (x2, y2) in enumerate(output_nodes):
            value = float(weights[i, j])
            strength = min(1.0, abs(value) / 1.5)
            stroke = "#D94F87" if value >= 0 else "#8E77B5"
            width = 1.1 + 4.2 * strength
            opacity = 0.24 + 0.66 * strength
            parts.append(
                f'<line class="neo-edge" x1="{x1+22}" y1="{y1}" x2="{x2-22}" y2="{y2}" '
                f'stroke="{stroke}" stroke-width="{width:.2f}" stroke-opacity="{opacity:.2f}" stroke-linecap="round"/>'
            )
    for i, (x, y) in enumerate(input_nodes):
        parts.append(f'<circle class="neo-node" cx="{x}" cy="{y}" r="22" fill="#FCE1ED" stroke="#D94F87" stroke-width="1.5"/>')
        parts.append(f'<text x="{x}" y="{y+4}" text-anchor="middle" font-size="9" font-weight="800" fill="#7C3154">{["V","A","U"][i]}</text>')
        parts.append(f'<text x="{x}" y="{y+37}" text-anchor="middle" font-size="9.5" fill="#806A76">{feature_labels[i]}</text>')
    for j, (x, y) in enumerate(output_nodes):
        parts.append(f'<circle class="neo-node" cx="{x}" cy="{y}" r="22" fill="#FFEBDD" stroke="#E6A27F" stroke-width="1.5"/>')
        parts.append(f'<text x="{x}" y="{y+4}" text-anchor="middle" font-size="9" font-weight="800" fill="#8A4E36">{["S","N","U"][j]}</text>')
        parts.append(f'<text x="{x}" y="{y+37}" text-anchor="middle" font-size="9.5" fill="#806A76">{response_labels[j]}</text>')
    parts.append('</svg><div style="display:flex;justify-content:center;gap:1rem;flex-wrap:wrap;font-size:.72rem;color:#806A76;padding:.25rem .2rem .45rem;">'
                 '<span>━ Stronger line = larger learned weight</span><span style="color:#D94F87;">━ Positive</span><span style="color:#8E77B5;">━ Negative</span></div></div>')
    return "".join(parts)


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


EMOTION_LABELS = ["sadness", "anger", "fear", "joy", "surprise", "neutral"]

TEXT_CUES = {
    "sadness": {
        "words": ["sad", "upset", "hurt", "cry", "crying", "lonely", "disappointed",
                  "embarrassed", "ashamed", "guilty", "miserable", "heartbroken",
                  "depressed", "down", "unhappy", "scolded", "rejected", "ignored"],
        "phrases": ["got scolded", "was scolded", "got yelled at", "i feel bad",
                    "feel terrible", "made me cry", "i got in trouble", "feel awful"]
    },
    "anger": {
        "words": ["angry", "mad", "furious", "annoyed", "irritated", "rage", "unfair",
                  "hate", "frustrated", "blamed"],
        "phrases": ["so unfair", "lost my temper", "can't stand", "got blamed"]
    },
    "fear": {
        "words": ["scared", "afraid", "anxious", "anxiety", "worried", "nervous",
                  "panic", "panicking", "overwhelmed", "stress", "stressed", "terrified"],
        "phrases": ["what if", "i am worried", "i'm worried", "under pressure"]
    },
    "joy": {
        "words": ["happy", "great", "excited", "joyful", "proud", "relieved", "love",
                  "wonderful", "amazing", "glad", "delighted", "hopeful"],
        "phrases": ["so happy", "can't wait", "that went well", "feeling good"]
    },
    "surprise": {
        "words": ["surprised", "shocked", "unexpected", "astonished", "unbelievable"],
        "phrases": ["didn't expect", "out of nowhere", "i can't believe"]
    }
}

def classify_text_fallback(text):
    """Explainable keyword fallback; not a trained model."""
    cleaned = re.sub(r"[^a-z0-9' ]+", " ", (text or "").lower())
    scores = {label: 0.0 for label in EMOTION_LABELS}
    evidence = {label: [] for label in EMOTION_LABELS}
    tokens = set(cleaned.split())
    for emotion, cues in TEXT_CUES.items():
        for phrase in cues["phrases"]:
            if phrase in cleaned:
                scores[emotion] += 2.0
                evidence[emotion].append(phrase)
        for word in cues["words"]:
            if word in tokens:
                scores[emotion] += 1.0
                evidence[emotion].append(word)
    if not any(scores.values()):
        scores["neutral"] = 1.0
        return scores, ["No strong emotion keywords found; treat this result as uncertain."]
    total = sum(scores.values())
    scores = {k: v / total for k, v in scores.items()}
    notes = [f"Text cues for {label}: {', '.join(evidence[label][:5])}."
             for label in EMOTION_LABELS if evidence[label]]
    return scores, notes

def transcribe_audio(audio_bytes):
    """Transcribe a recorded WAV clip using a pretrained speech-recognition model."""
    try:
        from scipy.io import wavfile
        from transformers import pipeline

        if "speech_recognition_pipeline" not in st.session_state:
            with st.spinner("Loading speech-to-text model for first use..."):
                st.session_state.speech_recognition_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-tiny.en"
                )

        sr, data = wavfile.read(io.BytesIO(audio_bytes))
        if data.size == 0:
            return "", ["The recorded audio was empty."]
        if data.ndim > 1:
            data = data.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            info = np.iinfo(data.dtype)
            scale = float(max(abs(info.min), abs(info.max)))
            data = data.astype(np.float32) / scale
        else:
            data = data.astype(np.float32)
            peak = float(np.max(np.abs(data))) if data.size else 0.0
            if peak > 1.0:
                data = data / peak

        result = st.session_state.speech_recognition_pipeline(
            {"raw": data, "sampling_rate": int(sr)}, chunk_length_s=20
        )
        transcript = str(result.get("text", "")).strip()
        if transcript:
            return transcript, ["Speech was transcribed automatically with Whisper Tiny English."]
        return "", ["Speech-to-text returned an empty transcript; emotion-from-words was skipped."]
    except Exception as exc:
        return "", [
            "Automatic speech transcription is unavailable, so spoken-word emotion analysis was skipped.",
            f"Speech-to-text error: {type(exc).__name__}. Check model download, memory, and audio format."
        ]

def infer_text_emotion(text):
    """Use a pretrained text model when available; fall back to transparent keyword cues."""
    if not text or not text.strip():
        return None, ["No text description provided."]
    try:
        from transformers import pipeline
        if "text_emotion_pipeline" not in st.session_state:
            with st.spinner("Loading the text emotion model for first use..."):
                st.session_state.text_emotion_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    top_k=None
                )
        output = st.session_state.text_emotion_pipeline(text[:2000])[0]
        scores = {label: 0.0 for label in EMOTION_LABELS}
        for item in output:
            label = str(item["label"]).lower()
            mapped = {"joy":"joy", "sadness":"sadness", "anger":"anger",
                      "fear":"fear", "surprise":"surprise", "neutral":"neutral",
                      "disgust":"anger"}.get(label)
            if mapped:
                scores[mapped] = max(scores[mapped], float(item["score"]))
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}, [
            "Text analyzed with a pretrained English emotion classifier."
        ]
    except Exception as exc:
        scores, notes = classify_text_fallback(text)
        notes.append(f"Pretrained text model unavailable ({type(exc).__name__}); keyword fallback used.")
        return scores, notes

def infer_facial_emotion(image_bytes):
    """Run a pretrained facial-expression model, or clearly mark it unavailable."""
    try:
        from PIL import Image
        from transformers import pipeline
        if "face_emotion_pipeline" not in st.session_state:
            with st.spinner("Loading the facial-expression model for first use..."):
                st.session_state.face_emotion_pipeline = pipeline(
                    "image-classification", model="trpakov/vit-face-expression"
                )
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        output = st.session_state.face_emotion_pipeline(image)
        scores = {label: 0.0 for label in EMOTION_LABELS}
        for item in output:
            label = str(item["label"]).lower()
            mapped = {"happy":"joy", "happiness":"joy", "sad":"sadness",
                      "sadness":"sadness", "angry":"anger", "anger":"anger",
                      "fear":"fear", "fearful":"fear", "surprise":"surprise",
                      "surprised":"surprise", "neutral":"neutral",
                      "disgust":"anger"}.get(label)
            if mapped:
                scores[mapped] = max(scores[mapped], float(item["score"]))
        total = sum(scores.values()) or 1.0
        return {k: v / total for k, v in scores.items()}, [
            "Facial-expression model estimates visible expression, not internal emotion with certainty."
        ]
    except Exception as exc:
        return None, [
            "Facial-expression model unavailable; the basic smile detector cannot reliably detect sadness.",
            f"Model error: {type(exc).__name__}. Text analysis is still available."
        ]

def combine_emotion_scores(text_scores=None, face_scores=None, audio_arousal=None):
    combined = {label: 0.0 for label in EMOTION_LABELS}
    if text_scores:
        for label, score in text_scores.items():
            combined[label] += 0.65 * float(score)
    if face_scores:
        for label, score in face_scores.items():
            combined[label] += 0.35 * float(score)
    if audio_arousal is not None:
        # Audio energy is only a weak arousal cue; it cannot identify emotion by itself.
        combined["fear"] += 0.025 * float(audio_arousal)
        combined["anger"] += 0.025 * float(audio_arousal)
    total = sum(combined.values())
    if total <= 0:
        return {"sadness": 0, "anger": 0, "fear": 0, "joy": 0, "surprise": 0, "neutral": 1}
    return {k: v / total for k, v in combined.items()}

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


def extract_video_frames(video_bytes, max_frames=5):
    """Sample a few frames from an uploaded video."""
    temp_path = None
    try:
        import cv2
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            temp_video.write(video_bytes)
            temp_path = temp_video.name
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            return [], ["Could not open this video. Try an MP4 or MOV file."]
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return [], ["No readable frames were found in the video."]
        indexes = np.linspace(0, frame_count - 1, min(max_frames, frame_count), dtype=int)
        frames = []
        for index in sorted(set(int(i) for i in indexes)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok:
                encoded_ok, encoded = cv2.imencode(".jpg", frame)
                if encoded_ok:
                    frames.append(encoded.tobytes())
        cap.release()
        return frames, [f"Sampled {len(frames)} frame(s) across the uploaded video."]
    except Exception as exc:
        return [], [f"Video frame extraction failed ({type(exc).__name__})."]
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def extract_video_audio(video_bytes):
    """Extract a mono WAV audio track from a video using bundled FFmpeg."""
    video_path = audio_path = None
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as video_file:
            video_file.write(video_bytes)
            video_path = video_file.name
        audio_handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio_path = audio_handle.name
        audio_handle.close()
        result = subprocess.run(
            [ffmpeg, "-y", "-i", video_path, "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", audio_path],
            capture_output=True, text=True, timeout=90
        )
        if result.returncode != 0 or not os.path.exists(audio_path) or os.path.getsize(audio_path) < 44:
            return None, ["No readable audio track was found in the video."]
        with open(audio_path, "rb") as audio_file:
            return audio_file.read(), ["Extracted the video's audio track for acoustic features and speech transcription."]
    except subprocess.TimeoutExpired:
        return None, ["Audio extraction timed out. Try a shorter video."]
    except Exception as exc:
        return None, [f"Could not extract video audio ({type(exc).__name__})."]
    finally:
        for path in (video_path, audio_path):
            if path and os.path.exists(path):
                os.remove(path)


def average_emotion_scores(score_sets):
    """Average emotion scores from sampled video frames."""
    valid = [scores for scores in score_sets if scores]
    if not valid:
        return None
    return {label: float(np.mean([scores.get(label, 0.0) for scores in valid]))
            for label in EMOTION_LABELS}


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
    "Video frames and the video's audio track feed a small Hebbian association network. "
    "A simulated artificial endocrine system modulates how strongly the network learns."
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
    st.write("Simulated regulatory variables — not measured biological hormones.")
    st.session_state.reward = st.slider("Reward-like signal (D)", 0.0, 1.0, float(st.session_state.reward), 0.05)
    st.session_state.stress = st.slider("Stress-like signal (S)", 0.0, 1.0, float(st.session_state.stress), 0.05)
    mod = endocrine_modulation(st.session_state.reward, st.session_state.stress)
    st.metric("Learning modulation g(H)", f"{mod:.2f}×")
    st.caption("Rule: g(H) = clip(1 + D − S, 0.2, 1.5)")
    if st.button("Reset learned weights and history", use_container_width=True):
        st.session_state.W = np.zeros((3, 3), dtype=float)
        st.session_state.history = []
        st.session_state.last_x = None
        st.session_state.last_features = None
        st.session_state.last_pred = None
        st.session_state.last_notes = []
        st.session_state.emotion_scores = None
        st.session_state.emotion_notes = []
        st.session_state.transcript = ""
        st.session_state.ab_results = None
        st.rerun()

st.markdown(
    f'<div class="neo-page-meta"><span class="neo-live-dot"></span>Workspace / <strong style="color:#A92F65;">{st.session_state.active_page}</strong></div>',
    unsafe_allow_html=True,
)


if st.session_state.active_page == "Emotion Studio":
    mode = st.radio("Input mode", ["Demo / simulated signals", "Upload a video"], horizontal=True)

    st.caption("Upload one video. NEUROENDO samples frames for visible-expression cues and extracts its audio track for speech and acoustic cues.")

    if mode == "Demo / simulated signals":
        st.caption("Adjust these simulated cues to explore how the learning system responds without uploading a video.")
        demo_col1, demo_col2, demo_col3 = st.columns(3)
        with demo_col1:
            visual_cue = st.slider("Visual cue", 0.0, 1.0, 0.55, 0.05, help="Simulated expression-related cue.")
        with demo_col2:
            audio_cue = st.slider("Audio arousal cue", 0.0, 1.0, 0.35, 0.05, help="Simulated acoustic arousal cue.")
        with demo_col3:
            visual_uncertainty = st.slider("Signal uncertainty", 0.0, 1.0, 0.20, 0.05, help="Simulated uncertainty in the visual signal.")
        notes = ["Demo mode is using simulated visual, audio, and uncertainty values."]
        
    if mode == "Upload a video":
        st.markdown("""
        <div class="neo-studio">
          <div style="font-size:.72rem;letter-spacing:.12em;text-transform:uppercase;font-weight:850;color:#A92F65;">Video analysis studio</div>
          <div style="font-size:1.22rem;font-weight:850;color:#432A38;margin:.25rem 0;">One video, visual + audio cues ✨</div>
          <div style="color:#806A76;font-size:.88rem;">Upload a short video with a visible face and spoken audio. The app samples frames and analyzes the extracted sound.</div>
          <span class="neo-step active">01 · Upload</span><span class="neo-step">02 · Analyze</span><span class="neo-step">03 · Review & learn</span>
        </div>
        """, unsafe_allow_html=True)
        video_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "mov", "m4v", "webm", "avi"],
            help="Short clips work best. Video processing is sample-based analysis, not live tracking."
        )
        if video_file is not None:
            st.video(video_file)
            st.caption(f"Selected: {video_file.name} · {video_file.size / (1024 * 1024):.1f} MB")
        if st.button("Analyze video", type="primary"):
            if video_file is None:
                st.warning("Upload a video first.")
            else:
                video_bytes = video_file.getvalue()
                with st.spinner("Sampling video frames and extracting audio…"):
                    frame_bytes, frame_notes = extract_video_frames(video_bytes)
                    audio_bytes, audio_extract_notes = extract_video_audio(video_bytes)
                    if not frame_bytes:
                        st.error("I couldn't read frames from this video. Try a short MP4 or MOV clip.")
                    else:
                        visual_results = [visual_features(frame) for frame in frame_bytes]
                        visual_cue = float(np.mean([result[0] for result in visual_results]))
                        visual_uncertainty = float(np.mean([result[1] for result in visual_results]))
                        vnotes = frame_notes + [note for result in visual_results for note in result[2]]
                        face_score_sets = []
                        face_notes = []
                        for frame in frame_bytes:
                            frame_scores, frame_note = infer_facial_emotion(frame)
                            if frame_scores:
                                face_score_sets.append(frame_scores)
                            face_notes.extend(frame_note)
                        face_scores = average_emotion_scores(face_score_sets)

                        st.session_state.transcript = ""
                        audio_cue = 0.0
                        anot = []
                        speech_notes = []
                        text_scores = None
                        if audio_bytes:
                            audio_cue, anot = audio_features(audio_bytes)
                            transcript, speech_notes = transcribe_audio(audio_bytes)
                            st.session_state.transcript = transcript
                            text_scores, text_notes = infer_text_emotion(transcript) if transcript else (None, [])
                        else:
                            text_notes = ["No audio track was available; spoken-word and acoustic analysis were skipped."]

                        st.session_state.emotion_scores = combine_emotion_scores(
                            text_scores=text_scores, face_scores=face_scores, audio_arousal=audio_cue
                        )
                        st.session_state.emotion_notes = speech_notes + text_notes + face_notes + anot + audio_extract_notes
                        st.session_state.emotion_source = "sampled video frames + extracted audio"
                        st.session_state.last_features = {
                            "visual": visual_cue, "audio": audio_cue, "uncertainty": visual_uncertainty
                        }
                        st.session_state.last_notes = vnotes + anot + audio_extract_notes
                        if st.session_state.transcript:
                            st.success("Video analyzed! Sampled frames and spoken words are ready to review.")
                        elif audio_bytes:
                            st.warning("Video frames and audio were analyzed, but no speech transcript was available.")
                        else:
                            st.warning("Video frames were analyzed, but no usable audio track was found.")


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
        Uploaded video (sampled frames + audio) or demo signals
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

    st.markdown('<div class="neo-section"><span class="neo-number">01</span><span class="neo-heading">Learning &amp; response</span><span class="neo-kicker">Live interaction</span></div>', unsafe_allow_html=True)

    if current is None:
        st.info("Select Demo mode or upload a video and click Analyze video.")
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

        if st.session_state.transcript:
            st.markdown("#### Automatically detected speech")
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


if st.session_state.active_page == "Model Insights":
    st.divider()
    st.markdown('<div class="neo-section"><span class="neo-number">02</span><span class="neo-heading">Inside the model</span><span class="neo-kicker">Weights &amp; history</span></div>', unsafe_allow_html=True)

    w_df = pd.DataFrame(st.session_state.W, index=FEATURES, columns=RESPONSES)
    st.markdown("**Learned association network**")
    st.markdown(render_network_svg(st.session_state.W), unsafe_allow_html=True)
    st.caption("The diagram is animated for visual clarity. Connection thickness reflects the current learned weight magnitude.")
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


if st.session_state.active_page == "A/B Experiment":
    st.divider()
    st.markdown('<div class="neo-section"><span class="neo-number">03</span><span class="neo-heading">A/B learning experiment</span><span class="neo-kicker">Compare variants</span></div>', unsafe_allow_html=True)
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
        a1.metric("Model A · Ordinary", f"{ab['plain_accuracy']:.1%}")
        a2.metric("Model B · Endocrine", f"{ab['endo_accuracy']:.1%}")
        difference = ab["endo_accuracy"] - ab["plain_accuracy"]
        a3.metric("Difference · B − A", f"{difference:+.1%}", delta=f"{difference:+.1%}")
        result_left, result_right = st.columns(2)
        with result_left:
            st.markdown(f'<div class="neo-insight-card"><div class="neo-insight-label">Ordinary Hebbian · Model A</div><div class="neo-insight-value">{ab["plain_accuracy"]:.1%}</div><div style="height:8px;border-radius:99px;background:#F7E2EB;overflow:hidden;margin-top:.6rem;"><div style="width:{100*ab["plain_accuracy"]:.1f}%;height:100%;background:linear-gradient(90deg,#D94F87,#E88FB2);border-radius:99px;"></div></div><div style="font-size:.75rem;color:#806A76;margin-top:.45rem;">Baseline update rule</div></div>', unsafe_allow_html=True)
        with result_right:
            st.markdown(f'<div class="neo-insight-card"><div class="neo-insight-label">Endocrine-modulated · Model B</div><div class="neo-insight-value">{ab["endo_accuracy"]:.1%}</div><div style="height:8px;border-radius:99px;background:#F7E2EB;overflow:hidden;margin-top:.6rem;"><div style="width:{100*ab["endo_accuracy"]:.1f}%;height:100%;background:linear-gradient(90deg,#F0A57F,#D94F87);border-radius:99px;"></div></div><div style="font-size:.75rem;color:#806A76;margin-top:.45rem;">Feedback-modulated update rule</div></div>', unsafe_allow_html=True)
        if abs(difference) < 1e-9:
            st.info("Both variants reached the same test accuracy on this synthetic split.")
        elif difference > 0:
            st.success("Model B scored higher on this synthetic split. This does not establish real-world superiority.")
        else:
            st.info("Model A scored higher on this synthetic split. Endocrine modulation changes update magnitude, not the information available.")
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
