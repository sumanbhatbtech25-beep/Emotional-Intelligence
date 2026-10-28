"""Learning state, endocrine modulation, and Hebbian experiment utilities."""
import numpy as np
import streamlit as st

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
