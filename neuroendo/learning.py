"""Emotion-driven association network with a simulated artificial endocrine system.

Data flow
---------
emotion scores + arousal + uncertainty  ->  feature vector x (8 values)
x @ W  + endocrine bias(S)              ->  response scores -> softmax p -> reply
user feedback (target y)                ->  three-factor Hebbian update of W
detected emotion + feedback outcome     ->  endocrine state (D, S) -> plasticity g

D ("reward", dopamine-like) rises when replies are accepted and when the user
shows positive emotion. S ("stress", cortisol/noradrenaline-like) rises when
replies are corrected and when the user shows negative, high-arousal emotion.
These are simulated controller variables, not measured hormones.
"""
import numpy as np

try:  # Streamlit is only needed for session-state helpers.
    import streamlit as st
except Exception:  # pragma: no cover - lets the core run headless (tests, A/B)
    st = None

EMOTIONS = ["sadness", "anger", "fear", "joy", "surprise", "neutral"]
FEATURES = EMOTIONS + ["arousal", "uncertainty"]
RESPONSES = ["Supportive", "De-escalating", "Calming", "Upbeat", "Curious", "Neutral"]

# Default emotion -> response pairing used as the network's starting prior.
DEFAULT_RESPONSE_FOR = {
    "sadness": "Supportive",
    "anger": "De-escalating",
    "fear": "Calming",
    "joy": "Upbeat",
    "surprise": "Curious",
    "neutral": "Neutral",
}

ETA = 0.25            # base learning rate
W_LIMIT = 3.0         # weight clip
ENDO_RATE = 0.25      # leak rate of the endocrine states
D0, S0 = 0.30, 0.20   # resting endocrine levels


def clip(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


def prior_weights(strength=1.0):
    """Weak, transparent starting associations so replies are sensible before any feedback."""
    W = np.zeros((len(FEATURES), len(RESPONSES)), dtype=float)
    for emotion, response in DEFAULT_RESPONSE_FOR.items():
        W[FEATURES.index(emotion), RESPONSES.index(response)] = strength
    # When the signal is unclear, lean towards a neutral, clarifying reply.
    W[FEATURES.index("uncertainty"), RESPONSES.index("Neutral")] = 0.5 * strength
    return W


def build_features(emotion_scores, arousal=0.0, uncertainty=0.0):
    """Turn fused emotion scores + acoustic arousal + uncertainty into the network input x."""
    emotion_scores = emotion_scores or {"neutral": 1.0}
    x = [float(emotion_scores.get(e, 0.0)) for e in EMOTIONS]
    x += [clip(arousal), clip(uncertainty)]
    return np.array(x, dtype=float)


# ---------------------------------------------------------------- endocrine

def emotional_drive(x):
    """How the user's detected state pushes the endocrine system.

    Negative, high-arousal emotion raises the stress drive; positive emotion
    raises the reward drive. Returns (reward_drive, stress_drive) in [0, 1].
    """
    sad, anger, fear, joy, surprise, _neutral, arousal, _unc = x
    negative = sad + anger + fear
    stress_drive = clip(0.7 * negative + 0.3 * arousal * negative + 0.3 * arousal * (1 - joy))
    reward_drive = clip(joy + 0.3 * surprise)
    return reward_drive, stress_drive


def update_endocrine(D, S, reward_in, stress_in, rate=ENDO_RATE):
    """Leaky integration: hormones drift towards their current inputs."""
    D = clip((1 - rate) * D + rate * reward_in)
    S = clip((1 - rate) * S + rate * stress_in)
    return D, S


def endocrine_step(D, S, x, accepted):
    """Update D and S from the interaction outcome (70%) and the user's emotion (30%)."""
    r_emo, s_emo = emotional_drive(x)
    reward_in = 0.7 * (1.0 if accepted else 0.0) + 0.3 * r_emo
    stress_in = 0.7 * (0.0 if accepted else 1.0) + 0.3 * s_emo
    return update_endocrine(D, S, reward_in, stress_in)


def endocrine_modulation(D, S):
    """Plasticity factor g(H).

    Stress (recent corrections, distressed user) raises plasticity so the network
    re-learns quickly when it is getting things wrong; sustained reward lowers it
    so learned associations are consolidated instead of being overwritten by
    one-off noisy feedback. Engineering rule inspired by neuromodulation, not a
    biological law.
    """
    return clip(1.0 + 2.0 * S - 1.0 * D, 0.2, 2.0)


def endocrine_bias(S):
    """Under stress the system prefers careful replies (Supportive / Calming / De-escalating)."""
    bias = np.zeros(len(RESPONSES))
    for name in ("Supportive", "Calming", "De-escalating"):
        bias[RESPONSES.index(name)] = 0.15 * S
    return bias


# ------------------------------------------------------------------ network

def softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def predict(x, W, S=0.0, use_bias=True):
    scores = x @ W + (endocrine_bias(S) if use_bias else 0.0)
    probs = softmax(3.0 * scores)
    return scores, probs, int(np.argmax(scores))


def hebbian_update(W, x, target_idx, probs, factor=1.0, eta=ETA):
    """Three-factor Hebbian rule with an anti-Hebbian term.

    dW = eta * g(H) * x (y - p)^T

    The Hebbian part x*y^T strengthens the input -> correct-response link; the
    anti-Hebbian part -x*p^T weakens links to responses the network currently
    favours, so wrong associations decay and weights can become negative.
    """
    y = np.zeros(len(RESPONSES))
    y[target_idx] = 1.0
    delta = eta * factor * np.outer(x, y - probs)
    return np.clip(W + delta, -W_LIMIT, W_LIMIT), delta


# ---------------------------------------------------------------- replies

def make_response(pred, emotion_scores=None, uncertainty=0.0, variant=0, text=""):
    """Reply in the chosen strategy, tailored to what the person said (see replies.py)."""
    from .replies import compose_reply
    return compose_reply(pred, text=text, variant=variant, uncertainty=uncertainty)


# ------------------------------------------------------------ session state

def init_state():
    defaults = {
        "W": prior_weights(),
        "reward": D0,
        "stress": S0,
        "history": [],
        "last_x": None,
        "last_features": None,
        "last_pred": None,
        "last_probs": None,
        "last_notes": [],
        "ab_results": None,
        "emotion_scores": None,
        "emotion_notes": [],
        "transcript": "",
        "reply_variant": 0,
        "messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if hasattr(value, "copy") else value


def reset_state():
    for key in ("W", "reward", "stress", "history", "last_x", "last_features", "last_pred",
                "last_probs", "last_notes", "ab_results", "emotion_scores", "emotion_notes",
                "transcript", "reply_variant", "messages"):
        st.session_state.pop(key, None)
    init_state()


# ------------------------------------------------------------ visualisation

def render_network_svg(weights):
    """Lightweight SVG of feature-to-response associations (positive pink, negative violet)."""
    n_in, n_out = weights.shape
    height = 60 + 34 * max(n_in, n_out)
    in_y = [50 + i * (height - 80) / (n_in - 1) for i in range(n_in)]
    out_y = [50 + j * (height - 80) / (n_out - 1) for j in range(n_out)]
    xi, xo = 95, 325
    parts = [
        '<div class="neo-network-card">',
        f'<svg viewBox="0 0 420 {height:.0f}" role="img" aria-label="Neural association network" style="width:100%;height:auto;display:block;">',
        f'<text x="{xi}" y="22" text-anchor="middle" font-size="11" font-weight="800" fill="#A92F65">INPUT FEATURES</text>',
        f'<text x="{xo}" y="22" text-anchor="middle" font-size="11" font-weight="800" fill="#A92F65">RESPONSES</text>',
    ]
    max_abs = max(1.0, float(np.max(np.abs(weights))))
    for i in range(n_in):
        for j in range(n_out):
            value = float(weights[i, j])
            strength = abs(value) / max_abs
            if strength < 0.03:
                continue
            stroke = "#D94F87" if value >= 0 else "#8E77B5"
            parts.append(
                f'<line class="neo-edge" x1="{xi+12}" y1="{in_y[i]:.1f}" x2="{xo-12}" y2="{out_y[j]:.1f}" '
                f'stroke="{stroke}" stroke-width="{0.8 + 4.5*strength:.2f}" '
                f'stroke-opacity="{0.2 + 0.7*strength:.2f}" stroke-linecap="round"/>'
            )
    for i, label in enumerate(FEATURES):
        parts.append(f'<circle class="neo-node" cx="{xi}" cy="{in_y[i]:.1f}" r="11" fill="#FCE1ED" stroke="#D94F87" stroke-width="1.5"/>')
        parts.append(f'<text x="{xi-18}" y="{in_y[i]+4:.1f}" text-anchor="end" font-size="10" fill="#806A76">{label}</text>')
    for j, label in enumerate(RESPONSES):
        parts.append(f'<circle class="neo-node" cx="{xo}" cy="{out_y[j]:.1f}" r="11" fill="#FFEBDD" stroke="#E6A27F" stroke-width="1.5"/>')
        parts.append(f'<text x="{xo+18}" y="{out_y[j]+4:.1f}" text-anchor="start" font-size="10" fill="#806A76">{label}</text>')
    parts.append(
        '</svg><div style="display:flex;justify-content:center;gap:1rem;flex-wrap:wrap;font-size:.72rem;color:#806A76;padding:.25rem .2rem .45rem;">'
        '<span>━ Thicker = larger weight</span><span style="color:#D94F87;">━ Positive</span>'
        '<span style="color:#8E77B5;">━ Negative (anti-Hebbian)</span></div></div>'
    )
    return "".join(parts)


# ------------------------------------------------------------ A/B experiment

def _simulate_user(rng, n, shift_at):
    """Synthetic interaction stream for one simulated user.

    Each step has a true emotion, noisy perceived emotion scores (like real
    classifiers), arousal and uncertainty. The user's preferred reply differs
    from the default prior for some emotions, and their preferences change
    part-way through (shift_at) - the situation the endocrine system is meant
    to help with. 10% of feedback labels are random (noisy users).
    """
    prefs_a = dict(DEFAULT_RESPONSE_FOR, anger="Calming", surprise="Upbeat")
    prefs_b = dict(prefs_a, sadness="Neutral", fear="Supportive", joy="Curious")
    X, y = [], []
    for t in range(n):
        true = rng.integers(len(EMOTIONS))
        uncertainty = rng.uniform(0, 0.8)
        clarity = 1.0 - 0.8 * uncertainty
        conc = np.full(len(EMOTIONS), 0.6)
        conc[true] += 6.0 * clarity
        scores = rng.dirichlet(conc)
        high_arousal = EMOTIONS[true] in ("anger", "fear", "surprise", "joy")
        arousal = clip(rng.normal(0.65 if high_arousal else 0.3, 0.15))
        X.append(np.concatenate([scores, [arousal, uncertainty]]))
        prefs = prefs_a if t < shift_at else prefs_b
        target = RESPONSES.index(prefs[EMOTIONS[true]])
        if rng.random() < 0.10:
            target = rng.integers(len(RESPONSES))
        y.append(target)
    return np.array(X), np.array(y)


def run_ab_experiment(seeds=20, samples=240):
    """Online (prequential) comparison of plain vs endocrine-modulated learning.

    Both variants see the identical stream, start from the same prior weights,
    use the same Hebbian/anti-Hebbian rule and base learning rate. Accuracy is
    whether the reply chosen *before* feedback matched what the user wanted.
    Variant B additionally runs the endocrine system (plasticity g and stress bias).
    """
    shift_at = samples // 2
    curves_a, curves_b, acc_a, acc_b = [], [], [], []
    post_a, post_b = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        X, y = _simulate_user(rng, samples, shift_at)
        W_a, W_b = prior_weights(), prior_weights()
        D, S = D0, S0
        hits_a, hits_b = [], []
        for x, target in zip(X, y):
            _, p_a, k_a = predict(x, W_a, use_bias=False)
            hits_a.append(k_a == target)
            W_a, _ = hebbian_update(W_a, x, target, p_a, factor=1.0)

            _, p_b, k_b = predict(x, W_b, S=S)
            hits_b.append(k_b == target)
            W_b, _ = hebbian_update(W_b, x, target, p_b, factor=endocrine_modulation(D, S))
            D, S = endocrine_step(D, S, x, accepted=(k_b == target))
        hits_a, hits_b = np.array(hits_a, float), np.array(hits_b, float)
        acc_a.append(hits_a.mean()); acc_b.append(hits_b.mean())
        post_a.append(hits_a[shift_at:shift_at + 40].mean())
        post_b.append(hits_b[shift_at:shift_at + 40].mean())
        curves_a.append(hits_a); curves_b.append(hits_b)

    # Fairness check: the best *hand-tuned* fixed learning rate for the plain
    # variant, chosen with hindsight on these same streams.
    tuned = {}
    for eta in (0.15, 0.35, 0.5, 0.7, 1.0):
        accs = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            X, y = _simulate_user(rng, samples, shift_at)
            W, hits = prior_weights(), 0
            for x, target in zip(X, y):
                _, p, k = predict(x, W, use_bias=False)
                hits += k == target
                W, _ = hebbian_update(W, x, target, p, eta=eta)
            accs.append(hits / samples)
        tuned[eta] = float(np.mean(accs))
    tuned[ETA] = float(np.mean(acc_a))
    best_eta = max(tuned, key=tuned.get)

    window = 20
    kernel = np.ones(window) / window
    curve_a = np.convolve(np.mean(curves_a, axis=0), kernel, mode="valid")
    curve_b = np.convolve(np.mean(curves_b, axis=0), kernel, mode="valid")
    acc_a, acc_b = np.array(acc_a), np.array(acc_b)
    return {
        "seeds": seeds,
        "samples": samples,
        "shift_at": shift_at,
        "plain_accuracy": float(acc_a.mean()),
        "endo_accuracy": float(acc_b.mean()),
        "plain_std": float(acc_a.std()),
        "endo_std": float(acc_b.std()),
        "plain_post_shift": float(np.mean(post_a)),
        "endo_post_shift": float(np.mean(post_b)),
        "endo_wins": int(np.sum(acc_b > acc_a)),
        "ties": int(np.sum(acc_b == acc_a)),
        "base_eta": ETA,
        "best_fixed_eta": float(best_eta),
        "best_fixed_accuracy": tuned[best_eta],
        "curve_plain": curve_a,
        "curve_endo": curve_b,
        "curve_window": window,
    }
