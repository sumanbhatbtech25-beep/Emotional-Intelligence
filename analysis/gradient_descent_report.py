"""Gradient descent in NEUROENDO — analysis code for the optimization report.

Shows that NEUROENDO's learning rule is (projected, stochastic) gradient
descent on a cross-entropy objective, verifies the gradient numerically, and
runs the experiments used in the report.

Run from the repository root:
    python analysis/gradient_descent_report.py
Outputs: analysis/figures/*.png and analysis/results.json
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neuroendo.learning import (  # the project's own code
    EMOTIONS, FEATURES, RESPONSES, ETA, W_LIMIT, D0, S0, PENALTY,
    prior_weights, predict, hebbian_update, endocrine_modulation, endocrine_step,
    endocrine_bias, _simulate_user,
)

BETA = 3.0                     # softmax sharpness used by predict(): p = softmax(3 * scores)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
GREY, BLUE, LIGHT = "#8a8a8a", "#2f6fdb", "#c9c9c9"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


# --------------------------------------------------------------------------------------
# 1. The objective and its gradient
# --------------------------------------------------------------------------------------
def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def objective(W, x, y, wrong=None, lam=PENALTY, S=0.0):
    """J(W) = -log p_y(W)  +  lam * z_wrong(W)        (z = BETA * (x W + b(S)))

    Cross-entropy of the reply the user wanted, plus a linear penalty on the
    logit of a reply the user rejected.
    """
    z = BETA * (x @ W + endocrine_bias(S))
    p = softmax(z)
    J = -np.log(p[y])
    if wrong is not None and wrong != y:
        J += lam * z[wrong]
    return J


def gradient(W, x, y, wrong=None, lam=PENALTY, S=0.0):
    """Analytic gradient dJ/dW = BETA * x (p - e_y)^T  [+ BETA * lam * x e_wrong^T]."""
    p = softmax(BETA * (x @ W + endocrine_bias(S)))
    e_y = np.eye(len(RESPONSES))[y]
    G = BETA * np.outer(x, p - e_y)
    if wrong is not None and wrong != y:
        G[:, wrong] += BETA * lam * x
    return G


def numerical_gradient(W, x, y, wrong=None, S=0.0, h=1e-6):
    """Central finite differences, one weight at a time."""
    G = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp, Wm = W.copy(), W.copy()
            Wp[i, j] += h
            Wm[i, j] -= h
            G[i, j] = (objective(Wp, x, y, wrong, S=S) - objective(Wm, x, y, wrong, S=S)) / (2 * h)
    return G


def gd_step(W, x, y, wrong, g):
    """One projected gradient-descent step with step size alpha = ETA * g / BETA."""
    alpha = ETA * g / BETA
    return np.clip(W - alpha * gradient(W, x, y, wrong), -W_LIMIT, W_LIMIT)


def check_gradients(trials=200, seed=0):
    rng = np.random.default_rng(seed)
    worst_fd, worst_rule = 0.0, 0.0
    for _ in range(trials):
        W = rng.normal(0, 1, (len(FEATURES), len(RESPONSES)))
        x = np.concatenate([rng.dirichlet(np.ones(6)), rng.uniform(0, 1, 2)])
        y = int(rng.integers(6))
        wrong = int(rng.integers(6))
        G = gradient(W, x, y, wrong)
        Gn = numerical_gradient(W, x, y, wrong)
        worst_fd = max(worst_fd, np.abs(G - Gn).max() / max(1e-12, np.abs(G).max()))
        # The project's Hebbian rule vs. a gradient-descent step (inside the box, no clipping).
        g = rng.uniform(0.2, 2.0)
        _, probs, _ = predict(x, W, S=0.0)
        W_rule, _ = hebbian_update(W, x, y, probs, factor=g, wrong_idx=wrong)
        W_gd = gd_step(W, x, y, wrong, g)
        worst_rule = max(worst_rule, np.abs(W_rule - W_gd).max())
    return worst_fd, worst_rule


# --------------------------------------------------------------------------------------
# 2. Online stochastic gradient descent on simulated users
# --------------------------------------------------------------------------------------
def run_online(X, y, mode="adaptive", eta=ETA):
    """Prequential run: predict, record accuracy and loss, then take one SGD step.

    mode = "adaptive": step size scaled by the endocrine factor g(D, S)
    mode = "fixed":    constant step size eta (g = 1)
    """
    W, D, S = prior_weights(), D0, S0
    hits, losses, steps = [], [], []
    for x, t in zip(X, y):
        S_used = S if mode == "adaptive" else 0.0
        _, p, k = predict(x, W, S=S_used, use_bias=(mode == "adaptive"))
        hits.append(k == t)
        losses.append(-np.log(max(p[t], 1e-12)))
        g = endocrine_modulation(D, S) if mode == "adaptive" else 1.0
        steps.append(eta * g)
        W, _ = hebbian_update(W, x, t, p, factor=g, eta=eta, wrong_idx=None if k == t else k)
        if mode == "adaptive":
            D, S = endocrine_step(D, S, x, accepted=(k == t))
    return np.array(hits, float), np.array(losses), np.array(steps), W


def online_experiment(users=20, n=240):
    shift = n // 2
    res = {"fixed": [], "adaptive": []}
    for seed in range(users):
        X, y = _simulate_user(np.random.default_rng(seed), n, shift)
        for mode in res:
            res[mode].append(run_online(X, y, mode))
    return res, shift


def lr_sweep(users=20, n=240, etas=(0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)):
    acc = {}
    for eta in etas:
        a = []
        for seed in range(users):
            X, y = _simulate_user(np.random.default_rng(seed), n, n // 2)
            a.append(run_online(X, y, "fixed", eta)[0].mean())
        acc[eta] = float(np.mean(a))
    return acc


# --------------------------------------------------------------------------------------
# 3. Full-batch gradient descent (classic view) and the gradient-ascent mirror
# --------------------------------------------------------------------------------------
def batch_objective(W, X, y):
    return float(np.mean([objective(W, x, t) for x, t in zip(X, y)]))


def batch_gradient(W, X, y):
    return np.mean([gradient(W, x, t) for x, t in zip(X, y)], axis=0)


def batch_gd(X, y, alpha, iters=150, W0=None):
    W = prior_weights() if W0 is None else W0.copy()
    path, losses = [W.copy()], [batch_objective(W, X, y)]
    for _ in range(iters):
        W = np.clip(W - alpha * batch_gradient(W, X, y), -W_LIMIT, W_LIMIT)
        path.append(W.copy())
        losses.append(batch_objective(W, X, y))
    return np.array(losses), path


def batch_accuracy(W, X, y):
    return float(np.mean([predict(x, W, use_bias=False)[2] == t for x, t in zip(X, y)]))


# --------------------------------------------------------------------------------------
# 4. Figures
# --------------------------------------------------------------------------------------
def smooth(a, w=20):
    return np.convolve(a, np.ones(w) / w, mode="valid")


def fig_online(res, shift):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for mode, colour, label in (("fixed", GREY, "Fixed step size (g = 1)"),
                                ("adaptive", BLUE, "Endocrine adaptive step size")):
        hits = np.mean([r[0] for r in res[mode]], axis=0)
        loss = np.mean([r[1] for r in res[mode]], axis=0)
        xs = np.arange(20, len(hits) + 1)
        axes[0].plot(xs, smooth(hits), color=colour, lw=2, label=label)
        axes[1].plot(xs, smooth(loss), color=colour, lw=2, label=label)
    for ax, ttl, yl in ((axes[0], "Reply accuracy (rolling 20)", "accuracy"),
                        (axes[1], "Cross-entropy loss (rolling 20)", "loss  −log p(wanted reply)")):
        ax.axvline(shift, color=LIGHT, ls="--", lw=1)
        ax.text(shift + 3, ax.get_ylim()[1] * 0.98, "user's preferences change", va="top", fontsize=8, color=GREY)
        ax.set_title(ttl, loc="left", fontsize=10)
        ax.set_xlabel("interaction (one SGD step each)")
        ax.set_ylabel(yl)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Online gradient descent: loss falls as feedback arrives; the adaptive step recovers faster after the change",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_online_sgd.png"))
    plt.close(fig)


def fig_step_size(res):
    """How the endocrine system moves the step size over time (one user)."""
    hits, _, steps, _ = res["adaptive"][0]
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.plot(steps, color=BLUE, lw=1.6, label="effective learning rate  η·g(D,S)")
    ax.axhline(ETA, color=GREY, ls="--", lw=1, label="fixed learning rate η = 0.25")
    wrong = np.where(hits == 0)[0]
    ax.scatter(wrong, np.full(len(wrong), steps.min() * 0.9), marker="|", color="#c0392b", s=30, label="wrong reply")
    ax.axvline(len(steps) // 2, color=LIGHT, ls="--", lw=1)
    ax.set_title("The step size rises after runs of wrong replies (stress) and falls after runs of correct ones (reward)",
                 loc="left", fontsize=10, fontweight="bold")
    ax.set_xlabel("interaction"); ax.set_ylabel("step size")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_step_size.png"))
    plt.close(fig)


def fig_lr_sweep(sweep, adaptive_acc):
    etas = list(sweep)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.plot(etas, [sweep[e] for e in etas], "o-", color=GREY, lw=1.6, label="fixed step size")
    ax.axhline(adaptive_acc, color=BLUE, lw=1.6, ls="--", label=f"endocrine adaptive (η = 0.25): {adaptive_acc:.1%}")
    best = max(sweep, key=sweep.get)
    ax.annotate(f"best fixed η = {best:g}: {sweep[best]:.1%}", (best, sweep[best]), xytext=(-30, -60),
                textcoords="offset points", fontsize=8, color=GREY, arrowprops=dict(arrowstyle="-", color=GREY))
    ax.set_xscale("log"); ax.set_xticks(etas); ax.set_xticklabels([f"{e:g}" for e in etas])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("learning rate η (log scale)"); ax.set_ylabel("online accuracy")
    ax.set_title("Too small learns slowly, too large overreacts to noise", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower center")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_lr_sweep.png"))
    plt.close(fig)


def fig_batch(curves):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    shades = {0.05: "#b9cdf2", 0.3: "#6f9be6", 1.5: BLUE, 6.0: "#c0392b"}
    for alpha, losses in curves.items():
        ax.plot(losses, color=shades.get(alpha, GREY), lw=1.8, label=f"α = {alpha:g}")
    ax.set_xlabel("iteration (full-batch gradient descent)"); ax.set_ylabel("average loss J(W)")
    ax.set_title("Batch gradient descent converges; too large a step oscillates", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_batch_gd.png"))
    plt.close(fig)


def fig_landscape(X, y, path, i_feat, j1, j2):
    """Loss over two weights (others fixed at the GD optimum) with the GD path."""
    W_star = path[-1]
    a = np.linspace(-3, 3, 61)
    b = np.linspace(-3, 3, 61)
    Z = np.zeros((len(b), len(a)))
    for bi, vb in enumerate(b):
        for ai, va in enumerate(a):
            W = W_star.copy(); W[i_feat, j1] = va; W[i_feat, j2] = vb
            Z[bi, ai] = batch_objective(W, X, y)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    cs = ax.contour(a, b, Z, levels=18, cmap="Greys", linewidths=0.8)
    ax.clabel(cs, inline=True, fontsize=6, fmt="%.2f")
    px = [W[i_feat, j1] for W in path]; py = [W[i_feat, j2] for W in path]
    ax.plot(px, py, "-", color=BLUE, lw=1.6)
    ax.plot(px[0], py[0], "o", color=GREY, ms=6, label="start (prior weights)")
    ax.plot(px[-1], py[-1], "*", color=BLUE, ms=12, label="after gradient descent")
    ax.set_xlabel(f"W[{FEATURES[i_feat]}, {RESPONSES[j1]}]")
    ax.set_ylabel(f"W[{FEATURES[i_feat]}, {RESPONSES[j2]}]")
    ax.set_title("Gradient descent walks downhill across the loss surface", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_landscape.png"))
    plt.close(fig)


# --------------------------------------------------------------------------------------
# 5. A worked example for the viva (one update by hand)
# --------------------------------------------------------------------------------------
def worked_example():
    x = np.array([0.70, 0.05, 0.05, 0.05, 0.05, 0.10, 0.40, 0.10])   # mostly sad, moderate arousal
    W = prior_weights()
    z = BETA * (x @ W)
    p = softmax(z)
    k = int(np.argmax(z))
    target = RESPONSES.index("Calming")          # the user says: Calming would have been better
    g = endocrine_modulation(D0, S0)
    W_new, delta = hebbian_update(W, x, target, p, factor=g, wrong_idx=k)
    p_new = softmax(BETA * (x @ W_new))
    return {
        "x": x.round(2).tolist(), "z": z.round(4).tolist(), "p": p.round(4).tolist(),
        "chosen": RESPONSES[k], "target": RESPONSES[target], "g": round(g, 4),
        "alpha": round(ETA * g / BETA, 4),
        "delta_sad_col": {r: round(float(delta[0, j]), 4) for j, r in enumerate(RESPONSES)},
        "W_sad_before": {r: round(float(W[0, j]), 4) for j, r in enumerate(RESPONSES)},
        "W_sad_after": {r: round(float(W_new[0, j]), 4) for j, r in enumerate(RESPONSES)},
        "loss_before": round(float(-np.log(p[target])), 4),
        "loss_after": round(float(-np.log(p_new[target])), 4),
        "p_target_before": round(float(p[target]), 4), "p_target_after": round(float(p_new[target]), 4),
        "p_new": p_new.round(4).tolist(),
    }


# --------------------------------------------------------------------------------------
# 6. Extra statistics: significance, recovery time, per-emotion accuracy, gradient norms
# --------------------------------------------------------------------------------------
def recovery_time(hits, shift, level=0.7, w=20):
    """Interactions after the preference change until rolling accuracy is back to `level`."""
    roll = smooth(hits, w)                      # roll[i] = mean of hits[i : i+w]
    for i in range(shift, len(roll)):
        if roll[i - w + 1] >= level if i - w + 1 >= shift else False:
            return i - shift + 1
    return len(hits) - shift                     # never recovered within the run


def extra_statistics(users=100, n=240):
    from scipy import stats
    shift = n // 2
    acc_f, acc_a, post_f, post_a, rec_f, rec_a = [], [], [], [], [], []
    per_emotion = {m: {e: [0, 0] for e in EMOTIONS} for m in ("fixed", "adaptive")}
    gnorm = {"fixed": [], "adaptive": []}
    for seed in range(users):
        X, y = _simulate_user(np.random.default_rng(seed), n, shift)
        true_em = [int(np.argmax(x[:6])) for x in X]   # detected (top-scoring) emotion
        for mode in ("fixed", "adaptive"):
            hits, _, _, _ = run_online(X, y, mode)
            (acc_f if mode == "fixed" else acc_a).append(hits.mean())
            (post_f if mode == "fixed" else post_a).append(hits[shift:shift + 40].mean())
            (rec_f if mode == "fixed" else rec_a).append(recovery_time(hits, shift))
            for e, h in zip(true_em, hits):
                per_emotion[mode][EMOTIONS[e]][0] += h
                per_emotion[mode][EMOTIONS[e]][1] += 1
            if seed < 20:   # gradient norm of each online step (before the step)
                W, D, S, norms = prior_weights(), D0, S0, []
                for x, t in zip(X, y):
                    _, p, k = predict(x, W, S=S if mode == "adaptive" else 0.0, use_bias=(mode == "adaptive"))
                    wrong = None if k == t else k
                    norms.append(np.linalg.norm(gradient(W, x, t, wrong, S=S if mode == "adaptive" else 0.0)))
                    g = endocrine_modulation(D, S) if mode == "adaptive" else 1.0
                    W, _ = hebbian_update(W, x, t, p, factor=g, wrong_idx=wrong)
                    if mode == "adaptive":
                        D, S = endocrine_step(D, S, x, accepted=(k == t))
                gnorm[mode].append(norms)
    acc_f, acc_a = np.array(acc_f), np.array(acc_a)
    post_f, post_a = np.array(post_f), np.array(post_a)
    diff = acc_a - acc_f
    rng = np.random.default_rng(1)
    boot = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
    t = stats.ttest_rel(acc_a, acc_f)
    w = stats.wilcoxon(acc_a, acc_f)
    tp = stats.ttest_rel(post_a, post_f)
    out = {
        "users": users,
        "acc_fixed_mean": float(acc_f.mean()), "acc_adaptive_mean": float(acc_a.mean()),
        "acc_fixed_ci95": [float(np.percentile(acc_f, 2.5)), float(np.percentile(acc_f, 97.5))],
        "diff_mean": float(diff.mean()), "diff_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "adaptive_better_users": int((diff > 0).sum()), "ties": int((diff == 0).sum()),
        "paired_t": float(t.statistic), "paired_t_p": float(t.pvalue), "wilcoxon_p": float(w.pvalue),
        "cohens_dz": float(diff.mean() / diff.std(ddof=1)),
        "post_shift_fixed": float(post_f.mean()), "post_shift_adaptive": float(post_a.mean()),
        "post_shift_t_p": float(tp.pvalue),
        "recovery_fixed_median": float(np.median(rec_f)), "recovery_adaptive_median": float(np.median(rec_a)),
        "recovery_fixed_mean": float(np.mean(rec_f)), "recovery_adaptive_mean": float(np.mean(rec_a)),
        "recovery_wilcoxon_p": float(stats.wilcoxon(rec_a, rec_f).pvalue),
        "per_emotion": {m: {e: v[0] / max(1, v[1]) for e, v in d.items()} for m, d in per_emotion.items()},
    }
    return out, (acc_f, acc_a, rec_f, rec_a), gnorm


def fig_significance(acc_f, acc_a, rec_f, rec_a):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for f, a in zip(acc_f, acc_a):
        ax.plot([0, 1], [f, a], color=BLUE if a > f else ("#c0392b" if a < f else GREY), alpha=0.35, lw=0.9)
    ax.plot([0, 1], [acc_f.mean(), acc_a.mean()], color="black", lw=2.2, marker="o")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["fixed step", "endocrine adaptive"])
    ax.set_xlim(-0.3, 1.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("online accuracy per simulated user")
    better = int((acc_a > acc_f).sum())
    ax.set_title(f"Same user, both learners: adaptive is higher for {better} of {len(acc_f)}", loc="left", fontsize=10, fontweight="bold")
    ax = axes[1]
    bins = np.arange(0, 125, 8)
    ax.hist(rec_f, bins=bins, color=GREY, alpha=0.75, label=f"fixed (median {np.median(rec_f):.0f})")
    ax.hist(rec_a, bins=bins, color=BLUE, alpha=0.6, label=f"adaptive (median {np.median(rec_a):.0f})")
    ax.set_xlabel("interactions to recover 70% accuracy after the change")
    ax.set_ylabel("number of users")
    ax.set_title("Recovery after preferences change", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_significance.png"))
    plt.close(fig)


def fig_per_emotion(per_emotion):
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    xs = np.arange(len(EMOTIONS)); w = 0.38
    f = [per_emotion["fixed"][e] for e in EMOTIONS]; a = [per_emotion["adaptive"][e] for e in EMOTIONS]
    ax.bar(xs - w / 2, f, w, color=GREY, label="fixed step")
    ax.bar(xs + w / 2, a, w, color=BLUE, label="endocrine adaptive")
    for i, (vf, va) in enumerate(zip(f, a)):
        ax.text(i - w / 2, vf + 0.01, f"{vf:.0%}", ha="center", fontsize=7, color=GREY)
        ax.text(i + w / 2, va + 0.01, f"{va:.0%}", ha="center", fontsize=7, color=BLUE)
    ax.set_xticks(xs); ax.set_xticklabels(EMOTIONS)
    ax.set_ylim(0, 1); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("reply accuracy")
    ax.set_title("Accuracy by the emotion the model detected (100 users)", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_per_emotion.png"))
    plt.close(fig)


def fig_gradient_norms(gnorm, batch_paths, Xb, yb):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    for mode, colour, label in (("fixed", GREY, "fixed step"), ("adaptive", BLUE, "endocrine adaptive")):
        m = np.mean(gnorm[mode], axis=0)
        ax.plot(np.arange(20, len(m) + 1), smooth(m), color=colour, lw=1.8, label=label)
    ax.axvline(120, color=LIGHT, ls="--", lw=1)
    ax.set_xlabel("interaction"); ax.set_ylabel("‖∇J‖ (rolling 20)")
    ax.set_title("Online: gradient shrinks as the model fits, jumps at the change", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    for alpha, colour in ((0.05, "#b9cdf2"), (0.3, "#6f9be6"), (1.5, BLUE)):
        norms = [np.linalg.norm(batch_gradient(W, Xb, yb)) for W in batch_paths[alpha]]
        ax.semilogy(norms, color=colour, lw=1.8, label=f"α = {alpha:g}")
    ax.set_xlabel("iteration (full batch)"); ax.set_ylabel("‖∇F‖ (log scale)")
    ax.set_title("Batch: gradient norm falls towards zero (optimum)", loc="left", fontsize=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_gradient_norms.png"))
    plt.close(fig)


def fig_weights(W_before, W_after):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, W, ttl in ((axes[0], W_before, "Before learning (default pairing)"),
                       (axes[1], W_after, "After 240 feedback steps (one user)")):
        im = ax.imshow(W, cmap="RdBu", vmin=-2, vmax=2)
        ax.set_xticks(range(len(RESPONSES))); ax.set_xticklabels(RESPONSES, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels(FEATURES, fontsize=8)
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if abs(W[i, j]) >= 0.05:
                    ax.text(j, i, f"{W[i, j]:+.1f}", ha="center", va="center", fontsize=6.5,
                            color="white" if abs(W[i, j]) > 1.1 else "black")
        ax.set_title(ttl, loc="left", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=axes, shrink=0.8, label="weight (blue +, red −)")
    fig.savefig(os.path.join(OUT, "fig9_weights.png"))
    plt.close(fig)


def main():
    out = {}
    fd_err, rule_err = check_gradients()
    out["gradient_check_max_relative_error"] = fd_err
    out["hebbian_vs_gd_max_abs_difference"] = rule_err

    res, shift = online_experiment()
    for mode in res:
        hits = np.array([r[0] for r in res[mode]])
        loss = np.array([r[1] for r in res[mode]])
        out[mode] = {
            "accuracy": float(hits.mean()), "accuracy_std_users": float(hits.mean(axis=1).std()),
            "loss_first_20": float(loss[:, :20].mean()), "loss_before_shift_last_20": float(loss[:, shift - 20:shift].mean()),
            "loss_after_shift_first_40": float(loss[:, shift:shift + 40].mean()),
            "acc_after_shift_first_40": float(hits[:, shift:shift + 40].mean()),
            "loss_last_20": float(loss[:, -20:].mean()), "accuracy_last_40": float(hits[:, -40:].mean()),
        }
    out["adaptive_wins_users"] = int(sum(a[0].mean() > f[0].mean() for a, f in zip(res["adaptive"], res["fixed"])))
    steps = np.concatenate([r[2] for r in res["adaptive"]])
    out["adaptive_step_range"] = [float(steps.min()), float(steps.max()), float(steps.mean())]

    sweep = lr_sweep()
    out["lr_sweep"] = sweep

    # Batch GD on one user's first half (stable preferences).
    X, y = _simulate_user(np.random.default_rng(0), 240, 120)
    Xb, yb = X[:120], y[:120]
    curves, paths = {}, {}
    for alpha in (0.05, 0.3, 1.5, 6.0):
        curves[alpha], paths[alpha] = batch_gd(Xb, yb, alpha)
    out["batch"] = {str(a): {"J0": float(c[0]), "J_end": float(c[-1]),
                             "acc_start": batch_accuracy(prior_weights(), Xb, yb),
                             "acc_end": batch_accuracy(paths[a][-1], Xb, yb)} for a, c in curves.items()}
    out["worked_example"] = worked_example()

    fig_online(res, shift)
    fig_step_size(res)
    fig_lr_sweep(sweep, out["adaptive"]["accuracy"])
    fig_batch(curves)
    fig_landscape(Xb, yb, paths[0.3], EMOTIONS.index("anger"), RESPONSES.index("De-escalating"),
                  RESPONSES.index("Calming"))

    st, (acc_f, acc_a, rec_f, rec_a), gnorm = extra_statistics()
    out["statistics_100_users"] = st
    fig_significance(acc_f, acc_a, rec_f, rec_a)
    fig_per_emotion(st["per_emotion"])
    fig_gradient_norms(gnorm, paths, Xb, yb)
    fig_weights(prior_weights(), res["adaptive"][0][3])

    with open(os.path.join(os.path.dirname(OUT), "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
