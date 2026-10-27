"""Figures for the poster, saved as 300 dpi PNG and as PDF for print.
XGBoost is blue, logistic regression orange, random forest green; the screening rules are grey."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODEL_COLORS = {"xgboost": "#2a78d6", "logistic": "#eb6834", "random_forest": "#1baf7a"}
MODEL_NAMES = {"xgboost": "XGBoost", "logistic": "Logistic regression", "random_forest": "Random forest"}
TIER_COLORS = {"low": "#0ca30c", "moderate": "#fab219", "high": "#d03b3b"}
TIER_TEXT = {"low": "#ffffff", "moderate": "#0b0b0b", "high": "#ffffff"}
UNITS = {"age": " (years)", "heartrate": " (beats/min)", "temperature": " (°C)", "resprate": " (breaths/min)",
         "o2sat": " (%)", "sbp": " (mmHg)", "dbp": " (mmHg)"}
DIVERGING = LinearSegmentedColormap.from_list("bgr", ["#2a78d6", "#c3c2b7", "#e34948"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 1, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titlelocation": "left",
})


RULE_MARKERS = {"SIRS (triage)": "s", "qSOFA (triage)": "^", "Modified NEWS (triage)": "D", "Triage acuity": "o"}
RULE_NAMES = {"SIRS (triage)": "SIRS ≥ 2", "qSOFA (triage)": "qSOFA ≥ 2", "Modified NEWS (triage)": "NEWS ≥ 5",
              "Triage acuity": "Acuity ESI 1–2"}


def _save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _percent_axis(axis, top):
    step = 0.02 if top <= 0.12 else 0.05 if top <= 0.3 else 0.1
    axis.set_major_locator(matplotlib.ticker.MultipleLocator(step))
    axis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))


def roc_pr(curves, rules, prevalence, path):
    """curves: {model: (fpr, tpr, recall, precision, auroc, auprc)}; rules: {name: (fpr, sens, ppv)}."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.6))
    for m, (fpr, tpr, rec, prec, auroc, auprc) in curves.items():
        a.plot(fpr, tpr, color=MODEL_COLORS[m], label=f"{MODEL_NAMES[m]}  AUROC {auroc:.2f}")
        b.plot(rec, prec, color=MODEL_COLORS[m], label=f"{MODEL_NAMES[m]}  AUPRC {auprc:.2f}")
    a.plot([0, 1], [0, 1], color=AXIS, lw=1)
    b.axhline(prevalence, color=AXIS, lw=1)
    b.annotate(f"Sepsis rate {prevalence:.1%}", (0, prevalence), xytext=(4, 4), textcoords="offset points",
               ha="left", color=MUTED, fontsize=9)
    for name, (fpr, sens, ppv) in rules.items():  # rules: grey points, told apart by marker shape
        for ax, x, y in ((a, fpr, sens), (b, sens, ppv)):
            ax.scatter([x], [y], s=64, marker=RULE_MARKERS.get(name, "o"), color=INK2, edgecolor=SURFACE,
                       linewidth=1.5, zorder=5, label=RULE_NAMES.get(name, name) if ax is a else None)
    a.set(xlabel="1 − specificity", ylabel="Sensitivity", xlim=(0, 1), ylim=(0, 1.01), title="ROC curve")
    b.set(xlabel="Sensitivity (recall)", ylabel="Positive predictive value", xlim=(0, 1), ylim=(0, 1.01),
          title="Precision–recall curve")
    a.legend(loc="lower right", fontsize=8.5)
    b.legend(loc="upper right", fontsize=8.5)
    _save(fig, path)


def calibration(bins_by_model, path):
    fig, ax = plt.subplots(figsize=(5, 4.6))
    top = max(max(d.predicted.max(), d.observed.max()) for d in bins_by_model.values()) * 1.1
    ax.plot([0, top], [0, top], color=AXIS, lw=1)
    for m, d in bins_by_model.items():
        ax.plot(d.predicted, d.observed, color=MODEL_COLORS[m], marker="o", ms=7, mec=SURFACE, mew=2,
                label=MODEL_NAMES[m])
    ax.set(xlim=(0, top), ylim=(0, top), xlabel="Predicted risk (calibrated)", ylabel="Observed sepsis rate",
           title="Calibration on the test set")
    _percent_axis(ax.xaxis, top)
    _percent_axis(ax.yaxis, top)
    ax.legend(loc="upper left", fontsize=8.5)
    _save(fig, path)


def decision_curve(nb_by_model, path):
    fig, ax = plt.subplots(figsize=(6, 4.4))
    any_nb = next(iter(nb_by_model.values()))
    ax.plot(any_nb.threshold, any_nb.treat_all, color=MUTED, lw=1.5, label="Review everyone")
    ax.axhline(0, color=INK2, lw=1, label="Review no one")
    for m, d in nb_by_model.items():
        ax.plot(d.threshold, d.net_benefit, color=MODEL_COLORS[m], label=MODEL_NAMES[m])
    hi = max(d.net_benefit.max() for d in nb_by_model.values())
    ax.set(ylim=(-hi * 0.15, hi * 1.15), xlabel="Risk threshold for review", ylabel="Net benefit",
           title="Decision curve (test set)")
    lo, top = any_nb.threshold.min(), any_nb.threshold.max()
    ax.set_xticks([lo] + [t for t in np.arange(0.05, top + 1e-9, 0.05) if t > lo])
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    ax.legend(fontsize=8.5)
    _save(fig, path)


def traffic_light(tiers, path):
    """tiers: DataFrame with tier, share_of_arrivals, share_of_cases (fractions)."""
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.grid(False)
    for row, col in ((1, "share_of_arrivals"), (0, "share_of_cases")):
        left = 0.0
        for t in ("low", "moderate", "high"):
            v = float(tiers.set_index("tier").loc[t, col])
            ax.barh(row, v, left=left, height=0.55, color=TIER_COLORS[t], edgecolor=SURFACE, linewidth=2)
            if v >= 0.07:
                ax.text(left + v / 2, row, f"{t.capitalize()}\n{v:.0%}", ha="center", va="center", fontsize=9,
                        color=TIER_TEXT[t])
            elif v > 0:  # too narrow to hold its label: put it above the segment
                ax.text(left + v / 2, row + 0.34, f"{t.capitalize()} {v:.0%}", ha="center", va="bottom",
                        fontsize=9, color=INK2)
            left += v
    ax.set_yticks([1, 0], ["All arrivals", "Sepsis cases"])
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(axis="y", length=0, colors=INK2)
    ax.set_ylim(-0.45, 1.75)
    ax.set_title("Where the sepsis cases fall in the traffic light (test set)")
    _save(fig, path)


def shap_summary(sv, X, top, path):
    """Beeswarm-style: one row per top feature, x = SHAP, colour = feature value percentile."""
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(top) + 1.2))
    rng = np.random.default_rng(0)
    for i, r in enumerate(top.itertuples()):
        j = list(X.columns).index(r.feature)
        x = X[r.feature].to_numpy(float)
        pct = (np.argsort(np.argsort(np.nan_to_num(x, nan=np.nanmedian(x) if np.isfinite(x).any() else 0)))
               / max(len(x) - 1, 1))
        y = len(top) - 1 - i + rng.uniform(-0.3, 0.3, len(x))
        ax.scatter(sv[:, j], y, c=pct, cmap=DIVERGING, s=3, alpha=0.6, linewidths=0, rasterized=True)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(top))[::-1], top.label)
    ax.grid(axis="y", visible=False)
    ax.set(xlabel="SHAP value (contribution to log-odds of sepsis)", title="What drives the XGBoost risk score")
    sm = plt.cm.ScalarMappable(cmap=DIVERGING)
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["Low value", "High value"], color=INK2, fontsize=8)
    cb.outline.set_visible(False)
    _save(fig, path)


def shap_dependence(sv, X, features, labels, path):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, f in zip(axes.flat, features):
        j = list(X.columns).index(f)
        x = X[f].to_numpy(float)
        ok = ~np.isnan(x)
        ax.scatter(x[ok], sv[ok, j], s=3, alpha=0.25, color=MODEL_COLORS["xgboost"], linewidths=0, rasterized=True)
        if ok.sum() > 50:  # binned mean as the readable trend
            edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, 21)))
            idx = np.clip(np.digitize(x[ok], edges[1:-1]), 0, len(edges) - 2)
            mx = [x[ok][idx == k].mean() for k in range(len(edges) - 1) if (idx == k).any()]
            my = [sv[ok, j][idx == k].mean() for k in range(len(edges) - 1) if (idx == k).any()]
            ax.plot(mx, my, color=INK, lw=2)
        ax.axhline(0, color=AXIS, lw=1)
        ax.set(xlabel=f"{labels[f]}{UNITS.get(f, '')}", ylabel="SHAP value", title=labels[f])
    fig.tight_layout()
    _save(fig, path)


def fairness(table, overall, path):
    """Dot + 95% CI per subgroup for AUROC and sensitivity at the high-risk cut-off."""
    t = table[table.auroc.notna()].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.32 * len(t) + 1.4), sharey=True)
    y = np.arange(len(t))[::-1]
    for ax, m, ttl in ((axes[0], "auroc", "AUROC"), (axes[1], "sensitivity_high", "Sensitivity, high-risk tier")):
        ax.hlines(y, t[f"{m}_lo"], t[f"{m}_hi"], color=MODEL_COLORS["xgboost"], lw=2)
        ax.scatter(t[m], y, s=48, color=MODEL_COLORS["xgboost"], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.axvline(overall[m], color=MUTED, lw=1)
        ax.set(title=ttl)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [f"{a}: {g}" for a, g in zip(t.attribute, t.group)])
    axes[0].tick_params(axis="y", colors=INK2)
    fig.suptitle("Performance by patient group (test set; grey line = all patients)", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def rules_matched(rows, path):
    """rows: rule, alert_rate, rule_sens(, lo, hi), model_sens(, lo, hi). Cases found per 100 at equal workload."""
    fig, ax = plt.subplots(figsize=(7.5, 0.95 * len(rows) + 1.1))
    ax.grid(axis="y", visible=False)
    h = 0.36
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        for dy, (v, lo, hi), color, who in ((h / 2 + 0.02, r["model"], MODEL_COLORS["xgboost"], "Model"),
                                            (-h / 2 - 0.02, r["rule"], AXIS, "Rule")):
            ax.barh(y + dy, v * 100, height=h, color=color, label=who if i == 0 else None)
            ax.hlines(y + dy, lo * 100, hi * 100, color=INK2, lw=1.2)
            ax.text(hi * 100 + 1.5, y + dy, f"{v * 100:.0f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows))[::-1],
                  [f"{RULE_NAMES.get(r['name'], r['name'])}\nflags {r['alert_rate']:.1%}" for r in rows])
    ax.tick_params(axis="y", length=0, colors=INK2)
    ax.set_xlim(0, 105)
    ax.set(xlabel="Sepsis cases found per 100 (95% CI)",
           title="Same number of alerts: cases found by the model and by each rule (test set)")
    ax.legend(loc="lower right", fontsize=8.5)
    _save(fig, path)


def shap_bars(top, path, n=10):
    """Mean absolute SHAP value of the top features, coloured by the direction of their effect."""
    t = top.head(n).iloc[::-1]
    up = t.direction.str.endswith("higher risk").to_numpy()
    colors = np.where(up, "#e34948", MODEL_COLORS["xgboost"])
    fig, ax = plt.subplots(figsize=(6.5, 0.38 * len(t) + 1.2))
    ax.barh(t.label, t.mean_abs_shap, color=colors, height=0.6)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0, colors=INK2)
    ax.set(xlabel="Average effect on the risk score (mean |SHAP|)", title="What drives the risk score (test set)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#e34948", label="Higher value → higher risk"),
                       Patch(color=MODEL_COLORS["xgboost"], label="Higher value → lower risk")],
              loc="lower right", fontsize=8.5)
    _save(fig, path)


def cohort_flow(start, excluded, final, splits, path):
    """start / final: box text; excluded: list of exclusion lines; splits: three box texts."""
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.axis("off")
    box = dict(boxstyle="round,pad=0.6", fc=SURFACE, ec=AXIS, lw=1.2)
    side = dict(boxstyle="round,pad=0.5", fc="#f1f0ec", ec=GRID, lw=1)
    arrow = dict(arrowstyle="-|>", color=MUTED, lw=1.2)
    ax.text(0.4, 0.9, start, ha="center", va="center", fontsize=10, bbox=box)
    ax.annotate("", (0.4, 0.5), (0.4, 0.82), arrowprops=arrow)
    ax.annotate("", (0.66, 0.66), (0.4, 0.66), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
    ax.text(0.68, 0.66, "Excluded\n" + "\n".join(excluded), ha="left", va="center", fontsize=9, color=INK2,
            bbox=side)
    ax.text(0.4, 0.42, final, ha="center", va="center", fontsize=10, fontweight="bold", bbox=box)
    for x, lab in zip((0.1, 0.4, 0.7), splits):
        ax.annotate("", (x, 0.2), (0.4, 0.33), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1))
        ax.text(x, 0.1, lab, ha="center", va="center", fontsize=8.5, bbox=box)
    ax.set(xlim=(-0.08, 1.0), ylim=(0, 1))
    ax.set_title("Who was studied", loc="left")
    _save(fig, path)
