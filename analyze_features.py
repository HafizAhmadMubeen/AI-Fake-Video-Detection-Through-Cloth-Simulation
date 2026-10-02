#!/usr/bin/env python
"""
analyze_features.py - Phase 4 feature validation and real-vs-fake separation analysis.

Reads phase4_features.csv (produced by compute_features.py) and reports:

  1. Completeness audit    - missing / non-finite values per feature
  2. Univariate separation - per-feature real vs fake stats, separation score, AUC
  3. Per-garment breakdown - upper vs lower garment analysed separately
  4. Video-level analysis  - segments averaged per video (the honest unit of analysis)
  5. Feature correlation   - which features are redundant
  6. Combined preview      - leave-one-video-out logistic regression (Phase 5 preview)

Dependencies: numpy, scipy, matplotlib (all already in requirements.txt).
No pandas / scikit-learn needed.

Usage:
    python analyze_features.py --features phase4_features.csv
    python analyze_features.py --features phase4_features.csv --plots
"""

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

import numpy as np

try:
    from scipy import stats as scipy_stats
except ImportError:
    scipy_stats = None


# Columns that identify a row rather than describe its physics.
METADATA_COLUMNS = {
    "video", "label", "garment", "seed_frame", "num_points",
    "drape_edge_count", "drape_edge_source",
    # Bookkeeping, not measurements.
    "in_fit_set", "fit_at_boundary",
    # A passed check, not a feature: 1.0 on every row confirms no segment ever
    # starts on an occluded frame (audit bug F). Analysing it only produced a
    # misleading "CONSTANT - PROBLEM" warning on every run.
    "seed_visible_frac",
}

# Columns that are NOT physics features but are analysed deliberately, to test
# whether real/fake separation is riding on tracking quality rather than cloth
# behaviour. If visible_frac separates the classes as well as the physics
# features do, the physics claim is not yet supported.
DIAGNOSTIC_COLUMNS = {
    "visible_frac", "seed_visible_frac",
    # Camera framing, not physics. pose_dist_mean_px and body_scale_px measure
    # how large the subject is in frame. In this dataset the fake subjects are
    # about twice the size of the real ones, which made every pixel-valued
    # feature separate by ~2x for reasons having nothing to do with cloth. They
    # are analysed as diagnostics so that confound stays visible, and they are
    # excluded from the combined classifier.
    "pose_dist_mean_px", "pose_dist_norm", "body_scale_px",
}

# Pixel-valued features are scale-dependent and therefore contaminated by the
# framing confound above. Their _norm counterparts are the defensible ones.
SCALE_DEPENDENT = {
    "residual_mean_px", "residual_median_px",
    "accel_mean_magnitude", "accel_std_magnitude",
}

# Populated from the CSV header by load_rows(), so new feature columns are
# picked up without editing this file.
FEATURES = []


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def to_float(value):
    """Parse a CSV cell into a float, returning NaN for blank / unparseable."""
    if value is None:
        return float("nan")
    text = str(value).strip()
    if text == "" or text.lower() in ("na", "nan", "none", "null"):
        return float("nan")
    try:
        parsed = float(text)
    except ValueError:
        return float("nan")
    if not math.isfinite(parsed):
        return float("nan")
    return parsed


def load_rows(path):
    """Load the feature CSV into a list of dicts with floats already parsed."""
    if not os.path.exists(path):
        sys.exit("ERROR: feature file not found: %s" % path)

    rows = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            sys.exit("ERROR: %s appears to be empty." % path)

        # Auto-detect the feature columns: everything that is not metadata.
        # Diagnostics are kept and analysed alongside the physics features so
        # the tracking-quality confound is visible in the same table.
        detected = [c for c in reader.fieldnames if c not in METADATA_COLUMNS]
        if not detected:
            sys.exit("ERROR: %s has no analysable feature columns." % path)
        FEATURES[:] = detected

        for raw in reader:
            row = {
                "video": (raw.get("video") or "").strip(),
                "label": (raw.get("label") or "").strip().lower(),
                "garment": (raw.get("garment") or "").strip().lower(),
                "seed_frame": (raw.get("seed_frame") or "").strip(),
                "num_points": to_float(raw.get("num_points")),
            }
            for feat in FEATURES:
                row[feat] = to_float(raw.get(feat))
            flag = to_float(raw.get("in_fit_set"))
            row["in_fit_set"] = bool(flag == flag and flag)   # NaN-safe
            rows.append(row)

    if not rows:
        sys.exit("ERROR: %s contains a header but no data rows." % path)
    return rows


# ----------------------------------------------------------------------------
# Statistics helpers
# ----------------------------------------------------------------------------

def finite(values):
    """Strip NaN / inf from an array."""
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def separation_score(real_vals, fake_vals):
    """
    |mean_real - mean_fake| / pooled_std

    Same metric used in Experiment 2, kept for continuity with PHASE4_LOG.md.
    Roughly: how many standard deviations apart the two groups sit.
    Larger is better. Below ~0.5 is weak, above ~1.0 is a usable signal.
    """
    a, b = finite(real_vals), finite(fake_vals)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_var = ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (
        len(a) + len(b) - 2
    )
    pooled_std = math.sqrt(pooled_var) if pooled_var > 0 else 0.0
    if pooled_std == 0.0:
        return float("nan")
    return abs(a.mean() - b.mean()) / pooled_std


def auc_and_p(real_vals, fake_vals):
    """
    Rank-based AUC (area under ROC) plus a Mann-Whitney p-value.

    AUC answers: pick one random real and one random fake segment. How often does
    this single feature rank them correctly? 0.5 = coin flip / no signal,
    1.0 = perfect separation. We report max(auc, 1-auc) with a direction flag,
    because a feature that is reliably LOWER in fakes is just as useful as one
    that is higher.
    """
    a, b = finite(real_vals), finite(fake_vals)
    if len(a) < 1 or len(b) < 1:
        return float("nan"), float("nan"), ""

    if scipy_stats is not None:
        try:
            result = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
            u_stat, p_value = float(result.statistic), float(result.pvalue)
        except ValueError:
            # scipy raises when every value is identical.
            return 0.5, 1.0, "none"
    else:
        # Manual fallback: rank-sum without scipy.
        combined = np.concatenate([a, b])
        order = combined.argsort()
        ranks = np.empty(len(combined), dtype=float)
        ranks[order] = np.arange(1, len(combined) + 1)
        # Average ranks for ties.
        sorted_vals = combined[order]
        i = 0
        while i < len(sorted_vals):
            j = i
            while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
                j += 1
            if j > i:
                mean_rank = (i + j + 2) / 2.0
                ranks[order[i : j + 1]] = mean_rank
            i = j + 1
        rank_sum_a = ranks[: len(a)].sum()
        u_stat = rank_sum_a - len(a) * (len(a) + 1) / 2.0
        p_value = float("nan")

    auc_raw = u_stat / (len(a) * len(b))
    if auc_raw >= 0.5:
        direction = "real > fake"
        auc = auc_raw
    else:
        direction = "real < fake"
        auc = 1.0 - auc_raw
    return auc, p_value, direction


def describe(values):
    """Return (n, mean, std, min, max) ignoring NaN."""
    arr = finite(values)
    if len(arr) == 0:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    std = arr.std(ddof=1) if len(arr) > 1 else 0.0
    return len(arr), arr.mean(), std, arr.min(), arr.max()


def fmt(value, width=10, places=3):
    """Format a float for the fixed-width report, printing 'n/a' for NaN."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a".rjust(width)
    return ("%.*f" % (places, value)).rjust(width)


# ----------------------------------------------------------------------------
# Report sections
# ----------------------------------------------------------------------------

def section(title, out):
    out.append("")
    out.append("=" * 78)
    out.append(title)
    out.append("=" * 78)


def report_completeness(rows, out):
    section("1. COMPLETENESS AUDIT", out)
    out.append("")
    out.append("Every feature should be present on (almost) every row. A high missing")
    out.append("count means that feature is silently failing - fix it before Phase 5.")
    out.append("")
    total = len(rows)
    out.append("%-26s %8s %8s %9s" % ("feature", "present", "missing", "missing%"))
    out.append("-" * 78)

    problems = []
    for feat in FEATURES:
        values = [r[feat] for r in rows]
        present = int(np.isfinite(np.asarray(values, dtype=float)).sum())
        missing = total - present
        pct = 100.0 * missing / total if total else 0.0
        flag = ""
        if pct > 20.0:
            flag = "   <-- PROBLEM"
            problems.append((feat, pct))
        elif pct > 5.0:
            flag = "   <-- check"
        out.append("%-26s %8d %8d %8.1f%%%s" % (feat, present, missing, pct, flag))

    out.append("-" * 78)
    out.append("total rows: %d" % total)

    # Degenerate-value check: a feature that never varies carries zero information.
    out.append("")
    out.append("Constant-value check (a feature that never varies is useless):")
    for feat in FEATURES:
        arr = finite([r[feat] for r in rows])
        if len(arr) > 1 and arr.std(ddof=1) == 0.0:
            out.append("  %-26s CONSTANT at %.4f  <-- PROBLEM" % (feat, arr[0]))
            problems.append((feat, -1))
    if not problems:
        out.append("  all features vary - good")

    return problems


def report_univariate(rows, out, title, label_key="label"):
    section(title, out)
    out.append("")
    out.append("separation = |mean_real - mean_fake| / pooled_std   (>1.0 is a good signal)")
    out.append("AUC        = rank-based discriminative power        (0.5 = no signal, 1.0 = perfect)")
    out.append("")
    header = "%-24s %9s %9s %9s %9s %7s %9s" % (
        "feature", "real_mean", "real_std", "fake_mean", "fake_std", "sep", "AUC",
    )
    out.append(header)
    out.append("-" * 78)

    results = {}
    for feat in FEATURES:
        real_vals = [r[feat] for r in rows if r[label_key] == "real"]
        fake_vals = [r[feat] for r in rows if r[label_key] == "fake"]
        _, r_mean, r_std, _, _ = describe(real_vals)
        _, f_mean, f_std, _, _ = describe(fake_vals)
        sep = separation_score(real_vals, fake_vals)
        auc, p_value, direction = auc_and_p(real_vals, fake_vals)
        results[feat] = {
            "sep": sep, "auc": auc, "p": p_value, "direction": direction,
            "real_mean": r_mean, "fake_mean": f_mean,
        }
        out.append(
            "%-24s %s %s %s %s %s %s"
            % (
                feat,
                fmt(r_mean, 9, 2), fmt(r_std, 9, 2),
                fmt(f_mean, 9, 2), fmt(f_std, 9, 2),
                fmt(sep, 7, 2), fmt(auc, 9, 3),
            )
        )
    out.append("-" * 78)

    # Ranked summary - the part that actually tells you what is working.
    out.append("")
    out.append("Ranked by AUC (strongest signal first):")
    ranked = sorted(
        results.items(),
        key=lambda kv: (kv[1]["auc"] if math.isfinite(kv[1]["auc"]) else 0.0),
        reverse=True,
    )
    for feat, info in ranked:
        verdict = "no signal"
        if math.isfinite(info["auc"]):
            if info["auc"] >= 0.85:
                verdict = "STRONG"
            elif info["auc"] >= 0.70:
                verdict = "moderate"
            elif info["auc"] >= 0.60:
                verdict = "weak"
        p_text = ""
        if math.isfinite(info["p"]):
            p_text = "  p=%.4f" % info["p"]
        marker = ""
        if feat in DIAGNOSTIC_COLUMNS:
            marker = "  [DIAGNOSTIC]"
        elif feat in SCALE_DEPENDENT:
            marker = "  [pixel units - scale-confounded]"
        out.append(
            "  %-24s AUC=%s  sep=%s  %-9s %s%s%s"
            % (feat, fmt(info["auc"], 6, 3), fmt(info["sep"], 6, 2),
               verdict, info["direction"], p_text, marker)
        )

    # The confound test. visible_frac is a measure of how well CoTracker coped,
    # not of cloth physics. If it discriminates as well as the physics features,
    # then those features may simply be detecting that fakes track badly.
    diag_aucs = [(f, results[f]["auc"]) for f in DIAGNOSTIC_COLUMNS
                 if f in results and math.isfinite(results[f]["auc"])]
    # Only scale-free features count as evidence of physics. A pixel-valued
    # feature beating the framing diagnostic proves nothing, since both are
    # driven by how large the subject is in frame.
    phys_aucs = [results[f]["auc"] for f in results
                 if f not in DIAGNOSTIC_COLUMNS and f not in SCALE_DEPENDENT
                 and math.isfinite(results[f]["auc"])]
    if diag_aucs and phys_aucs:
        best_diag_name, best_diag = max(diag_aucs, key=lambda kv: kv[1])
        best_phys = max(phys_aucs)
        out.append("")
        out.append("  CONFOUND CHECK: best diagnostic (%s) AUC=%.3f"
                   % (best_diag_name, best_diag))
        out.append("                  best SCALE-FREE physics feature AUC=%.3f" % best_phys)
        out.append("                  (pixel-valued features are excluded here: they rise and")
        out.append("                   fall with subject size in frame, so they cannot")
        out.append("                   distinguish physics from camera framing.)")
        if best_diag >= best_phys - 0.02:
            out.append("                  WARNING: a non-physics diagnostic (%s) separates the"
                       % best_diag_name)
            out.append("                  classes at least as well as any scale-free physics")
            out.append("                  feature. The physics claim is NOT supported on this")
            out.append("                  data - the separation is explained by how the videos")
            out.append("                  were captured (framing / tracking), not cloth motion.")
        elif best_diag >= 0.70:
            out.append("                  CAUTION: tracking quality carries real signal of its")
            out.append("                  own. Physics still leads, but report this honestly.")
        else:
            out.append("                  OK: physics features clearly outperform tracking")
            out.append("                  quality, so separation is not merely a tracking artefact.")

    return results


def report_per_garment(rows, out):
    section("3. PER-GARMENT BREAKDOWN", out)
    out.append("")
    out.append("Upper and lower garments drape differently, so a feature can work well")
    out.append("on one and not the other. If so, Phase 5 should treat them separately.")

    garments = sorted({r["garment"] for r in rows if r["garment"]})
    for garment in garments:
        subset = [r for r in rows if r["garment"] == garment]
        n_real = sum(1 for r in subset if r["label"] == "real")
        n_fake = sum(1 for r in subset if r["label"] == "fake")
        out.append("")
        out.append("--- garment: %s  (%d rows: %d real, %d fake) ---"
                   % (garment, len(subset), n_real, n_fake))
        if n_real < 2 or n_fake < 2:
            out.append("    too few rows in one class to analyse")
            continue
        out.append("%-24s %9s %9s %7s %9s" % ("feature", "real_mean", "fake_mean", "sep", "AUC"))
        for feat in FEATURES:
            real_vals = [r[feat] for r in subset if r["label"] == "real"]
            fake_vals = [r[feat] for r in subset if r["label"] == "fake"]
            _, r_mean, _, _, _ = describe(real_vals)
            _, f_mean, _, _, _ = describe(fake_vals)
            sep = separation_score(real_vals, fake_vals)
            auc, _, _ = auc_and_p(real_vals, fake_vals)
            out.append("%-24s %s %s %s %s"
                       % (feat, fmt(r_mean, 9, 2), fmt(f_mean, 9, 2),
                          fmt(sep, 7, 2), fmt(auc, 9, 3)))


def aggregate_by_video(rows):
    """
    Collapse segment rows to one row per video (mean of each feature).

    This matters: segments from the same video are NOT independent samples. Quoting
    segment-level separation overstates how well the method works, because 10
    segments from one video count as 10 'samples'. Video-level is the honest unit,
    and it is what an examiner will ask about.
    """
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["label"], row["video"])].append(row)

    aggregated = []
    for (label, video), group in sorted(buckets.items()):
        agg = {"video": video, "label": label, "garment": "all",
               "n_segments": len(group)}
        for feat in FEATURES:
            arr = finite([r[feat] for r in group])
            agg[feat] = arr.mean() if len(arr) else float("nan")
        aggregated.append(agg)
    return aggregated


def report_correlation(rows, out):
    section("5. FEATURE CORRELATION", out)
    out.append("")
    out.append("Two features correlated above ~0.9 are measuring the same thing; keeping")
    out.append("both adds no information and can destabilise a small-sample classifier.")
    out.append("")

    matrix = []
    usable = []
    for feat in FEATURES:
        col = np.asarray([r[feat] for r in rows], dtype=float)
        if np.isfinite(col).sum() >= 3:
            usable.append(feat)
            matrix.append(col)

    if len(usable) < 2:
        out.append("  not enough usable features to correlate")
        return

    short = [f[:14] for f in usable]
    out.append("%-16s%s" % ("", "".join("%9s" % s[:8] for s in short)))
    redundant = []
    for i, feat_i in enumerate(usable):
        cells = []
        for j, feat_j in enumerate(usable):
            both = np.isfinite(matrix[i]) & np.isfinite(matrix[j])
            if both.sum() < 3:
                cells.append("     n/a")
                continue
            a, b = matrix[i][both], matrix[j][both]
            if a.std() == 0 or b.std() == 0:
                cells.append("     n/a")
                continue
            corr = float(np.corrcoef(a, b)[0, 1])
            cells.append("%9.2f" % corr)
            if i < j and abs(corr) >= 0.90:
                redundant.append((feat_i, feat_j, corr))
        out.append("%-16s%s" % (feat_i[:15], "".join(cells)))

    out.append("")
    if redundant:
        out.append("Highly correlated pairs (|r| >= 0.90) - consider dropping one of each:")
        for feat_i, feat_j, corr in redundant:
            out.append("  %s  <->  %s   r=%.3f" % (feat_i, feat_j, corr))
    else:
        out.append("No pair exceeds |r| = 0.90 - features are carrying distinct information.")


# ----------------------------------------------------------------------------
# Leave-one-video-out logistic regression (Phase 5 preview)
# ----------------------------------------------------------------------------

def fit_logistic(X, y, l2=1.0, iters=2000, lr=0.1):
    """Minimal L2-regularised logistic regression via gradient descent (no sklearn)."""
    n, d = X.shape
    X_aug = np.hstack([np.ones((n, 1)), X])
    w = np.zeros(d + 1)
    for _ in range(iters):
        z = X_aug @ w
        z = np.clip(z, -30, 30)
        pred = 1.0 / (1.0 + np.exp(-z))
        grad = X_aug.T @ (pred - y) / n
        grad[1:] += (l2 / n) * w[1:]      # do not regularise the intercept
        w -= lr * grad
    return w


def predict_logistic(w, X):
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    z = np.clip(X_aug @ w, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def report_loo_preview(video_rows, out):
    section("6. COMBINED-FEATURE PREVIEW (leave-one-video-out)", out)
    out.append("")
    out.append("This is a PREVIEW of Phase 5, not Phase 5 itself. It trains a simple")
    out.append("logistic regression on all features at video level, holding out one video")
    out.append("at a time. It answers one question: do the Phase 4 features, taken")
    out.append("together, carry enough signal to be worth training a classifier on?")
    out.append("")

    labels = np.array([1.0 if r["label"] == "real" else 0.0 for r in video_rows])
    n_videos = len(video_rows)
    if n_videos < 4 or labels.sum() < 2 or (n_videos - labels.sum()) < 2:
        out.append("  too few videos to run leave-one-out - skipping")
        return

    # Scale-free physics features only. Training on visible_frac would let the
    # classifier win by detecting tracking failure; training on pixel-valued
    # features would let it win by detecting how zoomed-in the camera was.
    # Both are confounds we are trying to rule out, not signals.
    physics_features = [f for f in FEATURES
                        if f not in DIAGNOSTIC_COLUMNS and f not in SCALE_DEPENDENT]
    if not physics_features:
        out.append("  no physics features available - skipping")
        return

    raw = np.array([[r[f] for f in physics_features] for r in video_rows], dtype=float)
    # Drop features that are missing for more than half the videos.
    keep = [i for i in range(raw.shape[1])
            if np.isfinite(raw[:, i]).sum() >= max(3, n_videos // 2)]
    if not keep:
        out.append("  no usable features - skipping")
        return
    kept_names = [physics_features[i] for i in keep]
    raw = raw[:, keep]

    if len(kept_names) > max(3, n_videos // 4):
        out.append("  NOTE: %d features on %d videos is enough to overfit badly."
                   % (len(kept_names), n_videos))
        out.append("        Treat a LOW combined AUC as evidence of overfitting, not of")
        out.append("        weak features - compare it against the best single-feature AUC.")
        out.append("")
    out.append("  features used: %s" % ", ".join(kept_names))
    out.append("  videos: %d (%d real, %d fake)"
               % (n_videos, int(labels.sum()), int(n_videos - labels.sum())))
    out.append("")

    predictions = np.zeros(n_videos)
    for held_out in range(n_videos):
        train_idx = [i for i in range(n_videos) if i != held_out]
        X_train_raw = raw[train_idx]
        y_train = labels[train_idx]

        # Impute and standardise using TRAINING statistics only (no leakage).
        col_means = np.nanmean(np.where(np.isfinite(X_train_raw), X_train_raw, np.nan), axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        X_train = np.where(np.isfinite(X_train_raw), X_train_raw, col_means)
        col_std = X_train.std(axis=0)
        col_std = np.where(col_std > 1e-9, col_std, 1.0)
        X_train = (X_train - col_means) / col_std

        X_test_raw = raw[held_out : held_out + 1]
        X_test = np.where(np.isfinite(X_test_raw), X_test_raw, col_means)
        X_test = (X_test - col_means) / col_std

        weights = fit_logistic(X_train, y_train)
        predictions[held_out] = predict_logistic(weights, X_test)[0]

    predicted_labels = (predictions >= 0.5).astype(float)
    accuracy = float((predicted_labels == labels).mean())
    real_scores = predictions[labels == 1.0]
    fake_scores = predictions[labels == 0.0]
    auc, p_value, _ = auc_and_p(real_scores, fake_scores)

    out.append("  leave-one-video-out accuracy : %.1f%%  (%d / %d correct)"
               % (100.0 * accuracy, int((predicted_labels == labels).sum()), n_videos))
    out.append("  leave-one-video-out AUC      : %.3f" % auc)
    if math.isfinite(p_value):
        out.append("  Mann-Whitney p               : %.4f" % p_value)
    out.append("")
    out.append("  per-video predictions (score near 1.0 = predicted REAL):")
    out.append("  %-34s %-6s %7s %s" % ("video", "truth", "score", "verdict"))
    order = sorted(range(n_videos), key=lambda i: -predictions[i])
    for i in order:
        correct = "ok" if predicted_labels[i] == labels[i] else "WRONG"
        out.append("  %-34s %-6s %7.3f %s"
                   % (video_rows[i]["video"][:34], video_rows[i]["label"],
                      predictions[i], correct))

    out.append("")
    if auc >= 0.90:
        out.append("  VERDICT: Phase 4 features are strong. Proceed to Phase 5 with confidence.")
    elif auc >= 0.75:
        out.append("  VERDICT: usable signal, but there is headroom. Calibration and/or")
        out.append("           more data should improve this before Phase 5.")
    else:
        out.append("  VERDICT: weak. Do not move to Phase 5 yet - revisit the physics")
        out.append("           constants and check the per-feature AUCs above to see which")
        out.append("           feature is underperforming.")


# ----------------------------------------------------------------------------
# Optional plots
# ----------------------------------------------------------------------------

def make_plots(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available - skipping plots")
        return

    n = len(FEATURES)
    cols = 3
    plot_rows = int(math.ceil(n / float(cols)))
    fig, axes = plt.subplots(plot_rows, cols, figsize=(5 * cols, 3.4 * plot_rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, feat in enumerate(FEATURES):
        ax = axes[idx]
        real_vals = finite([r[feat] for r in rows if r["label"] == "real"])
        fake_vals = finite([r[feat] for r in rows if r["label"] == "fake"])
        if len(real_vals) == 0 and len(fake_vals) == 0:
            ax.set_title("%s (no data)" % feat, fontsize=9)
            ax.axis("off")
            continue
        combined = np.concatenate([real_vals, fake_vals]) if len(real_vals) and len(fake_vals) \
            else (real_vals if len(real_vals) else fake_vals)
        bins = np.linspace(combined.min(), combined.max(), 20) if combined.max() > combined.min() else 10
        if len(real_vals):
            ax.hist(real_vals, bins=bins, alpha=0.6, label="real", color="#2E7D32")
        if len(fake_vals):
            ax.hist(fake_vals, bins=bins, alpha=0.6, label="fake", color="#C62828")
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Phase 4 features: real vs fake distributions", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote plots -> %s" % path)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Validate Phase 4 features and measure real-vs-fake separation."
    )
    parser.add_argument("--features", default="phase4_features.csv",
                        help="path to the feature CSV (default: phase4_features.csv)")
    parser.add_argument("--out", default="feature_analysis.txt",
                        help="path for the text report (default: feature_analysis.txt)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Drop segments that were used to FIT each video's material "
                             "(in_fit_set=1). Those segments are not independent evidence, "
                             "since the material was chosen to suit them.")
    parser.add_argument("--plots", action="store_true",
                        help="also write feature_plots.png with real-vs-fake histograms")
    parser.add_argument("--plot-path", default="feature_plots.png")
    args = parser.parse_args()

    rows = load_rows(args.features)

    out = []
    out.append("PHASE 4 FEATURE ANALYSIS")
    out.append("source: %s" % os.path.abspath(args.features))

    if args.eval_only:
        before = len(rows)
        rows = [r for r in rows if not r.get("in_fit_set")]
        out.append("eval-only: dropped %d of %d segments used to fit each video's material"
                   % (before - len(rows), before))
        if not rows:
            sys.exit("ERROR: --eval-only removed every row. Was --material-fits used?")

    empty = [f for f in FEATURES
             if not any(math.isfinite(r[f]) for r in rows)]
    if empty:
        FEATURES[:] = [f for f in FEATURES if f not in empty]
        out.append("not computed in this run (skipped): %s" % ", ".join(empty))

    n_real = sum(1 for r in rows if r["label"] == "real")
    n_fake = sum(1 for r in rows if r["label"] == "fake")
    n_other = len(rows) - n_real - n_fake
    videos = {(r["label"], r["video"]) for r in rows}
    out.append("rows: %d  (%d real, %d fake%s)"
               % (len(rows), n_real, n_fake,
                  ", %d unlabelled" % n_other if n_other else ""))
    out.append("videos: %d" % len(videos))

    if n_other:
        out.append("")
        out.append("WARNING: %d rows have a label that is neither 'real' nor 'fake'." % n_other)
    if n_real == 0 or n_fake == 0:
        out.append("")
        out.append("ERROR: one class is empty - separation cannot be measured.")
        print("\n".join(out))
        return

    problems = report_completeness(rows, out)
    report_univariate(rows, out, "2. UNIVARIATE SEPARATION (segment level, all garments)")
    report_per_garment(rows, out)

    video_rows = aggregate_by_video(rows)
    report_univariate(video_rows, out,
                      "4. VIDEO-LEVEL SEPARATION (segments averaged per video)")
    out.append("")
    out.append("Note: video-level numbers are the honest ones to quote. Segment-level")
    out.append("numbers look better only because segments from one video are correlated.")

    report_correlation(rows, out)
    report_loo_preview(video_rows, out)

    section("SUMMARY", out)
    out.append("")
    if problems:
        out.append("Outstanding data problems to fix before Phase 5:")
        for feat, pct in problems:
            if pct < 0:
                out.append("  - %s is constant (carries no information)" % feat)
            else:
                out.append("  - %s is missing on %.1f%% of rows" % (feat, pct))
    else:
        out.append("No completeness problems detected. All features present and varying.")

    text = "\n".join(out)
    print(text)
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print("\nwrote report -> %s" % os.path.abspath(args.out))

    if args.plots:
        make_plots(rows, args.plot_path)


if __name__ == "__main__":
    main()