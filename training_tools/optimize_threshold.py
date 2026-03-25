#!/usr/bin/env python3
"""
optimize_threshold.py — точный подбор decision threshold для бинарной классификации.

Идея:
  - для бинарной модели с непрерывным score решение меняется только в конечном числе
    точек между уникальными значениями score;
  - поэтому вместо "черного ящика" используется exact search по всем значимым
    threshold-кандидатам;
  - для каждого threshold считаются метрики, затем выбирается лучший по заданной цели.

Типовой сценарий:
  1. Прогнать модель на независимом validation / real-world mini-set.
  2. Сохранить CSV с колонками:
       - confidence (или другая score-колонка)
       - manual_label
  3. Подобрать threshold этой утилитой.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    np = None
    pd = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact threshold search for binary spam/non-spam classification."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="CSV with model scores and manual labels."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory for sweep CSV and best-threshold JSON. Default: <input_stem>_threshold_search"
    )
    parser.add_argument(
        "--sep",
        default=",",
        help="CSV separator. Default: ','"
    )
    parser.add_argument(
        "--score-col",
        default="confidence",
        help="Score column name. Default: confidence"
    )
    parser.add_argument(
        "--label-col",
        default="manual_label",
        help="Ground-truth label column. Default: manual_label"
    )
    parser.add_argument(
        "--positive-label",
        default="spam",
        help="Positive class label. Default: spam"
    )
    parser.add_argument(
        "--negative-label",
        default="non_spam",
        help="Negative class label. Default: non_spam"
    )
    parser.add_argument(
        "--rule",
        choices=["greater_equal", "less_equal"],
        default="greater_equal",
        help="Decision rule. Default: greater_equal (positive if score >= threshold)"
    )
    parser.add_argument(
        "--optimize",
        choices=[
            "accuracy",
            "balanced_accuracy",
            "spam_f1",
            "spam_precision",
            "spam_recall",
            "youden_j",
        ],
        default="accuracy",
        help="Target metric to maximize. Default: accuracy"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="How many best thresholds to print. Default: 10"
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=None,
        help="Optional middle threshold for review-zone analysis."
    )
    parser.add_argument(
        "--block-threshold",
        type=float,
        default=None,
        help="Optional high-confidence threshold for block-zone analysis."
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Do not render ROC/PR plots."
    )
    return parser.parse_args()


def validate_binary_labels(
    df: pd.DataFrame,
    label_col: str,
    positive_label: str,
    negative_label: str,
) -> pd.DataFrame:
    out = df.copy()
    out[label_col] = out[label_col].astype(str).str.strip()
    empty_mask = out[label_col] == ""
    if empty_mask.any():
        out = out.loc[~empty_mask].copy()
    allowed = {positive_label, negative_label}
    bad = sorted(set(out[label_col]) - allowed)
    if bad:
        raise ValueError(
            f"Unexpected labels in '{label_col}': {bad}. "
            f"Expected only {sorted(allowed)}."
        )
    if out.empty:
        raise ValueError(
            f"No labeled rows left after filtering empty values in '{label_col}'."
        )
    return out


def build_threshold_candidates(scores: np.ndarray) -> list[float]:
    unique_scores = np.array(sorted(set(float(x) for x in scores)))
    if len(unique_scores) == 0:
        return []
    if len(unique_scores) == 1:
        return [unique_scores[0] - 1e-6, unique_scores[0] + 1e-6]

    span = unique_scores[-1] - unique_scores[0]
    eps = max(1e-9, span * 1e-6, 1e-6)

    thresholds = [float(unique_scores[0] - eps)]
    thresholds.extend(
        float((left + right) / 2.0)
        for left, right in zip(unique_scores[:-1], unique_scores[1:])
    )
    thresholds.append(float(unique_scores[-1] + eps))
    return thresholds


def predict_by_threshold(
    scores: np.ndarray,
    threshold: float,
    positive_label: str,
    negative_label: str,
    rule: str,
) -> np.ndarray:
    if rule == "greater_equal":
        positive_mask = scores >= threshold
    else:
        positive_mask = scores <= threshold
    return np.where(positive_mask, positive_label, negative_label)


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def normalize_scores_for_positive(scores: np.ndarray, rule: str) -> np.ndarray:
    """
    For ROC/PR we want larger values => stronger confidence in positive class.
    """
    return scores if rule == "greater_equal" else -scores


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: str,
    negative_label: str,
) -> dict[str, float]:
    tp = int(((y_true == positive_label) & (y_pred == positive_label)).sum())
    fp = int(((y_true == negative_label) & (y_pred == positive_label)).sum())
    tn = int(((y_true == negative_label) & (y_pred == negative_label)).sum())
    fn = int(((y_true == positive_label) & (y_pred == negative_label)).sum())

    accuracy = safe_div(tp + tn, tp + fp + tn + fn)
    spam_precision = safe_div(tp, tp + fp)
    spam_recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    balanced_accuracy = (spam_recall + specificity) / 2.0
    spam_f1 = safe_div(2 * spam_precision * spam_recall, spam_precision + spam_recall)
    youden_j = spam_recall + specificity - 1.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "spam_precision": spam_precision,
        "spam_recall": spam_recall,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
        "spam_f1": spam_f1,
        "youden_j": youden_j,
        "pred_spam": int((y_pred == positive_label).sum()),
        "pred_non_spam": int((y_pred == negative_label).sum()),
    }


def optimize_threshold(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    positive_label: str,
    negative_label: str,
    rule: str,
    optimize_metric: str,
) -> tuple[pd.DataFrame, dict]:
    scores = df[score_col].astype(float).to_numpy()
    y_true = df[label_col].astype(str).to_numpy()
    thresholds = build_threshold_candidates(scores)
    if not thresholds:
        raise ValueError("No threshold candidates could be built from score column.")

    rows = []
    for threshold in thresholds:
        y_pred = predict_by_threshold(
            scores=scores,
            threshold=threshold,
            positive_label=positive_label,
            negative_label=negative_label,
            rule=rule,
        )
        metrics = compute_metrics(y_true, y_pred, positive_label, negative_label)
        rows.append({"threshold": threshold, **metrics})

    sweep = pd.DataFrame(rows)
    sweep = sweep.sort_values(
        by=[optimize_metric, "balanced_accuracy", "spam_f1", "accuracy", "threshold"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    best = sweep.iloc[0].to_dict()
    return sweep, best


def compute_curve_metrics(
    scores: np.ndarray,
    y_true: np.ndarray,
    positive_label: str,
    rule: str,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    try:
        from sklearn.metrics import (
            average_precision_score,
            precision_recall_curve,
            roc_auc_score,
            roc_curve,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: install scikit-learn in the active environment."
        ) from exc

    y_bin = (y_true == positive_label).astype(int)
    pos_scores = normalize_scores_for_positive(scores, rule)

    fpr, tpr, roc_thresholds = roc_curve(y_bin, pos_scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_bin, pos_scores)
    roc_auc = float(roc_auc_score(y_bin, pos_scores))
    pr_auc = float(average_precision_score(y_bin, pos_scores))

    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr,
        "threshold_score": np.r_[roc_thresholds],
    })
    pr_df = pd.DataFrame({
        "precision": precision,
        "recall": recall,
        "threshold_score": np.r_[pr_thresholds, np.nan],
    })
    return roc_df, pr_df, roc_auc, pr_auc


def plot_curve(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
    diagonal: bool = False,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df[x_col], df[y_col], linewidth=2.0)
    if diagonal:
        ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="gray")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def build_zone_policy_table(
    scores: np.ndarray,
    y_true: np.ndarray,
    positive_label: str,
    negative_label: str,
    rule: str,
    review_threshold: float,
    block_threshold: float,
) -> pd.DataFrame:
    if rule == "greater_equal":
        if block_threshold < review_threshold:
            raise ValueError(
                "For greater_equal rule block-threshold must be >= review-threshold."
            )
        zone = np.where(
            scores >= block_threshold,
            "block",
            np.where(scores >= review_threshold, "review", "allow"),
        )
    else:
        if block_threshold > review_threshold:
            raise ValueError(
                "For less_equal rule block-threshold must be <= review-threshold."
            )
        zone = np.where(
            scores <= block_threshold,
            "block",
            np.where(scores <= review_threshold, "review", "allow"),
        )

    y_true = np.asarray(y_true)
    rows = []
    for zone_name in ["block", "review", "allow"]:
        mask = zone == zone_name
        count = int(mask.sum())
        if count == 0:
            rows.append({
                "zone": zone_name,
                "count": 0,
                "coverage": 0.0,
                "spam_share": 0.0,
                "non_spam_share": 0.0,
            })
            continue
        zone_true = y_true[mask]
        rows.append({
            "zone": zone_name,
            "count": count,
            "coverage": count / len(y_true),
            "spam_share": float((zone_true == positive_label).mean()),
            "non_spam_share": float((zone_true == negative_label).mean()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()

    if np is None or pd is None:
        raise SystemExit(
            "Missing dependencies: install pandas and numpy in the active environment."
        )

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else input_path.with_name(
        f"{input_path.stem}_threshold_search"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, sep=args.sep, dtype=str).fillna("")
    required = {args.score_col, args.label_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {list(df.columns)}"
        )

    df = validate_binary_labels(
        df,
        label_col=args.label_col,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
    )
    n_labeled = len(df)

    sweep, best = optimize_threshold(
        df=df,
        score_col=args.score_col,
        label_col=args.label_col,
        positive_label=args.positive_label,
        negative_label=args.negative_label,
        rule=args.rule,
        optimize_metric=args.optimize,
    )

    scores = df[args.score_col].astype(float).to_numpy()
    y_true = df[args.label_col].astype(str).to_numpy()
    roc_df, pr_df, roc_auc, pr_auc = compute_curve_metrics(
        scores=scores,
        y_true=y_true,
        positive_label=args.positive_label,
        rule=args.rule,
    )

    sweep_path = output_dir / "threshold_sweep.csv"
    best_path = output_dir / "best_threshold.json"
    summary_path = output_dir / "threshold_summary.txt"
    roc_path = output_dir / "roc_curve.csv"
    pr_path = output_dir / "pr_curve.csv"
    zone_path = output_dir / "zone_policy.csv"

    sweep.to_csv(sweep_path, index=False, encoding="utf-8")
    roc_df.to_csv(roc_path, index=False, encoding="utf-8")
    pr_df.to_csv(pr_path, index=False, encoding="utf-8")

    best_payload = {
        "input": str(input_path),
        "score_col": args.score_col,
        "label_col": args.label_col,
        "positive_label": args.positive_label,
        "negative_label": args.negative_label,
        "rule": args.rule,
        "optimize_metric": args.optimize,
        "best_threshold": float(best["threshold"]),
        "metrics": {
            k: float(v) if isinstance(v, (np.floating, float)) else int(v)
            for k, v in best.items()
            if k != "threshold"
        },
        "n_samples": int(n_labeled),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }
    best_path.write_text(
        json.dumps(best_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    zone_lines = []
    if args.review_threshold is not None and args.block_threshold is not None:
        zone_df = build_zone_policy_table(
            scores=scores,
            y_true=y_true,
            positive_label=args.positive_label,
            negative_label=args.negative_label,
            rule=args.rule,
            review_threshold=args.review_threshold,
            block_threshold=args.block_threshold,
        )
        zone_df.to_csv(zone_path, index=False, encoding="utf-8")
        zone_lines = [
            "",
            "Zone policy",
            f"  review_threshold = {args.review_threshold:.6f}",
            f"  block_threshold  = {args.block_threshold:.6f}",
            zone_df.to_string(index=False),
        ]

    if not args.skip_plots and plt is not None:
        plot_curve(
            roc_df,
            x_col="fpr",
            y_col="tpr",
            title=f"ROC Curve (AUC={roc_auc:.4f})",
            x_label="False Positive Rate",
            y_label="True Positive Rate",
            output_path=output_dir / "roc_curve.png",
            diagonal=True,
        )
        plot_curve(
            pr_df.dropna(subset=["threshold_score"]),
            x_col="recall",
            y_col="precision",
            title=f"Precision-Recall Curve (AP={pr_auc:.4f})",
            x_label="Recall",
            y_label="Precision",
            output_path=output_dir / "pr_curve.png",
            diagonal=False,
        )

    top = sweep.head(max(args.top_k, 1))
    lines = [
        "Exact threshold search summary",
        f"Input           : {input_path}",
        f"Samples         : {n_labeled}",
        f"Optimize metric : {args.optimize}",
        f"Decision rule   : {args.positive_label} if score "
        f"{'>=' if args.rule == 'greater_equal' else '<='} threshold",
        f"ROC AUC         : {roc_auc:.4f}",
        f"PR AUC          : {pr_auc:.4f}",
        "",
        "Best threshold",
        f"  threshold          = {best_payload['best_threshold']:.6f}",
        f"  accuracy           = {best_payload['metrics']['accuracy']:.4f}",
        f"  balanced_accuracy  = {best_payload['metrics']['balanced_accuracy']:.4f}",
        f"  spam_precision     = {best_payload['metrics']['spam_precision']:.4f}",
        f"  spam_recall        = {best_payload['metrics']['spam_recall']:.4f}",
        f"  spam_f1            = {best_payload['metrics']['spam_f1']:.4f}",
        f"  specificity        = {best_payload['metrics']['specificity']:.4f}",
        f"  youden_j           = {best_payload['metrics']['youden_j']:.4f}",
        f"  tp/fp/tn/fn        = "
        f"{best_payload['metrics']['tp']}/"
        f"{best_payload['metrics']['fp']}/"
        f"{best_payload['metrics']['tn']}/"
        f"{best_payload['metrics']['fn']}",
        "",
        f"Files:",
        f"  {sweep_path}",
        f"  {best_path}",
        f"  {roc_path}",
        f"  {pr_path}",
        "",
        "Top thresholds",
        top.to_string(index=False),
        *zone_lines,
        "",
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
