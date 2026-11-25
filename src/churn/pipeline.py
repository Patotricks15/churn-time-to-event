import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from churn.config import (
    COX_COVARIATES,
    COX_L2_PENALTY,
    H_DAYS,
    H_SENSITIVITY_DAYS,
    TEST_SNAPSHOTS,
    TRAIN_SNAPSHOTS,
)
from churn.labels import build_time_to_event, check_label_gap
from churn.survival import concordance_index, fit_cox, kaplan_meier, survival_at

FREQUENCY_BANDS = {"2 orders": (2, 2), "3-4 orders": (3, 4), "5+ orders": (5, np.inf)}
SEGMENT_CHECKPOINTS = [30, 60, 90]


def _main_population(panel: pd.DataFrame) -> pd.DataFrame:
    """Customers with at least two orders. Single-order customers are reported separately."""
    return panel[~panel["single_order"]].copy()


def run_survival(
    tx: pd.DataFrame,
    train_snapshots=TRAIN_SNAPSHOTS,
    test_snapshots=TEST_SNAPSHOTS,
    h_days: int = H_DAYS,
    l2: float = COX_L2_PENALTY,
) -> dict:
    check_label_gap(train_snapshots, test_snapshots, h_days)

    train = pd.concat([build_time_to_event(tx, s, h_days) for s in train_snapshots])
    fit = fit_cox(_main_population(train), COX_COVARIATES, "duration_days", "event", l2=l2)

    rows = []
    for s in test_snapshots:
        panel = build_time_to_event(tx, s, h_days)
        main = _main_population(panel)
        durations, events = main["duration_days"], main["event"]
        # Higher `hazard` means a shorter time to next purchase. Churn is the opposite, so churn scores are negated.
        hazard = fit.linear_predictor(main)
        km = kaplan_meier(durations, events)
        rows.append(
            {
                "snapshot": pd.Timestamp(s).date(),
                "n_customers": len(panel),
                "n_single_order": int(panel["single_order"].sum()),
                "n_main": len(main),
                "churn_rate_main": main["churned"].mean(),
                "km_churn_rate_at_h": survival_at(km, h_days - 1),
                "c_index_cox": concordance_index(durations, events, hazard),
                "c_index_recency_baseline": concordance_index(durations, events, -main["recency_days"]),
                "roc_auc_cox_vs_churn": roc_auc_score(main["churned"], -hazard),
                "pr_auc_cox_vs_churn": average_precision_score(main["churned"], -hazard),
                "roc_auc_recency_vs_churn": roc_auc_score(main["churned"], main["recency_days"]),
                "pr_auc_recency_vs_churn": average_precision_score(main["churned"], main["recency_days"]),
            }
        )
    metrics = pd.DataFrame(rows)

    latest = _main_population(build_time_to_event(tx, test_snapshots[-1], h_days))
    segments, curves = _segment_survival(latest)
    sensitivity = _h_sensitivity(tx, test_snapshots[-1])

    coefficients = pd.DataFrame(
        {
            "covariate": fit.columns,
            "coef_per_sd": fit.coef,
            "hazard_ratio_per_sd": np.exp(fit.coef),
        }
    )
    return {
        "metrics": metrics,
        "h_sensitivity": sensitivity,
        "segments": segments,
        "km_curves": curves,
        "cox_coefficients": coefficients,
        "latest_snapshot": pd.Timestamp(test_snapshots[-1]).date(),
    }


def _segment_survival(main: pd.DataFrame):
    rows, curves = [], {}
    for label, (lo, hi) in FREQUENCY_BANDS.items():
        seg = main[(main["frequency"] >= lo) & (main["frequency"] <= hi)]
        km = kaplan_meier(seg["duration_days"], seg["event"])
        curves[label] = km
        row = {"segment": label, "n_customers": len(seg)}
        for t in SEGMENT_CHECKPOINTS:
            row[f"survival_at_{t}d"] = survival_at(km, t - 1)
        rows.append(row)
    return pd.DataFrame(rows), curves


def _h_sensitivity(tx: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    rows = []
    for h in H_SENSITIVITY_DAYS:
        main = _main_population(build_time_to_event(tx, snapshot, h))
        km = kaplan_meier(main["duration_days"], main["event"])
        rows.append(
            {
                "h_days": h,
                "churn_rate_main": main["churned"].mean(),
                "km_churn_rate_at_h": survival_at(km, h - 1),
            }
        )
    return pd.DataFrame(rows)
