import numpy as np
import pandas as pd

from churn.config import COX_COVARIATES


def rfm_features(past: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """RFM-style features from transactions strictly before the snapshot."""
    grouped = past.groupby("customer_id")
    out = pd.DataFrame(
        {
            "recency_days": (snapshot - grouped["invoice_date"].max()).dt.days,
            "frequency": grouped["invoice"].nunique(),
            "monetary": grouped["revenue"].sum(),
            "tenure_days": (snapshot - grouped["invoice_date"].min()).dt.days,
        }
    )
    return out


def mean_gap_days(past: pd.DataFrame) -> pd.Series:
    """Mean days between consecutive orders. NaN for customers with a single order."""
    orders = past.groupby(["customer_id", "invoice"], as_index=False)["invoice_date"].min()
    orders = orders.sort_values(["customer_id", "invoice_date"])
    gaps = orders.groupby("customer_id")["invoice_date"].diff().dt.days
    return gaps.groupby(orders["customer_id"]).mean().rename("mean_gap_days")


def survival_features(past: pd.DataFrame, snapshot: pd.Timestamp) -> pd.DataFrame:
    """Covariates for the Cox model. Every value uses transactions strictly before the snapshot."""
    panel = rfm_features(past, pd.Timestamp(snapshot))
    panel = panel.join(mean_gap_days(past))
    panel["log_frequency"] = np.log1p(panel["frequency"])
    panel["log_monetary"] = np.log1p(panel["monetary"].clip(lower=0))
    return panel[["frequency", "monetary", "tenure_days", *COX_COVARIATES]]
