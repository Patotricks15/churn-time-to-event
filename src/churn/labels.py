import pandas as pd

from churn.features import survival_features


def build_time_to_event(tx: pd.DataFrame, snapshot: str | pd.Timestamp, h_days: int) -> pd.DataFrame:
    """One row per customer with at least one order before `snapshot`.

    `duration_days` is the whole number of days from the snapshot to the next purchase. Customers
    with no purchase by the end of the data are censored at the number of days to the last date.
    `churned` follows the CH01 rule: no purchase in [snapshot, snapshot + h_days).
    """
    snapshot = pd.Timestamp(snapshot).normalize()
    data_end = tx["invoice_date"].max().normalize()
    follow_up_days = (data_end - snapshot).days
    if follow_up_days < h_days:
        raise ValueError(
            f"Snapshot {snapshot.date()} has {follow_up_days} days of follow-up, "
            f"fewer than H={h_days}."
        )

    past = tx[tx["invoice_date"] < snapshot]
    future = tx[tx["invoice_date"] >= snapshot]
    next_purchase = (
        future.assign(day=future["invoice_date"].dt.normalize())
        .groupby("customer_id")["day"]
        .min()
    )

    panel = survival_features(past, snapshot)
    next_days = (next_purchase.reindex(panel.index) - snapshot).dt.days
    observed = next_days.notna()
    panel["event"] = observed.astype(int)
    panel["duration_days"] = next_days.fillna(follow_up_days).astype(int)
    panel["churned"] = (panel["duration_days"] >= h_days).astype(int)
    panel["single_order"] = panel["frequency"] == 1
    panel["snapshot_date"] = snapshot
    return panel.rename_axis("customer_id").reset_index()


def check_label_gap(train_snapshots, test_snapshots, h_days: int) -> None:
    """Training label windows must end before the first test snapshot starts."""
    last_label_end = max(pd.Timestamp(s) for s in train_snapshots) + pd.Timedelta(days=h_days)
    first_test = min(pd.Timestamp(s) for s in test_snapshots)
    if last_label_end > first_test:
        raise ValueError(
            f"Training label window ends {last_label_end.date()}, after the first test snapshot "
            f"{first_test.date()}. Move the test snapshots later by at least H={h_days} days."
        )
