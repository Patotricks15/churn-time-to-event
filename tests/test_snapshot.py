import pandas as pd
import pytest

from churn.labels import build_time_to_event, check_label_gap

H = 30
SNAPSHOT = "2010-03-01"
FEATURES = ["recency_days", "frequency", "monetary", "tenure_days", "mean_gap_days", "log_frequency", "log_monetary"]


def make_tx(rows):
    df = pd.DataFrame(rows, columns=["customer_id", "invoice", "invoice_date", "revenue"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


BASE_ROWS = [
    ("A", "a1", "2010-02-01", 10), ("A", "a2", "2010-02-15", 20),
    ("A", "a3", "2010-03-20", 30),
    ("B", "b1", "2010-02-01", 10), ("B", "b2", "2010-02-20", 10),
    ("B", "b3", "2010-04-15", 10),
    ("F", "f1", "2010-02-01", 10), ("F", "f2", "2010-05-01", 10),
]


def test_features_ignore_transactions_after_snapshot():
    base = make_tx(BASE_ROWS)
    # Add future orders for A and B and inflate their future revenue. Features must not change.
    altered = make_tx(
        BASE_ROWS
        + [("A", "a9", "2010-03-25", 999), ("B", "b9", "2010-03-02", 500)]
    )
    a = build_time_to_event(base, SNAPSHOT, H).set_index("customer_id")
    b = build_time_to_event(altered, SNAPSHOT, H).set_index("customer_id")
    pd.testing.assert_frame_equal(a.loc[:, FEATURES], b.loc[:, FEATURES])


def test_mean_gap_uses_only_past_orders():
    tx = make_tx(BASE_ROWS)
    panel = build_time_to_event(tx, SNAPSHOT, H).set_index("customer_id")
    # A's past orders are on Feb 1, Feb 15 (14 days apart). The March 20 order is in the future.
    assert panel.loc["A", "mean_gap_days"] == pytest.approx(14.0)


def test_label_gap_rejects_test_snapshot_inside_training_window():
    with pytest.raises(ValueError, match="Move the test snapshots later"):
        check_label_gap(["2010-03-01", "2010-06-01"], ["2010-08-01"], h_days=90)


def test_label_gap_accepts_test_snapshot_after_training_window():
    check_label_gap(["2010-03-01", "2010-06-01"], ["2010-09-01"], h_days=90)
