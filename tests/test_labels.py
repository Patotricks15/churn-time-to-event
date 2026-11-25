import pandas as pd
import pytest

from churn.labels import build_time_to_event
from churn.survival import kaplan_meier, survival_at

H = 30
SNAPSHOT = "2010-03-01"


def make_tx(rows):
    df = pd.DataFrame(rows, columns=["customer_id", "invoice", "invoice_date", "revenue"])
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    return df


@pytest.fixture
def tx():
    # Data ends 2010-05-01 (customer F's last order), so follow-up after the snapshot is 61 days.
    return make_tx(
        [
            ("A", "a1", "2010-02-01", 10), ("A", "a2", "2010-02-15", 10), ("A", "a3", "2010-03-20", 10),
            ("B", "b1", "2010-02-01", 10), ("B", "b2", "2010-02-20", 10), ("B", "b3", "2010-04-15", 10),
            ("C", "c1", "2010-02-10", 10), ("C", "c2", "2010-02-11", 10),
            ("D", "d1", "2010-02-01", 10), ("D", "d2", "2010-02-05", 10), ("D", "d3", "2010-03-31", 10),
            ("E", "e1", "2010-02-01", 10), ("E", "e2", "2010-02-05", 10), ("E", "e3", "2010-03-01", 10),
            ("F", "f1", "2010-02-01", 10), ("F", "f2", "2010-02-02", 10), ("F", "f3", "2010-05-01", 10),
            ("I", "i1", "2010-02-03", 10),
            ("G", "g1", "2010-03-05", 10),
        ]
    )


@pytest.fixture
def panel(tx):
    return build_time_to_event(tx, SNAPSHOT, H).set_index("customer_id")


def test_churn_follows_no_purchase_in_window(panel):
    # A buys on day 19 (inside the window), so not churned. B buys on day 45, so churned.
    assert panel.loc["A", "churned"] == 0
    assert panel.loc["B", "churned"] == 1


def test_purchase_on_window_boundary_counts_as_churn(panel):
    # D's next order is on day 30, which is outside [0, 30), so D churned.
    assert panel.loc["D", "duration_days"] == 30
    assert panel.loc["D", "churned"] == 1


def test_purchase_on_snapshot_day_is_not_churn(panel):
    assert panel.loc["E", "duration_days"] == 0
    assert panel.loc["E", "churned"] == 0


def test_no_future_purchase_is_censored_at_data_end(panel):
    assert panel.loc["C", "event"] == 0
    assert panel.loc["C", "duration_days"] == 61
    assert panel.loc["C", "churned"] == 1


def test_observed_next_purchase_sets_duration_and_event(panel):
    assert panel.loc["A", "event"] == 1
    assert panel.loc["A", "duration_days"] == 19


def test_single_order_customers_are_flagged_and_customers_without_history_excluded(panel):
    assert bool(panel.loc["I", "single_order"]) is True
    assert bool(panel.loc["A", "single_order"]) is False
    assert "G" not in panel.index


def test_km_at_h_minus_one_equals_churn_rate(panel):
    km = kaplan_meier(panel["duration_days"], panel["event"])
    assert survival_at(km, H - 1) == pytest.approx(panel["churned"].mean())


def test_rejects_snapshot_without_full_window(tx):
    with pytest.raises(ValueError, match="fewer than H"):
        build_time_to_event(tx, "2010-04-20", H)
