import numpy as np
import pandas as pd
import pytest

from churn.survival import concordance_index, fit_cox, kaplan_meier, survival_at


def test_kaplan_meier_matches_hand_computation():
    # Durations 2, 3, 3, 5 with the 3 on day 3 censored. Survival: 3/4, then 3/4 * 2/3, then 0.
    km = kaplan_meier([2, 3, 3, 5], [1, 1, 0, 1])
    assert km["time"].tolist() == [2, 3, 5]
    assert km["at_risk"].tolist() == [4, 3, 1]
    assert km["survival"].tolist() == pytest.approx([0.75, 0.5, 0.0])


def test_survival_before_first_event_is_one():
    km = kaplan_meier([5, 6], [1, 1])
    assert survival_at(km, 4) == 1.0
    assert survival_at(km, 5) == pytest.approx(0.5)


def test_concordance_is_one_when_higher_risk_fails_first():
    assert concordance_index([1, 2, 3], [1, 1, 1], [3, 2, 1]) == 1.0


def test_concordance_is_zero_when_ordering_is_reversed():
    assert concordance_index([1, 2, 3], [1, 1, 1], [1, 2, 3]) == 0.0


def test_concordance_counts_ties_as_half():
    assert concordance_index([1, 2], [1, 1], [0.5, 0.5]) == 0.5


def test_concordance_ignores_pairs_where_earlier_time_is_censored():
    # Only the pair (0, 1) and (0, 2) are comparable, since subject 0 has an event at time 1.
    # Subject 1 is censored, so the pair (1, 2) is not comparable.
    assert concordance_index([1, 2, 3], [1, 0, 1], [2, 0, 1]) == 1.0


def test_cox_recovers_true_coefficients_on_seeded_synthetic_data():
    rng = np.random.default_rng(0)
    n = 3000
    x1 = rng.normal(size=n)
    x2 = rng.integers(0, 2, size=n).astype(float)
    hazard = np.exp(0.8 * x1 - 0.5 * x2)
    latent = rng.exponential(1 / hazard)
    censor = rng.uniform(0, 3, size=n)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "duration": np.minimum(latent, censor),
            "event": (latent <= censor).astype(int),
        }
    )
    fit = fit_cox(df, ["x1", "x2"], "duration", "event", l2=1e-6)
    # Coefficients are per standard deviation. x1 has SD near 1; x2 is binary with SD near 0.5.
    expected = [0.8 * df["x1"].std(), -0.5 * df["x2"].std()]
    assert fit.coef == pytest.approx(expected, abs=0.1)
    assert concordance_index(df["duration"], df["event"], fit.linear_predictor(df)) > 0.6
