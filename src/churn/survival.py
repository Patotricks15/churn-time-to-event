from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def kaplan_meier(durations, events) -> pd.DataFrame:
    """Kaplan-Meier estimate with one row per distinct event time."""
    t = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=bool)
    rows = []
    survival = 1.0
    for time in np.unique(t[e]):
        at_risk = int(np.sum(t >= time))
        deaths = int(np.sum((t == time) & e))
        survival *= 1 - deaths / at_risk
        rows.append((time, at_risk, deaths, survival))
    return pd.DataFrame(rows, columns=["time", "at_risk", "events", "survival"])


def survival_at(km: pd.DataFrame, t: float) -> float:
    """S(t) from a Kaplan-Meier table. Equals 1.0 before the first event."""
    past = km[km["time"] <= t]
    return float(past["survival"].iloc[-1]) if len(past) else 1.0


@dataclass
class CoxFit:
    columns: list[str]
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray

    def linear_predictor(self, df: pd.DataFrame) -> np.ndarray:
        z = (df[self.columns].to_numpy(float) - self.mean) / self.scale
        return z @ self.coef


def fit_cox(df: pd.DataFrame, columns: list[str], duration_col: str, event_col: str,
            l2: float = 1e-4) -> CoxFit:
    """Cox proportional hazards with Breslow ties, fitted by L-BFGS on standardised covariates."""
    x = df[columns].to_numpy(float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale

    t = df[duration_col].to_numpy(float)
    e = df[event_col].to_numpy(bool)
    order = np.argsort(t, kind="stable")
    t, z, e = t[order], z[order], e[order]

    event_idx = np.flatnonzero(e)
    event_times = np.unique(t[event_idx])
    group = np.searchsorted(event_times, t[event_idx])
    deaths = np.bincount(group, minlength=len(event_times)).astype(float)
    start = np.searchsorted(t, event_times, side="left")
    event_x_sum = np.zeros((len(event_times), z.shape[1]))
    np.add.at(event_x_sum, group, z[event_idx])
    total_event_x = event_x_sum.sum(axis=0)

    def negative_penalised_loglik(beta: np.ndarray):
        eta = z @ beta
        shift = eta.max()
        w = np.exp(eta - shift)
        risk_sum = np.cumsum(w[::-1])[::-1][start]
        risk_x = np.cumsum((w[:, None] * z)[::-1], axis=0)[::-1][start]
        loglik = eta[event_idx].sum() - np.sum(deaths * (np.log(risk_sum) + shift))
        grad = total_event_x - np.sum((deaths / risk_sum)[:, None] * risk_x, axis=0)
        value = -loglik + 0.5 * l2 * beta @ beta
        return value, -grad + l2 * beta

    result = minimize(negative_penalised_loglik, np.zeros(z.shape[1]), jac=True, method="L-BFGS-B")
    return CoxFit(columns=list(columns), mean=mean, scale=scale, coef=result.x)


def concordance_index(durations, events, risk) -> float:
    """Harrell's C. A pair is comparable when the earlier time is an event. Higher risk should mean earlier."""
    t = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=bool)
    r = np.asarray(risk, dtype=float)
    concordant = 0.0
    comparable = 0
    for i in np.flatnonzero(e):
        later = t > t[i]
        comparable += int(later.sum())
        concordant += float(np.sum(r[i] > r[later]) + 0.5 * np.sum(r[i] == r[later]))
    return concordant / comparable if comparable else float("nan")
