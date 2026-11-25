# churn-time-to-event (CH03)

**Question:** When will a customer churn, not only whether?

This repository fits Kaplan-Meier survival curves and a Cox proportional-hazards model on time to next purchase, using Online Retail II. It reuses the CH01 snapshot dates, so results can be compared with the CH01 baseline on identical cuts.

**Status:** the code is tested with unit tests and a smoke run on synthetic data. **Results are not yet produced.** The notebook has not been executed, because the data is not on this machine. Headline numbers will be added after a Kaggle run.

## Data

- **Dataset:** Online Retail II UCI, Kaggle slug [`mashlyn/online-retail-ii-uci`](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci).
- **License (per Kaggle metadata):** CC0.
- **Read path:** `/kaggle/input/online-retail-ii-uci/` (see [data/README.md](data/README.md)).
- **Metadata read:** 2026-11-08 (plan Appendix A). No files were downloaded.

## Time-to-event definition

- **Snapshot:** a customer is observed at a snapshot date using only purchases strictly before it.
- **Duration:** whole days from the snapshot to the customer's next purchase.
- **Event:** the next purchase falls at or after `H` days, which means churn in the CH01 sense (no purchase in `[snapshot, snapshot + H)`).
- **Censoring:** a customer with no purchase by the end of the data is censored at the follow-up length (`data_end − snapshot`). Follow-up is at least `H`, so censoring never occurs before `H`. The Kaplan-Meier value `S(H − 1)` therefore equals the CH01 churn rate exactly. The code checks this.
- **Cleaning:** the same as CH01. Cancelled invoices (`C` prefix) and rows with non-positive price or quantity are removed.
- **Population:** customers with at least one purchase before the snapshot. Single-purchase customers are reported separately and are excluded from the main Cox fit.
- **Leakage guard:** features use only data before the snapshot. A test checks this. Another check raises an error if a training label window overlaps the test period.

## Snapshots and validation

- **Train snapshots:** 2010-03-01, 2010-06-01, 2010-09-01, 2010-12-01. The Cox model is fitted on these, pooled.
- **Test snapshots:** 2011-03-01, 2011-06-01, 2011-09-01. The headline result is the latest test snapshot, 2011-09-01.
- The data must extend at least `H` days past the last test snapshot. The run stops with an error if it does not.

## Models and metrics

- **Kaplan-Meier:** survival curves by order-frequency band (`FREQUENCY_BANDS` in `pipeline.py`).
- **Cox model:** covariates are `recency_days`, `mean_gap_days`, `log_frequency` and `log_monetary`. Coefficients are per standard deviation, with a small ridge penalty (`COX_L2_PENALTY`). Ties use the Breslow method. Hazard ratios are `exp(coef)`.
- **Churn scores:** the Cox linear predictor is a hazard of purchase, so the churn score is its negation.
- **Concordance index (C-index):** Harrell's C on the purchase-time ordering. The recency-only baseline (`−recency`) is reported next to it.
- **Classification at `H`:** ROC-AUC and PR-AUC against the churn label, for the Cox score and the recency baseline (scikit-learn).
- **Sensitivity:** the same metrics for `H` in {30, 60, 90} days.

## Implementation note

The plan's method uses `lifelines`, but it was not added because that dependency was not approved. Kaplan-Meier, the Cox fit (Breslow ties, L-BFGS) and Harrell's C are implemented in `src/churn/survival.py` with numpy and scipy. The unit tests check them against hand-computed values and a seeded recovery test. Replacing them with `lifelines` is a possible follow-up.

## Outputs

Running the script writes to `reports/`:

- `ch03_metrics.csv`: per-snapshot C-index, classification metrics and KM churn rate.
- `ch03_h_sensitivity.csv`: the same metrics for each `H`.
- `ch03_segments.csv`: KM survival by frequency band.
- `ch03_cox_coefficients.csv`: coefficients per standard deviation and their hazard ratios.
- `figures/km_by_segment.png` and `figures/cox_hazard_ratios.png`. Titles state the dataset, snapshot, `H`, and model version.

## Tests

- `tests/test_labels.py`: the label rule and its boundary, censoring, single-purchase customers, the KM identity and the insufficient-follow-up error.
- `tests/test_snapshot.py`: no leakage from after the snapshot, past-only mean gap, and the label-gap guard.
- `tests/test_survival.py`: a hand-computed KM example, survival before the first event, C-index cases (ties, censoring, comparability) and Cox recovery on seeded data.

## Results

**Not yet run on the real dataset.** The tests and a smoke run on randomly generated data only confirm that the code runs. Those numbers are not results and are not reported here.

## Reproduce

```bash
make install
make test
make reproduce INPUT=/kaggle/input/online-retail-ii-uci/<file>.xlsx
```

The notebook in `notebooks/` runs the same steps inside a Kaggle Notebook.
