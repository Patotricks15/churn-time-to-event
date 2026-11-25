MODEL_VERSION = "CH03 v0.1"

# Churn window in days. A customer churns if no purchase occurs in [snapshot, snapshot + H_DAYS).
H_DAYS = 90
H_SENSITIVITY_DAYS = [30, 60, 90]

# Same snapshot dates as CH01, so the two repositories compare on identical cuts.
TRAIN_SNAPSHOTS = ["2010-03-01", "2010-06-01", "2010-09-01", "2010-12-01"]
TEST_SNAPSHOTS = ["2011-03-01", "2011-06-01", "2011-09-01"]

# Ridge penalty on the Cox coefficients. Keeps the optimiser stable with correlated RFM covariates.
COX_L2_PENALTY = 1e-4

COX_COVARIATES = ["recency_days", "mean_gap_days", "log_frequency", "log_monetary"]
RANDOM_STATE = 42
