import argparse
from pathlib import Path

from churn.config import H_DAYS, MODEL_VERSION, TEST_SNAPSHOTS, TRAIN_SNAPSHOTS
from churn.data import load_transactions
from churn.pipeline import run_survival
from churn.plots import plot_hazard_ratios, plot_km_by_segment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CH03 survival analysis on Online Retail II.")
    parser.add_argument("--input-file", required=True, help="Path to online_retail_II.xlsx or a CSV export.")
    parser.add_argument("--out-dir", default="reports", help="Directory for CSV outputs and figures.")
    parser.add_argument("--h", type=int, default=H_DAYS, help="Churn window in days.")
    args = parser.parse_args()

    tx = load_transactions(args.input_file)
    result = run_survival(tx, h_days=args.h)

    out = Path(args.out_dir)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    result["metrics"].to_csv(out / "ch03_metrics.csv", index=False)
    result["h_sensitivity"].to_csv(out / "ch03_h_sensitivity.csv", index=False)
    result["segments"].to_csv(out / "ch03_segments.csv", index=False)
    result["cox_coefficients"].to_csv(out / "ch03_cox_coefficients.csv", index=False)

    snapshot = result["latest_snapshot"]
    subtitle = (
        f"Online Retail II, snapshot {snapshot}, H={args.h} days, "
        f"train {TRAIN_SNAPSHOTS[0]} to {TRAIN_SNAPSHOTS[-1]}, {MODEL_VERSION}"
    )
    plot_km_by_segment(
        result["km_curves"],
        f"Survival by order frequency. {subtitle}",
        args.h,
        figures / "km_by_segment.png",
    )
    plot_hazard_ratios(
        result["cox_coefficients"],
        f"Cox model covariates. {subtitle}",
        figures / "cox_hazard_ratios.png",
    )
    print(result["metrics"].to_string(index=False))
    print(f"Test snapshots: {', '.join(TEST_SNAPSHOTS)}")


if __name__ == "__main__":
    main()
