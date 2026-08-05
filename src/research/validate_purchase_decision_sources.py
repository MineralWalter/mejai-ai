from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PRIMARY_FILE = Path("data/analysis/mejai_matched_primary_features.parquet")
SENSITIVITY_FILE = Path("data/analysis/mejai_matched_sensitivity_features.parquet")

OUTPUT_DIR = Path("data/analysis/purchase_feature_balance")
PRIMARY_OUTPUT = OUTPUT_DIR / "primary_compact_feature_balance.csv"
SENSITIVITY_OUTPUT = OUTPUT_DIR / "sensitivity_compact_feature_balance.csv"
REPORT_OUTPUT = OUTPUT_DIR / "compact_purchase_feature_balance_report.txt"

FEATURES = [
    "purchase_time_minutes",
    "dark_seal_purchased_before_observation",
    "kills_last_5m",
    "deaths_last_5m",
    "assists_last_5m",
    "seconds_since_last_death",
]


# ============================================================
# HELPERS
# ============================================================

def log(message: str) -> None:
    print(message)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if not mask.any():
        return np.nan

    return float(
        np.average(
            values[mask].astype(float),
            weights=weights[mask].astype(float),
        )
    )


def weighted_variance(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()
    if not mask.any():
        return np.nan

    x = values[mask].astype(float).to_numpy()
    w = weights[mask].astype(float).to_numpy()
    mean = np.average(x, weights=w)

    weight_sum = w.sum()
    denominator = weight_sum - np.square(w).sum() / weight_sum
    if denominator <= 0:
        return np.nan

    return float(np.sum(w * np.square(x - mean)) / denominator)


def standardized_mean_difference(
    treated_values: pd.Series,
    treated_weights: pd.Series,
    control_values: pd.Series,
    control_weights: pd.Series,
) -> float:
    treated_mean = weighted_mean(treated_values, treated_weights)
    control_mean = weighted_mean(control_values, control_weights)
    treated_variance = weighted_variance(treated_values, treated_weights)
    control_variance = weighted_variance(control_values, control_weights)

    pooled_sd = np.sqrt((treated_variance + control_variance) / 2.0)
    if not np.isfinite(pooled_sd) or pooled_sd == 0:
        return 0.0 if np.isclose(treated_mean, control_mean, equal_nan=True) else np.nan

    return float((treated_mean - control_mean) / pooled_sd)


def balance_flag(smd: float) -> str:
    if not np.isfinite(smd):
        return "UNKNOWN"

    absolute = abs(smd)
    if absolute < 0.10:
        return "GOOD"
    if absolute < 0.20:
        return "CHECK"
    return "POOR"


# ============================================================
# ANALYSIS
# ============================================================

def load_sample(path: Path, sample_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{sample_name} file not found: {path}")

    df = pd.read_parquet(path)
    required = {"treatment", "matching_weight", "matched_set_id", *FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{sample_name} is missing required columns: {missing}")

    df = df.copy()
    df["treatment"] = pd.to_numeric(df["treatment"], errors="coerce")
    df["matching_weight"] = pd.to_numeric(df["matching_weight"], errors="coerce")
    df = df.dropna(subset=["treatment", "matching_weight", "matched_set_id"])
    df["treatment"] = df["treatment"].astype(int)

    return df.reset_index(drop=True)


def calculate_balance(df: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    treated = df[df["treatment"] == 1].copy()
    controls = df[df["treatment"] == 0].copy()
    rows = []

    for feature in FEATURES:
        treated_values = pd.to_numeric(treated[feature], errors="coerce")
        control_values = pd.to_numeric(controls[feature], errors="coerce")

        treated_mean = weighted_mean(treated_values, treated["matching_weight"])
        control_mean = weighted_mean(control_values, controls["matching_weight"])
        smd = standardized_mean_difference(
            treated_values,
            treated["matching_weight"],
            control_values,
            controls["matching_weight"],
        )

        rows.append(
            {
                "sample": sample_name,
                "feature": feature,
                "treated_non_missing_ratio": float(treated_values.notna().mean()),
                "control_non_missing_ratio": float(control_values.notna().mean()),
                "treated_mean": treated_mean,
                "control_mean": control_mean,
                "mean_difference": treated_mean - control_mean,
                "standardized_mean_difference": smd,
                "balance_flag": balance_flag(smd),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# REPORTING
# ============================================================

def format_balance_table(balance: pd.DataFrame) -> str:
    display = balance[
        [
            "feature",
            "treated_mean",
            "control_mean",
            "standardized_mean_difference",
            "balance_flag",
        ]
    ].copy()

    display = display.sort_values(
        "standardized_mean_difference",
        key=lambda series: series.abs(),
        ascending=False,
        na_position="last",
    )

    for column in ["treated_mean", "control_mean", "standardized_mean_difference"]:
        display[column] = display[column].map(
            lambda value: f"{value:.4f}" if np.isfinite(value) else "NA"
        )

    return display.to_string(index=False)


def make_report(primary: pd.DataFrame, sensitivity: pd.DataFrame) -> str:
    return "\n".join(
        [
            "=" * 80,
            "COMPACT PURCHASE FEATURE BALANCE",
            "=" * 80,
            "",
            "Interpretation thresholds:",
            "  |SMD| < 0.10  = GOOD",
            "  0.10-0.20     = CHECK",
            "  >= 0.20       = POOR",
            "",
            "PRIMARY",
            "-" * 80,
            format_balance_table(primary),
            "",
            "SENSITIVITY",
            "-" * 80,
            format_balance_table(sensitivity),
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    log("=" * 80)
    log("COMPACT PURCHASE FEATURE BALANCE")
    log("=" * 80)

    primary = load_sample(PRIMARY_FILE, "primary")
    sensitivity = load_sample(SENSITIVITY_FILE, "sensitivity")

    log(f"Primary rows loaded: {len(primary):,}")
    log(f"Primary matched sets: {primary['matched_set_id'].nunique():,}")
    log(f"Sensitivity rows loaded: {len(sensitivity):,}")
    log(f"Sensitivity matched sets: {sensitivity['matched_set_id'].nunique():,}")

    primary_balance = calculate_balance(primary, "primary")
    sensitivity_balance = calculate_balance(sensitivity, "sensitivity")
    report = make_report(primary_balance, sensitivity_balance)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    primary_balance.to_csv(PRIMARY_OUTPUT, index=False)
    sensitivity_balance.to_csv(SENSITIVITY_OUTPUT, index=False)
    REPORT_OUTPUT.write_text(report, encoding="utf-8")

    log("")
    log(report)
    log("")
    log(f"[SAVED] {PRIMARY_OUTPUT}")
    log(f"[SAVED] {SENSITIVITY_OUTPUT}")
    log(f"[SAVED] {REPORT_OUTPUT}")
    log("")
    log("[PASSED] COMPACT PURCHASE FEATURE BALANCE COMPLETE")


if __name__ == "__main__":
    main()