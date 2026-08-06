from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PRIMARY_INPUT = Path(
    "data/analysis/mejai_matched_primary_carry_features.parquet"
)
SENSITIVITY_INPUT = Path(
    "data/analysis/mejai_matched_sensitivity_carry_features.parquet"
)

OUTPUT_DIR = Path("data/analysis/carry_state_balance")
PRIMARY_OUTPUT = OUTPUT_DIR / "primary_carry_state_balance.csv"
SENSITIVITY_OUTPUT = OUTPUT_DIR / "sensitivity_carry_state_balance.csv"
REPORT_OUTPUT = OUTPUT_DIR / "carry_state_balance_report.txt"

FEATURES = [
    "player_gold_diff_vs_role_opponent",
    "player_xp_diff_vs_role_opponent",
    "rest_of_team_gold_diff",
    "rest_of_team_xp_diff",
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
            values.loc[mask].astype(float),
            weights=weights.loc[mask].astype(float),
        )
    )


def weighted_variance(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna()

    if not mask.any():
        return np.nan

    x = values.loc[mask].astype(float).to_numpy()
    w = weights.loc[mask].astype(float).to_numpy()

    weight_sum = w.sum()
    if weight_sum <= 0:
        return np.nan

    mean = np.average(x, weights=w)
    denominator = weight_sum - np.square(w).sum() / weight_sum

    if denominator <= 0:
        return np.nan

    return float(
        np.sum(w * np.square(x - mean)) / denominator
    )


def standardized_mean_difference(
    treated_values: pd.Series,
    treated_weights: pd.Series,
    control_values: pd.Series,
    control_weights: pd.Series,
) -> float:
    treated_mean = weighted_mean(
        treated_values,
        treated_weights,
    )
    control_mean = weighted_mean(
        control_values,
        control_weights,
    )

    treated_variance = weighted_variance(
        treated_values,
        treated_weights,
    )
    control_variance = weighted_variance(
        control_values,
        control_weights,
    )

    pooled_sd = np.sqrt(
        (treated_variance + control_variance) / 2.0
    )

    if not np.isfinite(pooled_sd) or pooled_sd == 0:
        if np.isclose(
            treated_mean,
            control_mean,
            equal_nan=True,
        ):
            return 0.0

        return np.nan

    return float(
        (treated_mean - control_mean) / pooled_sd
    )


def balance_flag(smd: float) -> str:
    if not np.isfinite(smd):
        return "UNKNOWN"

    absolute_smd = abs(smd)

    if absolute_smd < 0.10:
        return "GOOD"

    if absolute_smd < 0.20:
        return "CHECK"

    return "POOR"


# ============================================================
# LOADING
# ============================================================

def load_sample(
    path: Path,
    sample_name: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{sample_name} carry-state file not found: {path}"
        )

    df = pd.read_parquet(path).copy()

    required = {
        "matched_set_id",
        "treatment",
        "matching_weight",
        *FEATURES,
    }
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"{sample_name} is missing required columns: {missing}"
        )

    df["treatment"] = pd.to_numeric(
        df["treatment"],
        errors="coerce",
    )
    df["matching_weight"] = pd.to_numeric(
        df["matching_weight"],
        errors="coerce",
    )

    for feature in FEATURES:
        df[feature] = pd.to_numeric(
            df[feature],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "matched_set_id",
            "treatment",
            "matching_weight",
        ]
    )
    df["treatment"] = df["treatment"].astype(int)

    invalid_treatment = ~df["treatment"].isin([0, 1])
    if invalid_treatment.any():
        raise ValueError(
            f"{sample_name} contains treatment values outside 0 and 1"
        )

    nonpositive_weights = df["matching_weight"] <= 0
    if nonpositive_weights.any():
        raise ValueError(
            f"{sample_name} contains non-positive matching weights"
        )

    return df.reset_index(drop=True)


# ============================================================
# BALANCE ANALYSIS
# ============================================================

def calculate_balance(
    df: pd.DataFrame,
    sample_name: str,
) -> pd.DataFrame:
    treated = df[df["treatment"] == 1].copy()
    controls = df[df["treatment"] == 0].copy()

    if treated.empty:
        raise ValueError(
            f"{sample_name} has no treated rows"
        )

    if controls.empty:
        raise ValueError(
            f"{sample_name} has no control rows"
        )

    rows = []

    for feature in FEATURES:
        treated_values = treated[feature]
        control_values = controls[feature]

        treated_mean = weighted_mean(
            treated_values,
            treated["matching_weight"],
        )
        control_mean = weighted_mean(
            control_values,
            controls["matching_weight"],
        )

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
                "treated_non_missing_count": int(
                    treated_values.notna().sum()
                ),
                "control_non_missing_count": int(
                    control_values.notna().sum()
                ),
                "treated_non_missing_ratio": float(
                    treated_values.notna().mean()
                ),
                "control_non_missing_ratio": float(
                    control_values.notna().mean()
                ),
                "treated_mean": treated_mean,
                "control_mean": control_mean,
                "mean_difference": (
                    treated_mean - control_mean
                ),
                "standardized_mean_difference": smd,
                "absolute_standardized_mean_difference": (
                    abs(smd) if np.isfinite(smd) else np.nan
                ),
                "balance_flag": balance_flag(smd),
            }
        )

    balance = pd.DataFrame(rows)
    balance = balance.sort_values(
        "absolute_standardized_mean_difference",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    return balance


# ============================================================
# REPORTING
# ============================================================

def format_balance_table(
    balance: pd.DataFrame,
) -> str:
    display = balance[
        [
            "feature",
            "treated_mean",
            "control_mean",
            "standardized_mean_difference",
            "balance_flag",
        ]
    ].copy()

    for column in [
        "treated_mean",
        "control_mean",
        "standardized_mean_difference",
    ]:
        display[column] = display[column].map(
            lambda value: (
                f"{value:.4f}"
                if np.isfinite(value)
                else "NA"
            )
        )

    return display.to_string(index=False)


def build_report(
    primary_balance: pd.DataFrame,
    sensitivity_balance: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "=" * 80,
            "CARRY-STATE FEATURE BALANCE",
            "=" * 80,
            "",
            "Interpretation thresholds:",
            "  |SMD| < 0.10  = GOOD",
            "  0.10-0.20     = CHECK",
            "  >= 0.20       = POOR",
            "",
            "PRIMARY",
            "-" * 80,
            format_balance_table(primary_balance),
            "",
            "SENSITIVITY",
            "-" * 80,
            format_balance_table(sensitivity_balance),
        ]
    )


def print_sample_summary(
    df: pd.DataFrame,
    sample_name: str,
) -> None:
    log(
        f"{sample_name} rows loaded: {len(df):,}"
    )
    log(
        f"{sample_name} matched sets: "
        f"{df['matched_set_id'].nunique():,}"
    )
    log(
        f"{sample_name} treated rows: "
        f"{int((df['treatment'] == 1).sum()):,}"
    )
    log(
        f"{sample_name} control rows: "
        f"{int((df['treatment'] == 0).sum()):,}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    log("=" * 80)
    log("CARRY-STATE FEATURE BALANCE")
    log("=" * 80)

    primary = load_sample(
        PRIMARY_INPUT,
        "Primary",
    )
    sensitivity = load_sample(
        SENSITIVITY_INPUT,
        "Sensitivity",
    )

    print_sample_summary(
        primary,
        "Primary",
    )
    print_sample_summary(
        sensitivity,
        "Sensitivity",
    )

    primary_balance = calculate_balance(
        primary,
        "primary",
    )
    sensitivity_balance = calculate_balance(
        sensitivity,
        "sensitivity",
    )

    report = build_report(
        primary_balance,
        sensitivity_balance,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary_balance.to_csv(
        PRIMARY_OUTPUT,
        index=False,
    )
    sensitivity_balance.to_csv(
        SENSITIVITY_OUTPUT,
        index=False,
    )
    REPORT_OUTPUT.write_text(
        report,
        encoding="utf-8",
    )

    log("")
    log(report)
    log("")
    log(f"[SAVED] {PRIMARY_OUTPUT}")
    log(f"[SAVED] {SENSITIVITY_OUTPUT}")
    log(f"[SAVED] {REPORT_OUTPUT}")
    log("")
    log("[PASSED] CARRY-STATE FEATURE BALANCE COMPLETE")


if __name__ == "__main__":
    main()