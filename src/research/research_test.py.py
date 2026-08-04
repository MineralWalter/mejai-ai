from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path(
    "data/analysis/mejai_research_dataset.parquet"
)

NUMERIC_FEATURES = [
    "purchase_time_seconds",
    "player_current_gold",
    "player_total_gold",
    "player_level",
    "player_xp",
    "player_minions_killed",
    "player_jungle_minions_killed",
    "team_current_gold_diff",
    "team_total_gold_diff",
    "team_xp_diff",
    "team_cs_diff",
]


# ============================================================
# LOGGING
# ============================================================

def log(message=""):
    print(message)


# ============================================================
# LOAD
# ============================================================

def load_dataset():
    log("=" * 75)
    log("MEJAI MODEL DIAGNOSTICS")
    log("=" * 75)

    log("")
    log("[1] Loading research dataset...")

    if not DATA_PATH.exists():
        log(
            f"[ERROR] Dataset not found: "
            f"{DATA_PATH}"
        )
        return pd.DataFrame()

    df = pd.read_parquet(DATA_PATH)

    log(f"Rows:    {len(df):,}")
    log(f"Columns: {len(df.columns):,}")

    return df


# ============================================================
# PREPARE
# ============================================================

def prepare_data(df):
    log("")
    log("[2] Preparing completed Mejai decisions...")

    df = df[
        df["lifecycle_status"].isin(
            ["RETAINED", "SOLD"]
        )
    ].copy()

    log(
        f"Completed decisions: "
        f"{len(df):,}"
    )

    return df


# ============================================================
# FEATURE CHECK
# ============================================================

def check_features(df):
    log("")
    log("[3] Checking numeric predictors...")

    missing_columns = [
        feature
        for feature in NUMERIC_FEATURES
        if feature not in df.columns
    ]

    if missing_columns:
        log(
            "[ERROR] Missing columns:"
            f" {missing_columns}"
        )
        return False

    missing_values = df[
        NUMERIC_FEATURES
    ].isna().sum()

    if missing_values.any():
        log("")
        log("[WARNING] Missing values found:")

        log(
            missing_values[
                missing_values > 0
            ].to_string()
        )

    else:
        log(
            "[PASS] No missing values "
            "in numeric predictors."
        )

    return True


# ============================================================
# CORRELATION MATRIX
# ============================================================

def inspect_correlations(df):
    log("")
    log("=" * 75)
    log("[4] NUMERIC PREDICTOR CORRELATIONS")
    log("=" * 75)

    correlation = df[
        NUMERIC_FEATURES
    ].corr()

    log("")
    log(
        correlation.round(2).to_string()
    )

    return correlation


# ============================================================
# HIGH CORRELATION PAIRS
# ============================================================

def inspect_high_correlations(
    correlation,
    threshold=0.70,
):
    log("")
    log("=" * 75)
    log(
        f"[5] HIGH CORRELATION PAIRS "
        f"(absolute r >= {threshold:.2f})"
    )
    log("=" * 75)

    pairs = []

    for i in range(
        len(correlation.columns)
    ):
        for j in range(
            i + 1,
            len(correlation.columns),
        ):

            feature_a = correlation.columns[i]
            feature_b = correlation.columns[j]

            value = correlation.iloc[i, j]

            if abs(value) >= threshold:
                pairs.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "correlation": value,
                        "absolute_correlation": abs(
                            value
                        ),
                    }
                )

    if not pairs:
        log("")
        log(
            "[PASS] No predictor pairs "
            "exceeded the threshold."
        )
        return pd.DataFrame()

    pairs_df = (
        pd.DataFrame(pairs)
        .sort_values(
            "absolute_correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    log("")
    log(
        pairs_df[
            [
                "feature_a",
                "feature_b",
                "correlation",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    return pairs_df


# ============================================================
# GOLD DIFFERENCE FOCUSED ANALYSIS
# ============================================================

def inspect_gold_relationships(
    correlation,
):
    target = "team_total_gold_diff"

    log("")
    log("=" * 75)
    log(
        "[6] TEAM TOTAL GOLD DIFFERENCE "
        "RELATIONSHIPS"
    )
    log("=" * 75)

    if target not in correlation.columns:
        log(
            "[ERROR] team_total_gold_diff "
            "not found."
        )
        return

    relationships = (
        correlation[target]
        .drop(target)
        .sort_values(
            key=lambda values: values.abs(),
            ascending=False,
        )
    )

    log("")
    log(
        relationships.round(3).to_string()
    )

    log("")
    log(
        "Interpretation:"
    )

    log(
        "These correlations show which other "
        "numeric predictors overlap most "
        "strongly with team_total_gold_diff."
    )

    log(
        "High correlation does NOT prove that "
        "multicollinearity is harmful; VIF is "
        "checked next."
    )


# ============================================================
# VIF
# ============================================================

def calculate_vif(df):
    log("")
    log("=" * 75)
    log("[7] VARIANCE INFLATION FACTOR (VIF)")
    log("=" * 75)

    numeric_df = (
        df[NUMERIC_FEATURES]
        .astype(float)
        .copy()
    )

    # Standardise first so numerical scale does not
    # affect the matrix calculation.
    numeric_df = (
        numeric_df
        - numeric_df.mean()
    ) / numeric_df.std()

    # Add a constant for the VIF calculation.
    X = np.column_stack(
        [
            np.ones(len(numeric_df)),
            numeric_df.to_numpy(),
        ]
    )

    vif_rows = []

    for index, feature in enumerate(
        NUMERIC_FEATURES,
        start=1,
    ):
        vif = variance_inflation_factor(
            X,
            index,
        )

        vif_rows.append(
            {
                "feature": feature,
                "VIF": vif,
            }
        )

    vif_df = (
        pd.DataFrame(vif_rows)
        .sort_values(
            "VIF",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    log("")
    log(
        vif_df.round(3).to_string(
            index=False
        )
    )

    log("")
    log(
        "Rough interpretation:"
    )

    log(
        "VIF ~ 1       little redundancy"
    )

    log(
        "VIF 1-5       moderate redundancy"
    )

    log(
        "VIF 5-10      substantial redundancy"
    )

    log(
        "VIF > 10      strong multicollinearity"
    )

    return vif_df


# ============================================================
# CONDITION NUMBER
# ============================================================

def calculate_condition_number(df):
    log("")
    log("=" * 75)
    log("[8] CONDITION NUMBER")
    log("=" * 75)

    numeric_df = (
        df[NUMERIC_FEATURES]
        .astype(float)
        .copy()
    )

    numeric_df = (
        numeric_df
        - numeric_df.mean()
    ) / numeric_df.std()

    matrix = numeric_df.to_numpy()

    condition_number = np.linalg.cond(
        matrix
    )

    log("")
    log(
        f"Condition number: "
        f"{condition_number:,.2f}"
    )

    log("")
    log(
        "This is a broad diagnostic for "
        "linear dependence among predictors."
    )

    return condition_number


# ============================================================
# GOLD-ONLY VS RELATED VARIABLES
# ============================================================

def compare_gold_feature_group(
    df,
):
    log("")
    log("=" * 75)
    log(
        "[9] GOLD / XP / CS FEATURE GROUP"
    )
    log("=" * 75)

    features = [
        "team_total_gold_diff",
        "team_current_gold_diff",
        "team_xp_diff",
        "team_cs_diff",
        "player_total_gold",
        "player_current_gold",
        "player_xp",
        "player_level",
    ]

    available = [
        feature
        for feature in features
        if feature in df.columns
    ]

    correlation = (
        df[available]
        .corr()
        .round(2)
    )

    log("")
    log(
        correlation.to_string()
    )

    log("")
    log(
        "This smaller matrix focuses on "
        "variables that may represent "
        "overlapping game-state information."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    if df.empty:
        return

    df = prepare_data(df)

    if df.empty:
        log(
            "[ERROR] No completed decisions."
        )
        return

    if not check_features(df):
        return

    correlation = inspect_correlations(
        df
    )

    inspect_high_correlations(
        correlation,
        threshold=0.70,
    )

    inspect_gold_relationships(
        correlation
    )

    vif_df = calculate_vif(df)

    calculate_condition_number(
        df
    )

    compare_gold_feature_group(
        df
    )

    log("")
    log("=" * 75)
    log("DIAGNOSTIC SUMMARY")
    log("=" * 75)

    gold_vif = vif_df[
        vif_df["feature"]
        == "team_total_gold_diff"
    ]

    if not gold_vif.empty:
        value = gold_vif.iloc[0]["VIF"]

        log("")
        log(
            f"team_total_gold_diff VIF: "
            f"{value:.2f}"
        )

        if value > 10:
            log(
                "[WARNING] Strong multicollinearity "
                "involving team_total_gold_diff."
            )

        elif value > 5:
            log(
                "[WARNING] Substantial predictor "
                "redundancy involving "
                "team_total_gold_diff."
            )

        else:
            log(
                "[PASS] team_total_gold_diff "
                "does not show severe VIF."
            )

    log("")
    log(
        "No predictors were removed."
    )

    log(
        "Use these diagnostics to decide "
        "whether coefficient interpretation "
        "requires a reduced feature set."
    )

    log("")
    log("=" * 75)
    log("MODEL DIAGNOSTICS COMPLETE")
    log("=" * 75)


if __name__ == "__main__":
    main()