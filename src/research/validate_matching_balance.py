import numpy as np
import pandas as pd

from src.research.config import V2_MATCHING_DIR


OUTPUT_DIR = V2_MATCHING_DIR / "balance"

MATCHED_FILES = {
    "primary_variable_ratio": V2_MATCHING_DIR / "mejai_matched_primary.parquet",
    "relaxed_variable_ratio": V2_MATCHING_DIR / "mejai_matched_relaxed.parquet",
    "primary_1to1": V2_MATCHING_DIR / "mejai_matched_primary_1to1_robustness.parquet",
}

NUMERIC_COVARIATES = [
    "observation_time_minutes",
    "player_level",
    "player_total_gold",
    "player_current_gold",
    "player_xp",
    "player_minions_killed",
    "player_jungle_minions_killed",
    "player_gold_diff_vs_role_opponent",
    "player_xp_diff_vs_role_opponent",
    "rest_of_team_gold_diff",
    "rest_of_team_xp_diff",
    "team_total_gold_diff",
    "team_xp_diff",
    "team_cs_diff",
    "kills_last_5m",
    "deaths_last_5m",
    "assists_last_5m",
    "dark_seal_purchased_before_observation",
]

CATEGORICAL_COVARIATES = [
    "region",
    "team_position",
    "champion_name",
]

GOOD_SMD_THRESHOLD = 0.10
WARNING_SMD_THRESHOLD = 0.20


def log(message=""):
    print(message)


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[valid]
    weights = weights[valid]

    if len(values) == 0 or weights.sum() <= 0:
        return np.nan

    return float(np.average(values, weights=weights))


def weighted_variance(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[valid]
    weights = weights[valid]

    if len(values) == 0 or weights.sum() <= 0:
        return np.nan

    mean = np.average(values, weights=weights)
    return float(np.average((values - mean) ** 2, weights=weights))


def effective_sample_size(weights):
    weights = np.asarray(weights, dtype=float)
    weights = weights[np.isfinite(weights) & (weights > 0)]

    if len(weights) == 0:
        return np.nan

    return float((weights.sum() ** 2) / np.square(weights).sum())


def standardised_mean_difference(
    case_mean,
    control_mean,
    case_variance,
    control_variance,
):
    pooled_variance = np.nanmean([case_variance, control_variance])

    if not np.isfinite(pooled_variance):
        return np.nan

    if pooled_variance <= 0:
        return 0.0 if np.isclose(case_mean, control_mean) else np.inf

    return float((case_mean - control_mean) / np.sqrt(pooled_variance))


def classify_smd(abs_smd):
    if not np.isfinite(abs_smd):
        return "UNAVAILABLE"

    if abs_smd < GOOD_SMD_THRESHOLD:
        return "GOOD"

    if abs_smd < WARNING_SMD_THRESHOLD:
        return "REVIEW"

    return "POOR"


def load_matched_dataset(sample_name, filepath):
    if not filepath.exists():
        raise FileNotFoundError(
            f"{sample_name} matched dataset not found: {filepath}"
        )

    df = pd.read_parquet(filepath, engine="pyarrow").copy()

    required_columns = [
        "matched_set_id",
        "treatment",
        "matching_weight",
        "outcome_win",
        "observation_timestamp",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{sample_name} is missing required columns: {missing}"
        )

    df["matched_set_id"] = df["matched_set_id"].astype(str)
    df["treatment"] = pd.to_numeric(df["treatment"], errors="coerce")
    df["matching_weight"] = pd.to_numeric(
        df["matching_weight"],
        errors="coerce",
    )
    df["observation_timestamp"] = pd.to_numeric(
        df["observation_timestamp"],
        errors="coerce",
    )

    if df[
        [
            "treatment",
            "matching_weight",
            "observation_timestamp",
        ]
    ].isna().any(axis=None):
        raise ValueError(
            f"{sample_name} contains invalid treatment, weight, "
            "or observation timestamp values"
        )

    df["treatment"] = df["treatment"].astype(int)
    df["observation_time_minutes"] = (
        df["observation_timestamp"] / 60_000
    )

    validate_matched_structure(df, sample_name)
    return df


def validate_matched_structure(df, sample_name):
    if not df["treatment"].isin([0, 1]).all():
        raise ValueError(
            f"{sample_name}: treatment must contain only 0 and 1"
        )

    if (df["matching_weight"] <= 0).any():
        raise ValueError(
            f"{sample_name}: matching weights must be positive"
        )

    set_summary = (
        df.groupby("matched_set_id")
        .agg(
            case_rows=("treatment", "sum"),
            total_rows=("treatment", "size"),
        )
    )

    if not set_summary["case_rows"].eq(1).all():
        raise ValueError(
            f"{sample_name}: every set must contain exactly one case"
        )

    if not set_summary["total_rows"].between(2, 4).all():
        raise ValueError(
            f"{sample_name}: every set must contain one to three controls"
        )

    case_weights = (
        df[df["treatment"] == 1]
        .groupby("matched_set_id")["matching_weight"]
        .sum()
    )

    control_weights = (
        df[df["treatment"] == 0]
        .groupby("matched_set_id")["matching_weight"]
        .sum()
    )

    if not np.isclose(case_weights, 1.0).all():
        raise ValueError(
            f"{sample_name}: case weights do not sum to one per set"
        )

    if not np.isclose(control_weights, 1.0).all():
        raise ValueError(
            f"{sample_name}: control weights do not sum to one per set"
        )


def paired_set_differences(df, column):
    working = df[
        [
            "matched_set_id",
            "treatment",
            "matching_weight",
            column,
        ]
    ].copy()

    working[column] = pd.to_numeric(
        working[column],
        errors="coerce",
    )

    case_values = (
        working[working["treatment"] == 1]
        .set_index("matched_set_id")[column]
    )

    controls = working[working["treatment"] == 0].dropna(subset=[column])

    if controls.empty:
        return pd.Series(dtype=float)

    controls["weighted_value"] = (
        controls[column] * controls["matching_weight"]
    )

    control_means = (
        controls.groupby("matched_set_id")
        .agg(
            weighted_sum=("weighted_value", "sum"),
            weight_sum=("matching_weight", "sum"),
        )
    )

    control_means["control_mean"] = (
        control_means["weighted_sum"]
        / control_means["weight_sum"]
    )

    paired = pd.concat(
        [
            case_values.rename("case_value"),
            control_means["control_mean"],
        ],
        axis=1,
        join="inner",
    ).dropna()

    return paired["case_value"] - paired["control_mean"]


def calculate_numeric_balance(df, sample_name):
    rows = []

    case_rows = df[df["treatment"] == 1]
    control_rows = df[df["treatment"] == 0]

    available_columns = [
        column
        for column in NUMERIC_COVARIATES
        if column in df.columns
    ]

    for column in available_columns:
        case_values = pd.to_numeric(
            case_rows[column],
            errors="coerce",
        )
        control_values = pd.to_numeric(
            control_rows[column],
            errors="coerce",
        )

        case_valid = case_values.notna()
        control_valid = control_values.notna()

        case_mean = weighted_mean(
            case_values[case_valid],
            case_rows.loc[case_valid, "matching_weight"],
        )
        control_mean = weighted_mean(
            control_values[control_valid],
            control_rows.loc[control_valid, "matching_weight"],
        )

        case_variance = weighted_variance(
            case_values[case_valid],
            case_rows.loc[case_valid, "matching_weight"],
        )
        control_variance = weighted_variance(
            control_values[control_valid],
            control_rows.loc[control_valid, "matching_weight"],
        )

        smd = standardised_mean_difference(
            case_mean,
            control_mean,
            case_variance,
            control_variance,
        )

        paired_differences = paired_set_differences(df, column)

        rows.append(
            {
                "sample": sample_name,
                "covariate": column,
                "case_mean": case_mean,
                "control_mean": control_mean,
                "mean_difference": case_mean - control_mean,
                "case_standard_deviation": np.sqrt(case_variance),
                "control_standard_deviation": np.sqrt(control_variance),
                "standardised_mean_difference": smd,
                "absolute_smd": abs(smd),
                "balance_status": classify_smd(abs(smd)),
                "mean_paired_difference": (
                    float(paired_differences.mean())
                    if not paired_differences.empty
                    else np.nan
                ),
                "median_absolute_paired_difference": (
                    float(paired_differences.abs().median())
                    if not paired_differences.empty
                    else np.nan
                ),
                "maximum_absolute_paired_difference": (
                    float(paired_differences.abs().max())
                    if not paired_differences.empty
                    else np.nan
                ),
                "case_non_missing": int(case_valid.sum()),
                "control_non_missing": int(control_valid.sum()),
            }
        )

    if not rows:
        raise ValueError(
            f"{sample_name}: no expected numeric covariates were found"
        )

    return pd.DataFrame(rows).sort_values(
        ["absolute_smd", "covariate"],
        ascending=[False, True],
    ).reset_index(drop=True)


def weighted_category_proportions(df, column, treatment):
    sample = df[df["treatment"] == treatment][
        [column, "matching_weight"]
    ].copy()

    sample[column] = sample[column].fillna("MISSING").astype(str)
    total_weight = sample["matching_weight"].sum()

    if total_weight <= 0:
        return pd.Series(dtype=float)

    return (
        sample.groupby(column)["matching_weight"]
        .sum()
        .div(total_weight)
    )


def calculate_categorical_balance(df, sample_name):
    summary_rows = []
    detail_rows = []

    for column in CATEGORICAL_COVARIATES:
        if column not in df.columns:
            continue

        case_proportions = weighted_category_proportions(
            df,
            column,
            treatment=1,
        )
        control_proportions = weighted_category_proportions(
            df,
            column,
            treatment=0,
        )

        categories = sorted(
            set(case_proportions.index)
            | set(control_proportions.index)
        )

        differences = []

        for category in categories:
            case_value = float(case_proportions.get(category, 0.0))
            control_value = float(control_proportions.get(category, 0.0))
            difference = case_value - control_value
            differences.append(abs(difference))

            detail_rows.append(
                {
                    "sample": sample_name,
                    "covariate": column,
                    "category": category,
                    "case_proportion": case_value,
                    "control_proportion": control_value,
                    "proportion_difference": difference,
                    "absolute_proportion_difference": abs(difference),
                }
            )

        total_variation_distance = 0.5 * sum(differences)
        maximum_gap = max(differences) if differences else np.nan

        summary_rows.append(
            {
                "sample": sample_name,
                "covariate": column,
                "categories": len(categories),
                "total_variation_distance": total_variation_distance,
                "maximum_absolute_category_gap": maximum_gap,
            }
        )

    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(detail_rows)

    if not details.empty:
        details = details.sort_values(
            [
                "sample",
                "covariate",
                "absolute_proportion_difference",
            ],
            ascending=[True, True, False],
        ).reset_index(drop=True)

    return summary, details


def calculate_set_consistency(df, sample_name):
    rows = []

    fields = [
        "region",
        "team_position",
        "dark_seal_purchased_before_observation",
    ]

    for field in fields:
        if field not in df.columns:
            continue

        unique_values = (
            df.groupby("matched_set_id")[field]
            .nunique(dropna=False)
        )

        mismatched_sets = int((unique_values > 1).sum())

        rows.append(
            {
                "sample": sample_name,
                "field": field,
                "matched_sets": int(unique_values.size),
                "mismatched_sets": mismatched_sets,
                "mismatch_rate": (
                    mismatched_sets / unique_values.size
                    if unique_values.size
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def build_sample_summary(
    df,
    sample_name,
    numeric_balance,
    categorical_summary,
    consistency,
):
    case_rows = df[df["treatment"] == 1]
    control_rows = df[df["treatment"] == 0]

    smd_values = numeric_balance["absolute_smd"].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    poor_covariates = numeric_balance[
        numeric_balance["absolute_smd"] >= WARNING_SMD_THRESHOLD
    ]

    review_covariates = numeric_balance[
        (numeric_balance["absolute_smd"] >= GOOD_SMD_THRESHOLD)
        & (numeric_balance["absolute_smd"] < WARNING_SMD_THRESHOLD)
    ]

    champion_tvd = np.nan

    if not categorical_summary.empty:
        champion_rows = categorical_summary[
            categorical_summary["covariate"] == "champion_name"
        ]

        if not champion_rows.empty:
            champion_tvd = float(
                champion_rows.iloc[0]["total_variation_distance"]
            )

    exact_field_mismatches = (
        int(consistency["mismatched_sets"].sum())
        if not consistency.empty
        else 0
    )

    return {
        "sample": sample_name,
        "matched_sets": int(df["matched_set_id"].nunique()),
        "case_rows": int(len(case_rows)),
        "control_rows": int(len(control_rows)),
        "case_effective_sample_size": effective_sample_size(
            case_rows["matching_weight"]
        ),
        "control_effective_sample_size": effective_sample_size(
            control_rows["matching_weight"]
        ),
        "maximum_absolute_smd": float(smd_values.max()),
        "median_absolute_smd": float(smd_values.median()),
        "covariates_good_below_0_10": int(
            (numeric_balance["absolute_smd"] < GOOD_SMD_THRESHOLD).sum()
        ),
        "covariates_review_0_10_to_0_20": int(len(review_covariates)),
        "covariates_poor_0_20_or_more": int(len(poor_covariates)),
        "champion_total_variation_distance": champion_tvd,
        "exact_field_mismatched_sets_total": exact_field_mismatches,
    }


def print_sample_results(summary, numeric_balance, consistency):
    log("")
    log("=" * 72)
    log(summary["sample"].upper())
    log("=" * 72)
    log(f"Matched sets: {summary['matched_sets']:,}")
    log(f"Control rows: {summary['control_rows']:,}")
    log(
        "Control effective sample size: "
        f"{summary['control_effective_sample_size']:,.1f}"
    )
    log(
        "Maximum absolute SMD: "
        f"{summary['maximum_absolute_smd']:.4f}"
    )
    log(
        "Median absolute SMD: "
        f"{summary['median_absolute_smd']:.4f}"
    )
    log(
        "Covariates below 0.10: "
        f"{summary['covariates_good_below_0_10']:,}"
    )
    log(
        "Covariates from 0.10 to below 0.20: "
        f"{summary['covariates_review_0_10_to_0_20']:,}"
    )
    log(
        "Covariates at or above 0.20: "
        f"{summary['covariates_poor_0_20_or_more']:,}"
    )

    log("")
    log("Largest numeric imbalances:")
    log(
        numeric_balance[
            [
                "covariate",
                "case_mean",
                "control_mean",
                "standardised_mean_difference",
                "balance_status",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    if not consistency.empty:
        log("")
        log("Within-set exact-field checks:")
        log(
            consistency[
                [
                    "field",
                    "mismatched_sets",
                    "mismatch_rate",
                ]
            ].to_string(index=False)
        )


def write_text_report(summaries, numeric_results, consistency_results):
    lines = [
        "MEJAI MATCHING BALANCE REPORT",
        "=" * 72,
        "",
        "Interpretation guide:",
        "  |SMD| < 0.10: good balance",
        "  0.10 <= |SMD| < 0.20: review",
        "  |SMD| >= 0.20: poor balance",
        "",
    ]

    for summary in summaries:
        sample = summary["sample"]
        sample_numeric = numeric_results[
            numeric_results["sample"] == sample
        ]
        sample_consistency = consistency_results[
            consistency_results["sample"] == sample
        ]

        lines.extend(
            [
                sample.upper(),
                "-" * 72,
                f"Matched sets: {summary['matched_sets']:,}",
                f"Control rows: {summary['control_rows']:,}",
                (
                    "Control effective sample size: "
                    f"{summary['control_effective_sample_size']:,.1f}"
                ),
                (
                    "Maximum absolute SMD: "
                    f"{summary['maximum_absolute_smd']:.4f}"
                ),
                (
                    "Median absolute SMD: "
                    f"{summary['median_absolute_smd']:.4f}"
                ),
                (
                    "Covariates below 0.10: "
                    f"{summary['covariates_good_below_0_10']:,}"
                ),
                (
                    "Covariates from 0.10 to below 0.20: "
                    f"{summary['covariates_review_0_10_to_0_20']:,}"
                ),
                (
                    "Covariates at or above 0.20: "
                    f"{summary['covariates_poor_0_20_or_more']:,}"
                ),
                "",
                "Largest numeric imbalances:",
                sample_numeric[
                    [
                        "covariate",
                        "case_mean",
                        "control_mean",
                        "standardised_mean_difference",
                        "balance_status",
                    ]
                ]
                .head(10)
                .to_string(index=False),
                "",
            ]
        )

        if not sample_consistency.empty:
            lines.extend(
                [
                    "Within-set exact-field checks:",
                    sample_consistency[
                        [
                            "field",
                            "mismatched_sets",
                            "mismatch_rate",
                        ]
                    ].to_string(index=False),
                    "",
                ]
            )

    report_file = OUTPUT_DIR / "matching_balance_report.txt"
    report_file.write_text("\n".join(lines), encoding="utf-8")
    return report_file


def main():
    log("=" * 72)
    log("VALIDATE VERSION 2 MATCHING BALANCE")
    log("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_numeric = []
    all_categorical_summary = []
    all_categorical_details = []
    all_consistency = []
    summaries = []

    for sample_name, filepath in MATCHED_FILES.items():
        df = load_matched_dataset(sample_name, filepath)

        numeric_balance = calculate_numeric_balance(
            df,
            sample_name,
        )

        categorical_summary, categorical_details = (
            calculate_categorical_balance(
                df,
                sample_name,
            )
        )

        consistency = calculate_set_consistency(
            df,
            sample_name,
        )

        summary = build_sample_summary(
            df,
            sample_name,
            numeric_balance,
            categorical_summary,
            consistency,
        )

        print_sample_results(
            summary,
            numeric_balance,
            consistency,
        )

        numeric_balance.to_csv(
            OUTPUT_DIR / f"{sample_name}_numeric_balance.csv",
            index=False,
        )

        categorical_summary.to_csv(
            OUTPUT_DIR / f"{sample_name}_categorical_balance_summary.csv",
            index=False,
        )

        categorical_details.to_csv(
            OUTPUT_DIR / f"{sample_name}_categorical_balance_details.csv",
            index=False,
        )

        consistency.to_csv(
            OUTPUT_DIR / f"{sample_name}_set_consistency.csv",
            index=False,
        )

        all_numeric.append(numeric_balance)
        all_categorical_summary.append(categorical_summary)
        all_categorical_details.append(categorical_details)
        all_consistency.append(consistency)
        summaries.append(summary)

    numeric_results = pd.concat(all_numeric, ignore_index=True)
    categorical_summary_results = pd.concat(
        all_categorical_summary,
        ignore_index=True,
    )
    categorical_detail_results = pd.concat(
        all_categorical_details,
        ignore_index=True,
    )
    consistency_results = pd.concat(
        all_consistency,
        ignore_index=True,
    )
    summary_results = pd.DataFrame(summaries)

    numeric_results.to_csv(
        OUTPUT_DIR / "all_numeric_balance.csv",
        index=False,
    )
    categorical_summary_results.to_csv(
        OUTPUT_DIR / "all_categorical_balance_summary.csv",
        index=False,
    )
    categorical_detail_results.to_csv(
        OUTPUT_DIR / "all_categorical_balance_details.csv",
        index=False,
    )
    consistency_results.to_csv(
        OUTPUT_DIR / "all_set_consistency.csv",
        index=False,
    )
    summary_results.to_csv(
        OUTPUT_DIR / "balance_summary.csv",
        index=False,
    )

    report_file = write_text_report(
        summaries,
        numeric_results,
        consistency_results,
    )

    log("")
    log(f"[SAVED] {OUTPUT_DIR}")
    log(f"[SAVED] {report_file}")
    log("")
    log("[PASSED] MATCHING BALANCE VALIDATION COMPLETED")


if __name__ == "__main__":
    main()
