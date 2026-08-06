from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ENRICHED_DIR = Path("data/analysis/enriched_candidate_pool")

CASE_INPUT = ENRICHED_DIR / "mejai_case_candidates_enriched.parquet"
CONTROL_INPUT = ENRICHED_DIR / "mejai_control_candidates_enriched.parquet"
COMBINED_INPUT = ENRICHED_DIR / "mejai_candidate_pool_enriched.parquet"

SOURCE_CASE_INPUT = Path("data/analysis/mejai_research_dataset.parquet")
SOURCE_CONTROL_INPUT = Path("data/analysis/mejai_control_candidates.parquet")

OUTPUT_DIR = ENRICHED_DIR / "validation"
CASE_SUMMARY_OUTPUT = OUTPUT_DIR / "case_candidate_validation.csv"
CONTROL_SUMMARY_OUTPUT = OUTPUT_DIR / "control_candidate_validation.csv"
COMBINED_SUMMARY_OUTPUT = OUTPUT_DIR / "combined_candidate_validation.csv"

CASE_ERROR_OUTPUT = OUTPUT_DIR / "case_candidate_validation_errors.parquet"
CONTROL_ERROR_OUTPUT = OUTPUT_DIR / "control_candidate_validation_errors.parquet"
COMBINED_ERROR_OUTPUT = OUTPUT_DIR / "combined_candidate_validation_errors.parquet"

REPORT_OUTPUT = OUTPUT_DIR / "enriched_candidate_pool_validation_report.txt"

CORE_COMPACT_FEATURES = [
    "purchase_time_minutes",
    "dark_seal_purchased_before_observation",
    "kills_last_5m",
    "deaths_last_5m",
    "assists_last_5m",
    "seconds_since_last_death",
]

CORE_CARRY_FEATURES = [
    "player_gold_diff_vs_role_opponent",
    "player_xp_diff_vs_role_opponent",
    "rest_of_team_gold_diff",
    "rest_of_team_xp_diff",
]

IDENTITY_COLUMNS = [
    "match_id",
    "participant_id",
    "observation_timestamp",
    "observation_id",
]

GOLD_TOLERANCE = 1e-6
XP_TOLERANCE = 1e-6


# ============================================================
# HELPERS
# ============================================================

def log(message: str) -> None:
    print(message)


def safe_ratio(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else np.nan


def first_existing(columns, candidates):
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def canonical_observation_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["match_id"].astype(str)
        + "_"
        + pd.to_numeric(df["participant_id"], errors="coerce").astype("Int64").astype(str)
        + "_"
        + pd.to_numeric(df["observation_timestamp"], errors="coerce").astype("Int64").astype(str)
    )


def resolve_team_gold_diff(df: pd.DataFrame) -> str | None:
    return first_existing(
        df.columns,
        [
            "team_total_gold_diff",
            "team_gold_diff",
            "control_team_total_gold_diff",
            "mejai_team_total_gold_diff",
        ],
    )


def resolve_team_xp_diff(df: pd.DataFrame) -> str | None:
    return first_existing(
        df.columns,
        [
            "team_xp_diff",
            "control_team_xp_diff",
            "mejai_team_xp_diff",
        ],
    )


# ============================================================
# LOADING
# ============================================================

def load_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")

    return pd.read_parquet(path).copy()


def normalize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    numeric_columns = [
        "participant_id",
        "observation_timestamp",
        "treatment",
        "team_id",
        "opponent_participant_id",
        "player_snapshot_timestamp",
        "opponent_snapshot_timestamp",
        "player_total_gold",
        "opponent_total_gold",
        "player_xp",
        "opponent_xp",
        *CORE_COMPACT_FEATURES,
        *CORE_CARRY_FEATURES,
    ]

    for column in numeric_columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")

    return output


# ============================================================
# VALIDATION FLAGS
# ============================================================

def build_validation_flags(
    df: pd.DataFrame,
    expected_source: str | None,
) -> pd.DataFrame:
    validated = normalize_numeric_columns(df)

    required = [
        *IDENTITY_COLUMNS,
        "candidate_source",
        "treatment",
        *CORE_COMPACT_FEATURES,
        *CORE_CARRY_FEATURES,
        "player_total_gold",
        "opponent_total_gold",
        "player_xp",
        "opponent_xp",
    ]
    missing_columns = [column for column in required if column not in validated.columns]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required enriched columns: {missing_columns}"
        )

    validated["expected_observation_id"] = canonical_observation_id(validated)
    validated["error_observation_id_mismatch"] = (
        validated["observation_id"].astype(str)
        != validated["expected_observation_id"].astype(str)
    )

    validated["error_missing_identity"] = validated[
        ["match_id", "participant_id", "observation_timestamp", "observation_id"]
    ].isna().any(axis=1)

    validated["warning_reused_observation"] = validated.duplicated(
        subset=["observation_id"],
        keep=False,
    )

    validated["error_invalid_treatment"] = ~validated["treatment"].isin([0, 1])

    if expected_source is None:
        validated["error_wrong_candidate_source"] = ~validated[
            "candidate_source"
        ].isin(["CASE", "CONTROL"])
    else:
        validated["error_wrong_candidate_source"] = (
            validated["candidate_source"].astype(str).str.upper()
            != expected_source
        )

    if expected_source == "CASE":
        validated["error_wrong_treatment_for_source"] = (
            validated["treatment"] != 1
        )
    elif expected_source == "CONTROL":
        validated["error_wrong_treatment_for_source"] = (
            validated["treatment"] != 0
        )
    else:
        validated["error_wrong_treatment_for_source"] = (
            (
                validated["candidate_source"].astype(str).str.upper().eq("CASE")
                & validated["treatment"].ne(1)
            )
            |
            (
                validated["candidate_source"].astype(str).str.upper().eq("CONTROL")
                & validated["treatment"].ne(0)
            )
        )

    validated["error_missing_compact_feature"] = validated[
        CORE_COMPACT_FEATURES
    ].isna().all(axis=1)

    validated["error_missing_required_compact_feature"] = validated[
        [
            "purchase_time_minutes",
            "dark_seal_purchased_before_observation",
            "kills_last_5m",
            "deaths_last_5m",
            "assists_last_5m",
        ]
    ].isna().any(axis=1)

    validated["error_missing_carry_feature"] = validated[
        CORE_CARRY_FEATURES
    ].isna().any(axis=1)

    validated["error_invalid_dark_seal_flag"] = ~validated[
        "dark_seal_purchased_before_observation"
    ].isin([0, 1])

    validated["error_negative_recent_count"] = (
        validated[
            ["kills_last_5m", "deaths_last_5m", "assists_last_5m"]
        ] < 0
    ).any(axis=1)

    validated["error_negative_purchase_time"] = (
        validated["purchase_time_minutes"] < 0
    )

    validated["error_self_as_opponent"] = (
        validated["participant_id"]
        == validated["opponent_participant_id"]
    )

    validated["recomputed_player_gold_diff"] = (
        validated["player_total_gold"]
        - validated["opponent_total_gold"]
    )
    validated["recomputed_player_xp_diff"] = (
        validated["player_xp"]
        - validated["opponent_xp"]
    )

    validated["gold_player_identity_error"] = (
        validated["player_gold_diff_vs_role_opponent"]
        - validated["recomputed_player_gold_diff"]
    ).abs()

    validated["xp_player_identity_error"] = (
        validated["player_xp_diff_vs_role_opponent"]
        - validated["recomputed_player_xp_diff"]
    ).abs()

    validated["error_player_gold_identity"] = (
        validated["gold_player_identity_error"] > GOLD_TOLERANCE
    )
    validated["error_player_xp_identity"] = (
        validated["xp_player_identity_error"] > XP_TOLERANCE
    )

    team_gold_column = resolve_team_gold_diff(validated)
    team_xp_column = resolve_team_xp_diff(validated)

    if team_gold_column is None:
        validated["error_rest_gold_identity"] = True
        validated["gold_rest_identity_error"] = np.nan
    else:
        team_gold = pd.to_numeric(validated[team_gold_column], errors="coerce")
        recomputed = (
            team_gold
            - validated["player_gold_diff_vs_role_opponent"]
        )
        validated["gold_rest_identity_error"] = (
            validated["rest_of_team_gold_diff"] - recomputed
        ).abs()
        validated["error_rest_gold_identity"] = (
            validated["gold_rest_identity_error"] > GOLD_TOLERANCE
        ) | team_gold.isna()

    if team_xp_column is None:
        validated["error_rest_xp_identity"] = True
        validated["xp_rest_identity_error"] = np.nan
    else:
        team_xp = pd.to_numeric(validated[team_xp_column], errors="coerce")
        recomputed = (
            team_xp
            - validated["player_xp_diff_vs_role_opponent"]
        )
        validated["xp_rest_identity_error"] = (
            validated["rest_of_team_xp_diff"] - recomputed
        ).abs()
        validated["error_rest_xp_identity"] = (
            validated["xp_rest_identity_error"] > XP_TOLERANCE
        ) | team_xp.isna()

    if {
        "player_snapshot_timestamp",
        "opponent_snapshot_timestamp",
    }.issubset(validated.columns):
        validated["error_player_snapshot_after_observation"] = (
            validated["player_snapshot_timestamp"]
            > validated["observation_timestamp"]
        )
        validated["error_opponent_snapshot_after_observation"] = (
            validated["opponent_snapshot_timestamp"]
            > validated["observation_timestamp"]
        )
    else:
        validated["error_player_snapshot_after_observation"] = False
        validated["error_opponent_snapshot_after_observation"] = False

    error_columns = [
        column
        for column in validated.columns
        if column.startswith("error_")
    ]
    warning_columns = [
        column
        for column in validated.columns
        if column.startswith("warning_")
    ]

    validated["has_validation_error"] = validated[
        error_columns
    ].any(axis=1)
    validated["has_validation_warning"] = validated[
        warning_columns
    ].any(axis=1)

    return validated


# ============================================================
# SUMMARIES
# ============================================================

def summarize_dataset(
    validated: pd.DataFrame,
    sample_name: str,
) -> pd.DataFrame:
    metrics = {
        "rows": len(validated),
        "unique_observations": validated["observation_id"].nunique(dropna=True),
        "unique_matches": validated["match_id"].nunique(dropna=True),
        "identity_complete_ratio": safe_ratio(
            ~validated["error_missing_identity"]
        ),
        "observation_id_correct_ratio": safe_ratio(
            ~validated["error_observation_id_mismatch"]
        ),
        "reused_observation_ratio": safe_ratio(
            validated["warning_reused_observation"]
        ),
        "candidate_source_correct_ratio": safe_ratio(
            ~validated["error_wrong_candidate_source"]
        ),
        "treatment_correct_ratio": safe_ratio(
            ~validated["error_wrong_treatment_for_source"]
        ),
        "compact_required_complete_ratio": safe_ratio(
            ~validated["error_missing_required_compact_feature"]
        ),
        "carry_feature_complete_ratio": safe_ratio(
            ~validated["error_missing_carry_feature"]
        ),
        "dark_seal_flag_valid_ratio": safe_ratio(
            ~validated["error_invalid_dark_seal_flag"]
        ),
        "recent_counts_nonnegative_ratio": safe_ratio(
            ~validated["error_negative_recent_count"]
        ),
        "self_opponent_error_ratio": safe_ratio(
            validated["error_self_as_opponent"]
        ),
        "player_gold_identity_error_ratio": safe_ratio(
            validated["error_player_gold_identity"]
        ),
        "player_xp_identity_error_ratio": safe_ratio(
            validated["error_player_xp_identity"]
        ),
        "rest_gold_identity_error_ratio": safe_ratio(
            validated["error_rest_gold_identity"]
        ),
        "rest_xp_identity_error_ratio": safe_ratio(
            validated["error_rest_xp_identity"]
        ),
        "player_snapshot_after_observation_ratio": safe_ratio(
            validated["error_player_snapshot_after_observation"]
        ),
        "opponent_snapshot_after_observation_ratio": safe_ratio(
            validated["error_opponent_snapshot_after_observation"]
        ),
        "any_validation_error_ratio": safe_ratio(
            validated["has_validation_error"]
        ),
        "any_validation_warning_ratio": safe_ratio(
            validated["has_validation_warning"]
        ),
        "dark_seal_rate": float(
            validated["dark_seal_purchased_before_observation"].mean()
        ),
        "mean_kills_last_5m": float(validated["kills_last_5m"].mean()),
        "mean_deaths_last_5m": float(validated["deaths_last_5m"].mean()),
        "mean_player_gold_diff_vs_opponent": float(
            validated["player_gold_diff_vs_role_opponent"].mean()
        ),
        "mean_rest_of_team_gold_diff": float(
            validated["rest_of_team_gold_diff"].mean()
        ),
        "max_player_gold_identity_error": float(
            validated["gold_player_identity_error"].max()
        ),
        "max_player_xp_identity_error": float(
            validated["xp_player_identity_error"].max()
        ),
        "max_rest_gold_identity_error": float(
            validated["gold_rest_identity_error"].max()
        ),
        "max_rest_xp_identity_error": float(
            validated["xp_rest_identity_error"].max()
        ),
    }

    return pd.DataFrame(
        [
            {
                "sample": sample_name,
                "metric": metric,
                "value": value,
            }
            for metric, value in metrics.items()
        ]
    )


def compare_source_row_counts(
    cases: pd.DataFrame,
    controls: pd.DataFrame,
    combined: pd.DataFrame,
) -> pd.DataFrame:
    source_cases = load_required(SOURCE_CASE_INPUT, "source case dataset")
    source_controls = load_required(
        SOURCE_CONTROL_INPUT,
        "source control dataset",
    )

    rows = [
        {
            "metric": "source_case_rows",
            "value": len(source_cases),
        },
        {
            "metric": "enriched_case_rows",
            "value": len(cases),
        },
        {
            "metric": "case_row_difference",
            "value": len(cases) - len(source_cases),
        },
        {
            "metric": "source_control_rows",
            "value": len(source_controls),
        },
        {
            "metric": "enriched_control_rows",
            "value": len(controls),
        },
        {
            "metric": "control_row_difference",
            "value": len(controls) - len(source_controls),
        },
        {
            "metric": "expected_combined_rows",
            "value": len(cases) + len(controls),
        },
        {
            "metric": "actual_combined_rows",
            "value": len(combined),
        },
        {
            "metric": "combined_row_difference",
            "value": len(combined) - len(cases) - len(controls),
        },
    ]

    return pd.DataFrame(rows)


def validation_errors(validated: pd.DataFrame) -> pd.DataFrame:
    useful_columns = [
        column
        for column in [
            "candidate_source",
            "case_id",
            "match_id",
            "participant_id",
            "observation_timestamp",
            "observation_id",
            "treatment",
            "team_id",
            "team_position",
            "opponent_participant_id",
            *CORE_COMPACT_FEATURES,
            *CORE_CARRY_FEATURES,
        ]
        if column in validated.columns
    ]

    flag_columns = [
        column
        for column in validated.columns
        if column.startswith("error_")
        or column.startswith("warning_")
    ]

    return validated.loc[
        validated["has_validation_error"]
        | validated["has_validation_warning"],
        useful_columns + flag_columns,
    ].copy()


# ============================================================
# REPORTING
# ============================================================

def format_summary(summary: pd.DataFrame) -> str:
    display = summary[["metric", "value"]].copy()

    def format_value(row):
        metric = row["metric"]
        value = row["value"]

        if metric.endswith("_ratio") or metric.endswith("_rate"):
            return f"{float(value):.4%}"

        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"

        value = float(value)

        if value.is_integer():
            return f"{int(value):,}"

        return f"{value:.4f}"

    display["value"] = display.apply(format_value, axis=1)
    return display.to_string(index=False)


def build_report(
    case_summary: pd.DataFrame,
    control_summary: pd.DataFrame,
    combined_summary: pd.DataFrame,
    row_counts: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "=" * 88,
            "ENRICHED CANDIDATE POOL VALIDATION",
            "=" * 88,
            "",
            "Expected:",
            "  - case and control row differences should be 0",
            "  - combined row difference should be 0",
            "  - identity, source, treatment, and feature completeness should be 100%",
            "  - all carry-state arithmetic error ratios should be 0%",
            "  - snapshots after observation should be 0%",
            "  - repeated control observations are allowed and reported as reuse warnings",
            "    because one control may be linked to multiple case relationships",
            "",
            "ROW PRESERVATION",
            "-" * 88,
            format_summary(row_counts),
            "",
            "CASES",
            "-" * 88,
            format_summary(case_summary),
            "",
            "CONTROLS",
            "-" * 88,
            format_summary(control_summary),
            "",
            "COMBINED",
            "-" * 88,
            format_summary(combined_summary),
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    log("=" * 88)
    log("VALIDATE ENRICHED CANDIDATE POOL")
    log("=" * 88)

    cases = load_required(CASE_INPUT, "enriched case candidates")
    controls = load_required(
        CONTROL_INPUT,
        "enriched control candidates",
    )
    combined = load_required(
        COMBINED_INPUT,
        "combined enriched candidate pool",
    )

    log(f"Enriched case rows loaded: {len(cases):,}")
    log(f"Enriched control rows loaded: {len(controls):,}")
    log(f"Combined rows loaded: {len(combined):,}")

    case_validated = build_validation_flags(
        cases,
        expected_source="CASE",
    )
    control_validated = build_validation_flags(
        controls,
        expected_source="CONTROL",
    )
    combined_validated = build_validation_flags(
        combined,
        expected_source=None,
    )

    case_summary = summarize_dataset(
        case_validated,
        "cases",
    )
    control_summary = summarize_dataset(
        control_validated,
        "controls",
    )
    combined_summary = summarize_dataset(
        combined_validated,
        "combined",
    )

    row_counts = compare_source_row_counts(
        cases,
        controls,
        combined,
    )

    case_errors = validation_errors(case_validated)
    control_errors = validation_errors(control_validated)
    combined_errors = validation_errors(combined_validated)

    report = build_report(
        case_summary,
        control_summary,
        combined_summary,
        row_counts,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    case_summary.to_csv(
        CASE_SUMMARY_OUTPUT,
        index=False,
    )
    control_summary.to_csv(
        CONTROL_SUMMARY_OUTPUT,
        index=False,
    )
    combined_summary.to_csv(
        COMBINED_SUMMARY_OUTPUT,
        index=False,
    )

    case_errors.to_parquet(
        CASE_ERROR_OUTPUT,
        index=False,
    )
    control_errors.to_parquet(
        CONTROL_ERROR_OUTPUT,
        index=False,
    )
    combined_errors.to_parquet(
        COMBINED_ERROR_OUTPUT,
        index=False,
    )

    REPORT_OUTPUT.write_text(
        report,
        encoding="utf-8",
    )

    log("")
    log(report)
    log("")
    log(f"[SAVED] {CASE_SUMMARY_OUTPUT}")
    log(f"[SAVED] {CONTROL_SUMMARY_OUTPUT}")
    log(f"[SAVED] {COMBINED_SUMMARY_OUTPUT}")
    log(f"[SAVED] {CASE_ERROR_OUTPUT}")
    log(f"[SAVED] {CONTROL_ERROR_OUTPUT}")
    log(f"[SAVED] {COMBINED_ERROR_OUTPUT}")
    log(f"[SAVED] {REPORT_OUTPUT}")
    log("")

    row_count_errors = (
        row_counts.loc[
            row_counts["metric"].isin(
                [
                    "case_row_difference",
                    "control_row_difference",
                    "combined_row_difference",
                ]
            ),
            "value",
        ]
        != 0
    ).any()

    validation_error = any(
        float(frame["has_validation_error"].mean()) > 0
        for frame in [
            case_validated,
            control_validated,
            combined_validated,
        ]
    )

    validation_warning = any(
        float(frame["has_validation_warning"].mean()) > 0
        for frame in [
            case_validated,
            control_validated,
            combined_validated,
        ]
    )

    if row_count_errors or validation_error:
        log(
            "[WARNING] ENRICHED CANDIDATE POOL VALIDATION FOUND ERRORS. "
            "Inspect the saved parquet files before matching."
        )
    elif validation_warning:
        log(
            "[PASSED WITH EXPECTED REUSE WARNINGS] "
            "ENRICHED CANDIDATE POOL VALIDATION COMPLETE"
        )
    else:
        log(
            "[PASSED] ENRICHED CANDIDATE POOL VALIDATION COMPLETE"
        )


if __name__ == "__main__":
    main()