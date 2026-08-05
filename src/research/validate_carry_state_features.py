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

OUTPUT_DIR = Path("data/analysis/carry_state_validation")
PRIMARY_SUMMARY_OUTPUT = OUTPUT_DIR / "primary_carry_state_validation.csv"
SENSITIVITY_SUMMARY_OUTPUT = OUTPUT_DIR / "sensitivity_carry_state_validation.csv"
PRIMARY_ERRORS_OUTPUT = OUTPUT_DIR / "primary_carry_state_validation_errors.parquet"
SENSITIVITY_ERRORS_OUTPUT = (
    OUTPUT_DIR / "sensitivity_carry_state_validation_errors.parquet"
)
REPORT_OUTPUT = OUTPUT_DIR / "carry_state_validation_report.txt"

CORE_FEATURES = [
    "player_gold_diff_vs_role_opponent",
    "player_xp_diff_vs_role_opponent",
    "rest_of_team_gold_diff",
    "rest_of_team_xp_diff",
]

REQUIRED_COLUMNS = [
    "match_id",
    "participant_id",
    "observation_timestamp",
    "team_id",
    "team_position",
    "opponent_participant_id",
    "player_snapshot_timestamp",
    "opponent_snapshot_timestamp",
    "player_total_gold",
    "opponent_total_gold",
    "player_xp",
    "opponent_xp",
    "team_total_gold_diff",
    "team_xp_diff",
    *CORE_FEATURES,
]

GOLD_TOLERANCE = 1e-6
XP_TOLERANCE = 1e-6
MAX_SNAPSHOT_AGE_MS = 120_000
MAX_PLAYER_OPPONENT_SNAPSHOT_GAP_MS = 60_000


# ============================================================
# HELPERS
# ============================================================

def log(message: str) -> None:
    print(message)


def first_existing(columns, candidates):
    available = set(columns)
    return next(
        (candidate for candidate in candidates if candidate in available),
        None,
    )


def resolve_team_gold_diff_column(df: pd.DataFrame) -> str:
    column = first_existing(
        df.columns,
        [
            "team_total_gold_diff",
            "team_gold_diff",
            "teamGoldDiff",
        ],
    )
    if column is None:
        raise ValueError("Could not find a team gold-difference column")
    return column


def resolve_team_xp_diff_column(df: pd.DataFrame) -> str:
    column = first_existing(
        df.columns,
        [
            "team_xp_diff",
            "teamXpDiff",
        ],
    )
    if column is None:
        raise ValueError("Could not find a team XP-difference column")
    return column


def safe_ratio(mask: pd.Series) -> float:
    return float(mask.mean()) if len(mask) else np.nan


# ============================================================
# LOADING
# ============================================================

def coalesce_column(
    df: pd.DataFrame,
    target: str,
    candidates: list[str],
) -> pd.DataFrame:
    existing = [column for column in candidates if column in df.columns]
    if not existing:
        return df

    if target not in df.columns:
        df[target] = df[existing[0]]
        existing = existing[1:]

    for column in existing:
        if column == target:
            continue
        df[target] = df[target].combine_first(df[column])

    return df


def normalize_merged_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Carry-state enrichment may create pandas merge suffixes when the original
    matched dataset already contains the same field. Coalesce those variants
    into one canonical validation column.
    """
    alias_groups = {
        "team_id": ["team_id", "team_id_x", "team_id_y"],
        "team_position": [
            "team_position",
            "team_position_x",
            "team_position_y",
            "position",
            "position_x",
            "position_y",
        ],
        "player_total_gold": [
            "player_total_gold",
            "player_total_gold_x",
            "player_total_gold_y",
        ],
        "player_xp": [
            "player_xp",
            "player_xp_x",
            "player_xp_y",
        ],
        "team_total_gold_diff": [
            "team_total_gold_diff",
            "team_total_gold_diff_x",
            "team_total_gold_diff_y",
            "team_gold_diff",
            "team_gold_diff_x",
            "team_gold_diff_y",
            "teamGoldDiff",
        ],
        "team_xp_diff": [
            "team_xp_diff",
            "team_xp_diff_x",
            "team_xp_diff_y",
            "teamXpDiff",
        ],
    }

    normalized = df.copy()
    for target, candidates in alias_groups.items():
        normalized = coalesce_column(
            normalized,
            target,
            candidates,
        )

    return normalized


def load_dataset(path: Path, sample_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{sample_name} carry-state dataset not found: {path}"
        )

    df = pd.read_parquet(path).copy()
    df = normalize_merged_columns(df)

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]
    if missing:
        similar = {
            column: [
                candidate
                for candidate in df.columns
                if column.split("_")[0] in candidate
            ][:10]
            for column in missing
        }
        raise ValueError(
            f"{sample_name} is missing required columns: {missing}\n"
            f"Potential related columns: {similar}"
        )

    numeric_columns = [
        "participant_id",
        "observation_timestamp",
        "team_id",
        "opponent_participant_id",
        "player_snapshot_timestamp",
        "opponent_snapshot_timestamp",
        "player_total_gold",
        "opponent_total_gold",
        "player_xp",
        "opponent_xp",
        "team_total_gold_diff",
        "team_xp_diff",
        *CORE_FEATURES,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.reset_index(drop=True)


# ============================================================
# VALIDATION
# ============================================================

def build_validation_columns(df: pd.DataFrame) -> pd.DataFrame:
    validated = df.copy()

    validated["recomputed_player_gold_diff"] = (
        validated["player_total_gold"]
        - validated["opponent_total_gold"]
    )
    validated["recomputed_player_xp_diff"] = (
        validated["player_xp"]
        - validated["opponent_xp"]
    )
    validated["recomputed_rest_of_team_gold_diff"] = (
        validated["team_total_gold_diff"]
        - validated["player_gold_diff_vs_role_opponent"]
    )
    validated["recomputed_rest_of_team_xp_diff"] = (
        validated["team_xp_diff"]
        - validated["player_xp_diff_vs_role_opponent"]
    )

    validated["gold_player_identity_error"] = (
        validated["player_gold_diff_vs_role_opponent"]
        - validated["recomputed_player_gold_diff"]
    ).abs()
    validated["xp_player_identity_error"] = (
        validated["player_xp_diff_vs_role_opponent"]
        - validated["recomputed_player_xp_diff"]
    ).abs()
    validated["gold_rest_identity_error"] = (
        validated["rest_of_team_gold_diff"]
        - validated["recomputed_rest_of_team_gold_diff"]
    ).abs()
    validated["xp_rest_identity_error"] = (
        validated["rest_of_team_xp_diff"]
        - validated["recomputed_rest_of_team_xp_diff"]
    ).abs()

    validated["player_snapshot_age_ms"] = (
        validated["observation_timestamp"]
        - validated["player_snapshot_timestamp"]
    )
    validated["opponent_snapshot_age_ms"] = (
        validated["observation_timestamp"]
        - validated["opponent_snapshot_timestamp"]
    )
    validated["player_opponent_snapshot_gap_ms"] = (
        validated["player_snapshot_timestamp"]
        - validated["opponent_snapshot_timestamp"]
    ).abs()

    validated["error_missing_core_feature"] = validated[
        CORE_FEATURES
    ].isna().any(axis=1)

    validated["error_missing_role_opponent"] = (
        validated["opponent_participant_id"].isna()
    )

    validated["error_self_as_opponent"] = (
        validated["participant_id"]
        == validated["opponent_participant_id"]
    )

    validated["error_player_snapshot_after_observation"] = (
        validated["player_snapshot_age_ms"] < 0
    )
    validated["error_opponent_snapshot_after_observation"] = (
        validated["opponent_snapshot_age_ms"] < 0
    )

    validated["warning_player_snapshot_old"] = (
        validated["player_snapshot_age_ms"] > MAX_SNAPSHOT_AGE_MS
    )
    validated["warning_opponent_snapshot_old"] = (
        validated["opponent_snapshot_age_ms"] > MAX_SNAPSHOT_AGE_MS
    )
    validated["warning_snapshot_pair_far_apart"] = (
        validated["player_opponent_snapshot_gap_ms"]
        > MAX_PLAYER_OPPONENT_SNAPSHOT_GAP_MS
    )

    validated["error_player_gold_identity"] = (
        validated["gold_player_identity_error"] > GOLD_TOLERANCE
    )
    validated["error_player_xp_identity"] = (
        validated["xp_player_identity_error"] > XP_TOLERANCE
    )
    validated["error_rest_gold_identity"] = (
        validated["gold_rest_identity_error"] > GOLD_TOLERANCE
    )
    validated["error_rest_xp_identity"] = (
        validated["xp_rest_identity_error"] > XP_TOLERANCE
    )

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


def summarize_validation(
    validated: pd.DataFrame,
    sample_name: str,
) -> pd.DataFrame:
    checks = {
        "rows": len(validated),
        "unique_matches": validated["match_id"].nunique(),
        "core_feature_complete_ratio": safe_ratio(
            ~validated["error_missing_core_feature"]
        ),
        "role_opponent_complete_ratio": safe_ratio(
            ~validated["error_missing_role_opponent"]
        ),
        "self_opponent_error_ratio": safe_ratio(
            validated["error_self_as_opponent"]
        ),
        "player_snapshot_after_observation_ratio": safe_ratio(
            validated["error_player_snapshot_after_observation"]
        ),
        "opponent_snapshot_after_observation_ratio": safe_ratio(
            validated["error_opponent_snapshot_after_observation"]
        ),
        "player_snapshot_old_ratio": safe_ratio(
            validated["warning_player_snapshot_old"]
        ),
        "opponent_snapshot_old_ratio": safe_ratio(
            validated["warning_opponent_snapshot_old"]
        ),
        "snapshot_pair_far_apart_ratio": safe_ratio(
            validated["warning_snapshot_pair_far_apart"]
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
        "any_validation_error_ratio": safe_ratio(
            validated["has_validation_error"]
        ),
        "any_validation_warning_ratio": safe_ratio(
            validated["has_validation_warning"]
        ),
        "median_player_snapshot_age_seconds": float(
            validated["player_snapshot_age_ms"].median() / 1000.0
        ),
        "median_opponent_snapshot_age_seconds": float(
            validated["opponent_snapshot_age_ms"].median() / 1000.0
        ),
        "median_snapshot_pair_gap_seconds": float(
            validated["player_opponent_snapshot_gap_ms"].median() / 1000.0
        ),
        "max_gold_player_identity_error": float(
            validated["gold_player_identity_error"].max()
        ),
        "max_xp_player_identity_error": float(
            validated["xp_player_identity_error"].max()
        ),
        "max_gold_rest_identity_error": float(
            validated["gold_rest_identity_error"].max()
        ),
        "max_xp_rest_identity_error": float(
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
            for metric, value in checks.items()
        ]
    )


def error_rows(validated: pd.DataFrame) -> pd.DataFrame:
    useful_columns = [
        "match_id",
        "participant_id",
        "observation_timestamp",
        "team_id",
        "team_position",
        "opponent_participant_id",
        "player_snapshot_timestamp",
        "opponent_snapshot_timestamp",
        "player_snapshot_age_ms",
        "opponent_snapshot_age_ms",
        "player_opponent_snapshot_gap_ms",
        "player_total_gold",
        "opponent_total_gold",
        "player_xp",
        "opponent_xp",
        "team_total_gold_diff",
        "team_xp_diff",
        *CORE_FEATURES,
    ]

    flag_columns = [
        column
        for column in validated.columns
        if column.startswith("error_")
        or column.startswith("warning_")
    ]

    mask = (
        validated["has_validation_error"]
        | validated["has_validation_warning"]
    )

    return validated.loc[
        mask,
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

        if metric.endswith("_ratio"):
            return f"{value:.4%}"
        if "seconds" in metric:
            return f"{value:.3f}"
        if float(value).is_integer():
            return f"{int(value):,}"
        return f"{value:.6f}"

    display["value"] = display.apply(format_value, axis=1)
    return display.to_string(index=False)


def build_report(
    primary_summary: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "=" * 80,
            "CARRY-STATE FEATURE VALIDATION",
            "=" * 80,
            "",
            "Expected:",
            "  - arithmetic identity error ratios should be 0%",
            "  - snapshots after observation should be 0%",
            "  - self-opponent errors should be 0%",
            "  - core feature and opponent coverage should be close to 100%",
            "  - snapshot-age warnings are diagnostic, not automatic failures",
            "",
            "PRIMARY",
            "-" * 80,
            format_summary(primary_summary),
            "",
            "SENSITIVITY",
            "-" * 80,
            format_summary(sensitivity_summary),
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    log("=" * 80)
    log("VALIDATE CARRY-STATE FEATURES")
    log("=" * 80)

    primary = load_dataset(PRIMARY_INPUT, "primary")
    sensitivity = load_dataset(SENSITIVITY_INPUT, "sensitivity")

    log(f"Primary rows loaded: {len(primary):,}")
    log(f"Sensitivity rows loaded: {len(sensitivity):,}")

    primary_validated = build_validation_columns(primary)
    sensitivity_validated = build_validation_columns(sensitivity)

    primary_summary = summarize_validation(
        primary_validated,
        "primary",
    )
    sensitivity_summary = summarize_validation(
        sensitivity_validated,
        "sensitivity",
    )

    primary_errors = error_rows(primary_validated)
    sensitivity_errors = error_rows(sensitivity_validated)

    report = build_report(
        primary_summary,
        sensitivity_summary,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    primary_summary.to_csv(
        PRIMARY_SUMMARY_OUTPUT,
        index=False,
    )
    sensitivity_summary.to_csv(
        SENSITIVITY_SUMMARY_OUTPUT,
        index=False,
    )
    primary_errors.to_parquet(
        PRIMARY_ERRORS_OUTPUT,
        index=False,
    )
    sensitivity_errors.to_parquet(
        SENSITIVITY_ERRORS_OUTPUT,
        index=False,
    )
    REPORT_OUTPUT.write_text(
        report,
        encoding="utf-8",
    )

    log("")
    log(report)
    log("")
    log(f"[SAVED] {PRIMARY_SUMMARY_OUTPUT}")
    log(f"[SAVED] {SENSITIVITY_SUMMARY_OUTPUT}")
    log(f"[SAVED] {PRIMARY_ERRORS_OUTPUT}")
    log(f"[SAVED] {SENSITIVITY_ERRORS_OUTPUT}")
    log(f"[SAVED] {REPORT_OUTPUT}")
    log("")

    primary_error_ratio = float(
        primary_validated["has_validation_error"].mean()
    )
    sensitivity_error_ratio = float(
        sensitivity_validated["has_validation_error"].mean()
    )

    if primary_error_ratio > 0 or sensitivity_error_ratio > 0:
        log(
            "[WARNING] VALIDATION COMPLETED WITH ERRORS. "
            "Inspect the saved error parquet files."
        )
    else:
        log("[PASSED] CARRY-STATE FEATURE VALIDATION COMPLETE")


if __name__ == "__main__":
    main()