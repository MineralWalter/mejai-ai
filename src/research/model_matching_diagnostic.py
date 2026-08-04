from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

LIFECYCLE_FILE = Path(
    "data/analysis/mejai_purchase_lifecycles.json"
)

CONTROL_FILE = Path(
    "data/analysis/mejai_control_candidates.parquet"
)

# Optional. Used for case outcome/champion diagnostics when
# the file exists and contains compatible identifiers.
RESEARCH_FILE = Path(
    "data/analysis/mejai_research_dataset.parquet"
)

OUTPUT_DIR = Path(
    "data/analysis/matching_diagnostics"
)

MATCHED_CASES_FILE = (
    OUTPUT_DIR / "matched_case_coverage.csv"
)

STATUS_COVERAGE_FILE = (
    OUTPUT_DIR / "matching_coverage_by_status.csv"
)

TIME_COVERAGE_FILE = (
    OUTPUT_DIR / "matching_coverage_by_purchase_time.csv"
)

BALANCE_FILE = (
    OUTPUT_DIR / "matched_covariate_balance.csv"
)

CONTROL_REUSE_FILE = (
    OUTPUT_DIR / "control_player_reuse.csv"
)

CONTROL_MATCH_REUSE_FILE = (
    OUTPUT_DIR / "control_match_reuse.csv"
)

SCORE_OUTLIERS_FILE = (
    OUTPUT_DIR / "matching_score_outliers.csv"
)

CASE_CONTROL_OUTCOMES_FILE = (
    OUTPUT_DIR / "matched_case_control_outcomes.csv"
)

# Purchase-time bins used for coverage diagnostics.
PURCHASE_TIME_BINS_MINUTES = [
    0,
    10,
    15,
    20,
    25,
    30,
    35,
    40,
    50,
    60,
    np.inf,
]

# Number of highest-score matched rows saved for inspection.
TOP_SCORE_OUTLIERS = 250


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(message)


# ============================================================
# GENERAL HELPERS
# ============================================================

def first_existing_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column

    return None


def to_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series

    if pd.api.types.is_numeric_dtype(series):
        return series.map(
            {
                1: True,
                0: False,
                1.0: True,
                0.0: False,
            }
        )

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "win": True,
            "loss": False,
            "won": True,
            "lost": False,
        }
    )


def relative_gap_series(case_values, control_values):
    denominator = np.maximum.reduce(
        [
            case_values.abs().to_numpy(),
            control_values.abs().to_numpy(),
            np.ones(len(case_values)),
        ]
    )

    return (
        (case_values - control_values)
        .abs()
        .to_numpy()
        / denominator
    )


def standardized_mean_difference(
    case_values,
    control_values,
):
    case_values = pd.to_numeric(
        case_values,
        errors="coerce",
    )

    control_values = pd.to_numeric(
        control_values,
        errors="coerce",
    )

    valid = (
        case_values.notna()
        & control_values.notna()
    )

    case_values = case_values[valid]
    control_values = control_values[valid]

    if len(case_values) < 2:
        return np.nan

    case_variance = case_values.var(ddof=1)
    control_variance = control_values.var(ddof=1)

    pooled_sd = np.sqrt(
        (
            case_variance
            + control_variance
        )
        / 2
    )

    if pooled_sd == 0 or pd.isna(pooled_sd):
        return 0.0

    return (
        case_values.mean()
        - control_values.mean()
    ) / pooled_sd


def make_case_id(
    match_id,
    participant_id,
    purchase_timestamp,
):
    return (
        match_id.astype(str)
        + "_"
        + participant_id.astype(int).astype(str)
        + "_"
        + purchase_timestamp.astype(int).astype(str)
    )


# ============================================================
# LOAD LIFECYCLES
# ============================================================

def load_lifecycles():
    if not LIFECYCLE_FILE.exists():
        log(
            f"[ERROR] Lifecycle file not found: "
            f"{LIFECYCLE_FILE}"
        )
        return pd.DataFrame()

    try:
        with open(
            LIFECYCLE_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except Exception as error:
        log(
            f"[ERROR] Could not read lifecycle file: "
            f"{error}"
        )
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    lifecycles = pd.DataFrame(data)

    required = [
        "match_id",
        "participant_id",
        "purchase_timestamp",
        "status",
    ]

    missing = [
        column
        for column in required
        if column not in lifecycles.columns
    ]

    if missing:
        log(
            f"[ERROR] Missing lifecycle columns: "
            f"{missing}"
        )
        return pd.DataFrame()

    lifecycles = lifecycles.copy()

    lifecycles["match_id"] = (
        lifecycles["match_id"].astype(str)
    )

    lifecycles["participant_id"] = pd.to_numeric(
        lifecycles["participant_id"],
        errors="coerce",
    )

    lifecycles["purchase_timestamp"] = pd.to_numeric(
        lifecycles["purchase_timestamp"],
        errors="coerce",
    )

    lifecycles = lifecycles.dropna(
        subset=[
            "match_id",
            "participant_id",
            "purchase_timestamp",
            "status",
        ]
    )

    lifecycles["participant_id"] = (
        lifecycles["participant_id"].astype(int)
    )

    lifecycles["purchase_timestamp"] = (
        lifecycles["purchase_timestamp"].astype(int)
    )

    lifecycles["status"] = (
        lifecycles["status"]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace(
            {
                "UNDO": "UNDONE",
            }
        )
    )

    lifecycles["case_id"] = make_case_id(
        lifecycles["match_id"],
        lifecycles["participant_id"],
        lifecycles["purchase_timestamp"],
    )

    lifecycles = lifecycles.drop_duplicates(
        subset=["case_id"],
        keep="first",
    )

    return lifecycles.reset_index(drop=True)


# ============================================================
# LOAD CONTROLS
# ============================================================

def load_controls():
    if not CONTROL_FILE.exists():
        log(
            f"[ERROR] Control candidate file not found: "
            f"{CONTROL_FILE}"
        )
        return pd.DataFrame()

    try:
        controls = pd.read_parquet(
            CONTROL_FILE
        )

    except Exception as error:
        log(
            f"[ERROR] Could not read control candidates: "
            f"{error}"
        )
        return pd.DataFrame()

    required = [
        "case_id",
        "control_match_id",
        "control_participant_id",
        "control_match_state_score",
    ]

    missing = [
        column
        for column in required
        if column not in controls.columns
    ]

    if missing:
        log(
            f"[ERROR] Missing control columns: "
            f"{missing}"
        )
        return pd.DataFrame()

    controls = controls.copy()

    controls["case_id"] = (
        controls["case_id"].astype(str)
    )

    controls["control_match_id"] = (
        controls["control_match_id"].astype(str)
    )

    controls["control_participant_id"] = (
        pd.to_numeric(
            controls["control_participant_id"],
            errors="coerce",
        )
    )

    controls["control_match_state_score"] = (
        pd.to_numeric(
            controls["control_match_state_score"],
            errors="coerce",
        )
    )

    controls = controls.dropna(
        subset=[
            "case_id",
            "control_match_id",
            "control_participant_id",
            "control_match_state_score",
        ]
    )

    controls["control_participant_id"] = (
        controls["control_participant_id"]
        .astype(int)
    )

    controls = controls.drop_duplicates(
        subset=[
            "case_id",
            "control_match_id",
            "control_participant_id",
        ]
    )

    return controls.reset_index(drop=True)


# ============================================================
# OPTIONAL RESEARCH DATA
# ============================================================

def load_optional_research_data():
    if not RESEARCH_FILE.exists():
        log(
            f"[INFO] Optional research dataset not found: "
            f"{RESEARCH_FILE}"
        )
        return pd.DataFrame()

    try:
        research = pd.read_parquet(
            RESEARCH_FILE
        )

    except Exception as error:
        log(
            f"[WARNING] Could not read research dataset: "
            f"{error}"
        )
        return pd.DataFrame()

    match_column = first_existing_column(
        research,
        [
            "match_id",
        ],
    )

    participant_column = first_existing_column(
        research,
        [
            "participant_id",
            "mejai_participant_id",
        ],
    )

    timestamp_column = first_existing_column(
        research,
        [
            "purchase_timestamp",
            "mejai_purchase_timestamp",
        ],
    )

    if (
        match_column is None
        or participant_column is None
        or timestamp_column is None
    ):
        log(
            "[WARNING] Research dataset does not contain "
            "compatible case identifiers."
        )
        return pd.DataFrame()

    research = research.copy()

    research[match_column] = (
        research[match_column].astype(str)
    )

    research[participant_column] = pd.to_numeric(
        research[participant_column],
        errors="coerce",
    )

    research[timestamp_column] = pd.to_numeric(
        research[timestamp_column],
        errors="coerce",
    )

    research = research.dropna(
        subset=[
            match_column,
            participant_column,
            timestamp_column,
        ]
    )

    research[participant_column] = (
        research[participant_column].astype(int)
    )

    research[timestamp_column] = (
        research[timestamp_column].astype(int)
    )

    research["case_id"] = make_case_id(
        research[match_column],
        research[participant_column],
        research[timestamp_column],
    )

    keep_columns = ["case_id"]

    optional_columns = [
        first_existing_column(
            research,
            [
                "outcome_win",
                "win",
                "case_outcome_win",
            ],
        ),
        first_existing_column(
            research,
            [
                "champion_name",
                "mejai_champion_name",
            ],
        ),
        first_existing_column(
            research,
            [
                "team_position",
                "position",
            ],
        ),
    ]

    for column in optional_columns:
        if (
            column is not None
            and column not in keep_columns
        ):
            keep_columns.append(column)

    research = (
        research[keep_columns]
        .drop_duplicates(
            subset=["case_id"],
            keep="first",
        )
    )

    rename_map = {}

    outcome_column = first_existing_column(
        research,
        [
            "outcome_win",
            "win",
            "case_outcome_win",
        ],
    )

    champion_column = first_existing_column(
        research,
        [
            "champion_name",
            "mejai_champion_name",
        ],
    )

    position_column = first_existing_column(
        research,
        [
            "team_position",
            "position",
        ],
    )

    if outcome_column is not None:
        rename_map[outcome_column] = (
            "case_outcome_win"
        )

    if champion_column is not None:
        rename_map[champion_column] = (
            "case_champion_name"
        )

    if position_column is not None:
        rename_map[position_column] = (
            "case_team_position"
        )

    return research.rename(
        columns=rename_map
    )


# ============================================================
# COVERAGE DIAGNOSTICS
# ============================================================

def build_case_coverage(
    lifecycles,
    controls,
    research,
):
    control_counts = (
        controls.groupby(
            "case_id"
        )
        .size()
        .rename("controls_retained")
    )

    coverage = lifecycles.merge(
        control_counts,
        on="case_id",
        how="left",
    )

    coverage["controls_retained"] = (
        coverage["controls_retained"]
        .fillna(0)
        .astype(int)
    )

    coverage["matched"] = (
        coverage["controls_retained"] > 0
    )

    coverage["purchase_time_seconds"] = (
        coverage["purchase_timestamp"]
        / 1000
    )

    coverage["purchase_time_minutes"] = (
        coverage["purchase_time_seconds"]
        / 60
    )

    if not research.empty:
        coverage = coverage.merge(
            research,
            on="case_id",
            how="left",
        )

    return coverage


def build_status_coverage(coverage):
    result = (
        coverage.groupby(
            "status",
            dropna=False,
        )
        .agg(
            total_cases=(
                "case_id",
                "nunique",
            ),
            matched_cases=(
                "matched",
                "sum",
            ),
            mean_purchase_min=(
                "purchase_time_minutes",
                "mean",
            ),
            median_purchase_min=(
                "purchase_time_minutes",
                "median",
            ),
        )
        .reset_index()
    )

    result["unmatched_cases"] = (
        result["total_cases"]
        - result["matched_cases"]
    )

    result["match_rate"] = (
        result["matched_cases"]
        / result["total_cases"]
    )

    return result.sort_values(
        "match_rate",
        ascending=False,
    )


def build_time_coverage(coverage):
    labels = []

    for start, end in zip(
        PURCHASE_TIME_BINS_MINUTES[:-1],
        PURCHASE_TIME_BINS_MINUTES[1:],
    ):
        if np.isinf(end):
            labels.append(
                f"{int(start)}+"
            )
        else:
            labels.append(
                f"{int(start)}-{int(end)}"
            )

    data = coverage.copy()

    data["purchase_time_bin_min"] = pd.cut(
        data["purchase_time_minutes"],
        bins=PURCHASE_TIME_BINS_MINUTES,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    result = (
        data.groupby(
            "purchase_time_bin_min",
            observed=False,
            dropna=False,
        )
        .agg(
            total_cases=(
                "case_id",
                "nunique",
            ),
            matched_cases=(
                "matched",
                "sum",
            ),
        )
        .reset_index()
    )

    result["unmatched_cases"] = (
        result["total_cases"]
        - result["matched_cases"]
    )

    result["match_rate"] = np.where(
        result["total_cases"] > 0,
        (
            result["matched_cases"]
            / result["total_cases"]
        ),
        np.nan,
    )

    return result


# ============================================================
# BALANCE DIAGNOSTICS
# ============================================================

def build_balance_table(controls):
    feature_pairs = [
        (
            "player_total_gold",
            "mejai_player_total_gold",
            "control_player_total_gold",
        ),
        (
            "player_level",
            "mejai_player_level",
            "control_player_level",
        ),
        (
            "player_xp",
            "mejai_player_xp",
            "control_player_xp",
        ),
        (
            "player_minions_killed",
            "mejai_player_minions_killed",
            "control_player_minions_killed",
        ),
        (
            "player_jungle_minions_killed",
            "mejai_player_jungle_minions_killed",
            "control_player_jungle_minions_killed",
        ),
        (
            "team_total_gold",
            "mejai_team_total_gold",
            "control_team_total_gold",
        ),
        (
            "team_xp",
            "mejai_team_xp",
            "control_team_xp",
        ),
        (
            "team_cs",
            "mejai_team_cs",
            "control_team_cs",
        ),
        (
            "team_total_gold_diff",
            "mejai_team_total_gold_diff",
            "control_team_total_gold_diff",
        ),
        (
            "team_xp_diff",
            "mejai_team_xp_diff",
            "control_team_xp_diff",
        ),
        (
            "team_cs_diff",
            "mejai_team_cs_diff",
            "control_team_cs_diff",
        ),
    ]

    rows = []

    for (
        feature,
        case_column,
        control_column,
    ) in feature_pairs:
        if (
            case_column not in controls.columns
            or control_column not in controls.columns
        ):
            continue

        case_values = pd.to_numeric(
            controls[case_column],
            errors="coerce",
        )

        control_values = pd.to_numeric(
            controls[control_column],
            errors="coerce",
        )

        valid = (
            case_values.notna()
            & control_values.notna()
        )

        case_values = case_values[valid]
        control_values = control_values[valid]

        if case_values.empty:
            continue

        relative_gaps = relative_gap_series(
            case_values,
            control_values,
        )

        smd = standardized_mean_difference(
            case_values,
            control_values,
        )

        rows.append(
            {
                "feature": feature,
                "paired_rows": len(case_values),
                "case_mean": case_values.mean(),
                "control_mean": control_values.mean(),
                "mean_difference": (
                    case_values
                    - control_values
                ).mean(),
                "mean_absolute_difference": (
                    case_values
                    - control_values
                ).abs().mean(),
                "mean_relative_gap": (
                    np.mean(relative_gaps)
                ),
                "median_relative_gap": (
                    np.median(relative_gaps)
                ),
                "standardized_mean_difference": smd,
                "absolute_smd": abs(smd),
                "balance_flag": (
                    "GOOD"
                    if abs(smd) < 0.10
                    else (
                        "CHECK"
                        if abs(smd) < 0.20
                        else "POOR"
                    )
                ),
            }
        )

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values(
            "absolute_smd",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# CONTROL REUSE
# ============================================================

def build_control_reuse(controls):
    player_reuse = (
        controls.groupby(
            [
                "control_match_id",
                "control_participant_id",
            ]
        )
        .agg(
            times_used=(
                "case_id",
                "nunique",
            ),
            mean_match_score=(
                "control_match_state_score",
                "mean",
            ),
            max_match_score=(
                "control_match_state_score",
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "times_used",
                "max_match_score",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    match_reuse = (
        controls.groupby(
            "control_match_id"
        )
        .agg(
            times_used=(
                "case_id",
                "nunique",
            ),
            unique_control_players=(
                "control_participant_id",
                "nunique",
            ),
            mean_match_score=(
                "control_match_state_score",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "times_used",
            ascending=False,
        )
    )

    return player_reuse, match_reuse


# ============================================================
# OUTCOME DIAGNOSTICS
# ============================================================

def build_outcome_summary(
    coverage,
    controls,
):
    rows = []

    if "case_outcome_win" in coverage.columns:
        matched_cases = coverage[
            coverage["matched"]
        ].copy()

        matched_cases["case_outcome_win"] = to_bool(
            matched_cases["case_outcome_win"]
        )

        valid_case_outcomes = (
            matched_cases["case_outcome_win"]
            .dropna()
        )

        if not valid_case_outcomes.empty:
            rows.append(
                {
                    "group": "matched_mejai_cases",
                    "observations": len(
                        valid_case_outcomes
                    ),
                    "win_rate": (
                        valid_case_outcomes.mean()
                    ),
                }
            )

    if "outcome_win" in controls.columns:
        control_outcomes = to_bool(
            controls["outcome_win"]
        ).dropna()

        if not control_outcomes.empty:
            rows.append(
                {
                    "group": "matched_control_rows",
                    "observations": len(
                        control_outcomes
                    ),
                    "win_rate": (
                        control_outcomes.mean()
                    ),
                }
            )

        case_level_control_outcome = (
            controls.assign(
                _control_win=to_bool(
                    controls["outcome_win"]
                )
            )
            .groupby("case_id")[
                "_control_win"
            ]
            .mean()
            .dropna()
        )

        if not case_level_control_outcome.empty:
            rows.append(
                {
                    "group": (
                        "mean_control_win_rate_per_case"
                    ),
                    "observations": len(
                        case_level_control_outcome
                    ),
                    "win_rate": (
                        case_level_control_outcome.mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    coverage,
    status_coverage,
    balance,
    player_reuse,
    match_reuse,
    outcomes,
    controls,
):
    total_cases = coverage["case_id"].nunique()

    matched_cases = (
        coverage.loc[
            coverage["matched"],
            "case_id",
        ]
        .nunique()
    )

    log("")
    log("=" * 72)
    log("MATCHING DIAGNOSTIC SUMMARY")
    log("=" * 72)

    log(
        f"Total Mejai purchase cases: "
        f"{total_cases:,}"
    )

    log(
        f"Matched cases: "
        f"{matched_cases:,}"
    )

    log(
        f"Unmatched cases: "
        f"{total_cases - matched_cases:,}"
    )

    log(
        f"Overall match rate: "
        f"{matched_cases / total_cases:.2%}"
    )

    log("")
    log("MATCH RATE BY LIFECYCLE STATUS")
    log(
        status_coverage[
            [
                "status",
                "total_cases",
                "matched_cases",
                "unmatched_cases",
                "match_rate",
            ]
        ].to_string(
            index=False,
            formatters={
                "match_rate": (
                    lambda value:
                    f"{value:.2%}"
                ),
            },
        )
    )

    if not balance.empty:
        log("")
        log("COVARIATE BALANCE")
        log(
            balance[
                [
                    "feature",
                    "standardized_mean_difference",
                    "mean_relative_gap",
                    "balance_flag",
                ]
            ].to_string(
                index=False,
                formatters={
                    "standardized_mean_difference": (
                        lambda value:
                        f"{value:.4f}"
                    ),
                    "mean_relative_gap": (
                        lambda value:
                        f"{value:.4f}"
                    ),
                },
            )
        )

    log("")
    log("CONTROL REUSE")

    log(
        f"Unique control players: "
        f"{len(player_reuse):,}"
    )

    log(
        f"Maximum uses of one control player: "
        f"{player_reuse['times_used'].max():,}"
    )

    log(
        f"Control players used more than once: "
        f"{(player_reuse['times_used'] > 1).sum():,}"
    )

    log(
        f"Maximum cases using one control match: "
        f"{match_reuse['times_used'].max():,}"
    )

    if (
        "mejai_player_level" in controls.columns
        and "control_player_level"
        in controls.columns
    ):
        level_gap = (
            pd.to_numeric(
                controls["mejai_player_level"],
                errors="coerce",
            )
            - pd.to_numeric(
                controls["control_player_level"],
                errors="coerce",
            )
        ).abs()

        log("")
        log("LEVEL GAP")

        log(
            level_gap.value_counts(
                dropna=False
            )
            .sort_index()
            .to_string()
        )

    if not outcomes.empty:
        log("")
        log("MATCHED OUTCOMES")

        printable = outcomes.copy()

        printable["win_rate"] = (
            printable["win_rate"]
            .map(
                lambda value:
                f"{value:.2%}"
            )
        )

        log(
            printable.to_string(
                index=False
            )
        )

    score = pd.to_numeric(
        controls[
            "control_match_state_score"
        ],
        errors="coerce",
    )

    log("")
    log("MATCHING SCORE")

    log(
        score.describe().to_string()
    )


# ============================================================
# SAVE
# ============================================================

def save_outputs(
    coverage,
    status_coverage,
    time_coverage,
    balance,
    player_reuse,
    match_reuse,
    score_outliers,
    outcomes,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    coverage.to_csv(
        MATCHED_CASES_FILE,
        index=False,
    )

    status_coverage.to_csv(
        STATUS_COVERAGE_FILE,
        index=False,
    )

    time_coverage.to_csv(
        TIME_COVERAGE_FILE,
        index=False,
    )

    balance.to_csv(
        BALANCE_FILE,
        index=False,
    )

    player_reuse.to_csv(
        CONTROL_REUSE_FILE,
        index=False,
    )

    match_reuse.to_csv(
        CONTROL_MATCH_REUSE_FILE,
        index=False,
    )

    score_outliers.to_csv(
        SCORE_OUTLIERS_FILE,
        index=False,
    )

    outcomes.to_csv(
        CASE_CONTROL_OUTCOMES_FILE,
        index=False,
    )

    log("")
    log(
        f"[SAVED] Diagnostics written to: "
        f"{OUTPUT_DIR}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    log("=" * 72)
    log("MEJAI MATCHING DIAGNOSTICS")
    log("=" * 72)

    lifecycles = load_lifecycles()

    if lifecycles.empty:
        log(
            "[ERROR] No usable lifecycle data"
        )
        return

    controls = load_controls()

    if controls.empty:
        log(
            "[ERROR] No usable control candidate data"
        )
        return

    research = load_optional_research_data()

    coverage = build_case_coverage(
        lifecycles,
        controls,
        research,
    )

    status_coverage = build_status_coverage(
        coverage
    )

    time_coverage = build_time_coverage(
        coverage
    )

    balance = build_balance_table(
        controls
    )

    player_reuse, match_reuse = (
        build_control_reuse(
            controls
        )
    )

    score_outliers = (
        controls.sort_values(
            "control_match_state_score",
            ascending=False,
        )
        .head(TOP_SCORE_OUTLIERS)
        .reset_index(drop=True)
    )

    outcomes = build_outcome_summary(
        coverage,
        controls,
    )

    print_summary(
        coverage,
        status_coverage,
        balance,
        player_reuse,
        match_reuse,
        outcomes,
        controls,
    )

    save_outputs(
        coverage,
        status_coverage,
        time_coverage,
        balance,
        player_reuse,
        match_reuse,
        score_outliers,
        outcomes,
    )

    log("")
    log(
        "[PASSED] MATCHING DIAGNOSTICS COMPLETE"
    )


if __name__ == "__main__":
    main()