from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

CONTROL_FILE = Path(
    "data/analysis/mejai_control_candidates.parquet"
)

RESEARCH_FILE = Path(
    "data/analysis/mejai_research_dataset.parquet"
)

OUTPUT_DIR = Path("data/analysis")

PRIMARY_OUTPUT_FILE = (
    OUTPUT_DIR / "mejai_matched_primary.parquet"
)

SENSITIVITY_OUTPUT_FILE = (
    OUTPUT_DIR / "mejai_matched_sensitivity.parquet"
)

PRIMARY_STATUSES = {
    "RETAINED",
    "SOLD",
}

SENSITIVITY_STATUSES = {
    "RETAINED",
    "SOLD",
    "UNDONE",
}


# ============================================================
# LOGGING
# ============================================================


def log(message):
    print(message)


# ============================================================
# HELPERS
# ============================================================


def first_existing_column(
    df,
    candidates,
    required=False,
    label=None,
):
    for column in candidates:
        if column in df.columns:
            return column

    if required:
        description = label or "/".join(candidates)
        raise ValueError(
            f"Could not find required column: {description}"
        )

    return None


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


def normalise_status(series):
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .replace({"UNDO": "UNDONE"})
    )


def normalise_boolean(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        return numeric.map(
            {
                1: True,
                0: False,
                1.0: True,
                0.0: False,
            }
        ).astype("boolean")

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return cleaned.map(
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
    ).astype("boolean")


# ============================================================
# LOAD CONTROL CANDIDATES
# ============================================================


def load_control_candidates():
    if not CONTROL_FILE.exists():
        log(
            f"[ERROR] Control candidate file not found: "
            f"{CONTROL_FILE}"
        )
        return pd.DataFrame()

    try:
        controls = pd.read_parquet(CONTROL_FILE)
    except Exception as error:
        log(
            f"[ERROR] Could not read control file: "
            f"{error}"
        )
        return pd.DataFrame()

    required = [
        "case_id",
        "match_id",
        "mejai_participant_id",
        "mejai_purchase_timestamp",
        "control_match_id",
        "control_participant_id",
        "control_snapshot_timestamp",
        "control_match_state_score",
        "outcome_win",
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

    for column in [
        "case_id",
        "match_id",
        "control_match_id",
    ]:
        controls[column] = controls[column].astype(str)

    integer_columns = [
        "mejai_participant_id",
        "mejai_purchase_timestamp",
        "control_participant_id",
        "control_snapshot_timestamp",
    ]

    for column in integer_columns:
        controls[column] = pd.to_numeric(
            controls[column],
            errors="coerce",
        )

    controls[
        "control_match_state_score"
    ] = pd.to_numeric(
        controls["control_match_state_score"],
        errors="coerce",
    )

    controls = controls.dropna(
        subset=[
            "case_id",
            "match_id",
            *integer_columns,
            "control_match_state_score",
            "outcome_win",
        ]
    )

    for column in integer_columns:
        controls[column] = controls[column].astype(int)

    status_column = first_existing_column(
        controls,
        [
            "mejai_lifecycle_status",
            "lifecycle_status",
            "status",
        ],
        required=True,
        label="Mejai lifecycle status",
    )

    controls[
        "mejai_lifecycle_status"
    ] = normalise_status(
        controls[status_column]
    )

    controls["outcome_win"] = normalise_boolean(
        controls["outcome_win"]
    )

    controls = controls.dropna(
        subset=["outcome_win"]
    )

    controls = controls.drop_duplicates(
        subset=[
            "case_id",
            "control_match_id",
            "control_participant_id",
        ],
        keep="first",
    )

    return controls.reset_index(drop=True)


# ============================================================
# LOAD CASE RESEARCH DATA
# ============================================================


def load_case_research_data():
    if not RESEARCH_FILE.exists():
        log(
            f"[ERROR] Research dataset not found: "
            f"{RESEARCH_FILE}"
        )
        return pd.DataFrame()

    try:
        research = pd.read_parquet(RESEARCH_FILE)
    except Exception as error:
        log(
            f"[ERROR] Could not read research dataset: "
            f"{error}"
        )
        return pd.DataFrame()

    match_column = first_existing_column(
        research,
        ["match_id"],
        required=True,
        label="match_id",
    )

    participant_column = first_existing_column(
        research,
        [
            "participant_id",
            "mejai_participant_id",
        ],
        required=True,
        label="case participant ID",
    )

    timestamp_column = first_existing_column(
        research,
        [
            "purchase_timestamp",
            "mejai_purchase_timestamp",
        ],
        required=True,
        label="purchase timestamp",
    )

    outcome_column = first_existing_column(
        research,
        [
            "outcome_win",
            "win",
            "case_outcome_win",
        ],
        required=True,
        label="case win outcome",
    )

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
            outcome_column,
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

    optional_columns = {
        "case_champion_name": first_existing_column(
            research,
            [
                "champion_name",
                "mejai_champion_name",
            ],
        ),
        "case_champion_id": first_existing_column(
            research,
            [
                "champion_id",
                "mejai_champion_id",
            ],
        ),
        "case_team_position": first_existing_column(
            research,
            [
                "team_position",
                "position",
            ],
        ),
        "case_team_id": first_existing_column(
            research,
            [
                "team_id",
                "mejai_team_id",
            ],
        ),
    }

    selected = pd.DataFrame(
        {
            "case_id": research["case_id"],
            "case_match_id": research[match_column],
            "case_participant_id": (
                research[participant_column]
            ),
            "case_observation_timestamp": (
                research[timestamp_column]
            ),
            "case_outcome_win": (
                normalise_boolean(
                    research[outcome_column]
                )
            ),
        }
    )

    for destination, source in (
        optional_columns.items()
    ):
        if source is not None:
            selected[destination] = research[source]

    selected = selected.dropna(
        subset=["case_outcome_win"]
    )

    selected = selected.drop_duplicates(
        subset=["case_id"],
        keep="first",
    )

    return selected.reset_index(drop=True)


# ============================================================
# BUILD CASE ROWS
# ============================================================


def build_case_rows(
    controls,
    case_research,
):
    case_source = (
        controls.sort_values(
            [
                "case_id",
                "control_match_state_score",
            ]
        )
        .drop_duplicates(
            subset=["case_id"],
            keep="first",
        )
        .copy()
    )

    case_source = case_source.merge(
        case_research,
        on="case_id",
        how="inner",
        validate="one_to_one",
    )

    case_rows = pd.DataFrame(
        {
            "matched_set_id": case_source["case_id"],
            "case_id": case_source["case_id"],
            "treatment": 1,
            "sample_type": "MEJAI",
            "lifecycle_status": (
                case_source[
                    "mejai_lifecycle_status"
                ]
            ),
            "match_id": case_source["match_id"],
            "participant_id": (
                case_source[
                    "mejai_participant_id"
                ]
            ),
            "observation_timestamp": (
                case_source[
                    "mejai_purchase_timestamp"
                ]
            ),
            "outcome_win": (
                case_source[
                    "case_outcome_win"
                ]
            ),
            "matching_weight": 1.0,
            "control_match_state_score": np.nan,
        }
    )

    metadata_map = {
        "case_team_position": "team_position",
        "case_champion_id": "champion_id",
        "case_champion_name": "champion_name",
        "case_team_id": "team_id",
    }

    for source, destination in metadata_map.items():
        if source in case_source.columns:
            case_rows[destination] = (
                case_source[source]
            )

    feature_map = {
        "mejai_player_current_gold": (
            "player_current_gold"
        ),
        "mejai_player_total_gold": (
            "player_total_gold"
        ),
        "mejai_player_level": "player_level",
        "mejai_player_xp": "player_xp",
        "mejai_player_minions_killed": (
            "player_minions_killed"
        ),
        "mejai_player_jungle_minions_killed": (
            "player_jungle_minions_killed"
        ),
        "mejai_team_total_gold": "team_total_gold",
        "mejai_enemy_total_gold": "enemy_total_gold",
        "mejai_team_total_gold_diff": (
            "team_total_gold_diff"
        ),
        "mejai_team_xp": "team_xp",
        "mejai_enemy_xp": "enemy_xp",
        "mejai_team_xp_diff": "team_xp_diff",
        "mejai_team_cs": "team_cs",
        "mejai_enemy_cs": "enemy_cs",
        "mejai_team_cs_diff": "team_cs_diff",
    }

    for source, destination in feature_map.items():
        if source in case_source.columns:
            case_rows[destination] = (
                case_source[source]
            )

    return case_rows


# ============================================================
# BUILD CONTROL ROWS
# ============================================================


def build_control_rows(
    controls,
    valid_case_ids,
):
    controls = controls[
        controls["case_id"].isin(valid_case_ids)
    ].copy()

    control_counts = (
        controls.groupby("case_id")
        .size()
        .rename("controls_in_set")
    )

    controls = controls.merge(
        control_counts,
        on="case_id",
        how="left",
        validate="many_to_one",
    )

    control_rows = pd.DataFrame(
        {
            "matched_set_id": controls["case_id"],
            "case_id": controls["case_id"],
            "treatment": 0,
            "sample_type": "CONTROL",
            "lifecycle_status": "CONTROL",
            "match_id": controls["control_match_id"],
            "participant_id": (
                controls["control_participant_id"]
            ),
            "observation_timestamp": (
                controls[
                    "control_snapshot_timestamp"
                ]
            ),
            "outcome_win": (
                controls["outcome_win"]
            ),
            "matching_weight": (
                1.0 / controls["controls_in_set"]
            ),
            "control_match_state_score": (
                controls[
                    "control_match_state_score"
                ]
            ),
        }
    )

    metadata_map = {
        "team_position": "team_position",
        "champion_id": "champion_id",
        "champion_name": "champion_name",
        "team_id": "team_id",
    }

    for source, destination in metadata_map.items():
        if source in controls.columns:
            control_rows[destination] = (
                controls[source]
            )

    feature_map = {
        "control_player_current_gold": (
            "player_current_gold"
        ),
        "control_player_total_gold": (
            "player_total_gold"
        ),
        "control_player_level": "player_level",
        "control_player_xp": "player_xp",
        "control_player_minions_killed": (
            "player_minions_killed"
        ),
        "control_player_jungle_minions_killed": (
            "player_jungle_minions_killed"
        ),
        "control_team_total_gold": "team_total_gold",
        "control_enemy_total_gold": "enemy_total_gold",
        "control_team_total_gold_diff": (
            "team_total_gold_diff"
        ),
        "control_team_xp": "team_xp",
        "control_enemy_xp": "enemy_xp",
        "control_team_xp_diff": "team_xp_diff",
        "control_team_cs": "team_cs",
        "control_enemy_cs": "enemy_cs",
        "control_team_cs_diff": "team_cs_diff",
    }

    for source, destination in feature_map.items():
        if source in controls.columns:
            control_rows[destination] = (
                controls[source]
            )

    return control_rows


# ============================================================
# BUILD MATCHED TABLE
# ============================================================


def build_matched_table(
    controls,
    case_research,
):
    case_rows = build_case_rows(
        controls,
        case_research,
    )

    if case_rows.empty:
        return pd.DataFrame()

    valid_case_ids = set(case_rows["case_id"])

    control_rows = build_control_rows(
        controls,
        valid_case_ids,
    )

    if control_rows.empty:
        return pd.DataFrame()

    all_columns = sorted(
        set(case_rows.columns)
        | set(control_rows.columns)
    )

    matched = pd.concat(
        [
            case_rows.reindex(columns=all_columns),
            control_rows.reindex(columns=all_columns),
        ],
        ignore_index=True,
    )

    matched["treatment"] = (
        matched["treatment"].astype(int)
    )

    matched["matching_weight"] = pd.to_numeric(
        matched["matching_weight"],
        errors="coerce",
    )

    matched["outcome_win"] = normalise_boolean(
        matched["outcome_win"]
    )

    return matched.sort_values(
        [
            "matched_set_id",
            "treatment",
            "control_match_state_score",
        ],
        ascending=[True, False, True],
        na_position="first",
    ).reset_index(drop=True)


# ============================================================
# SAMPLE FILTERING
# ============================================================


def filter_sample(
    matched,
    allowed_statuses,
):
    included_sets = set(
        matched.loc[
            (
                matched["treatment"] == 1
            )
            & (
                matched["lifecycle_status"].isin(
                    allowed_statuses
                )
            ),
            "matched_set_id",
        ]
    )

    return matched[
        matched["matched_set_id"].isin(
            included_sets
        )
    ].copy().reset_index(drop=True)


# ============================================================
# VALIDATION
# ============================================================


def validate_matched_sample(
    df,
    sample_name,
):
    if df.empty:
        raise ValueError(
            f"{sample_name} sample is empty"
        )

    set_counts = (
        df.groupby("matched_set_id")
        .agg(
            treatment_rows=("treatment", "sum"),
            total_rows=("treatment", "size"),
        )
    )

    treatment_weights = (
        df[df["treatment"] == 1]
        .groupby("matched_set_id")[
            "matching_weight"
        ]
        .sum()
        .rename("treatment_weight")
    )

    control_weights = (
        df[df["treatment"] == 0]
        .groupby("matched_set_id")[
            "matching_weight"
        ]
        .sum()
        .rename("control_weight")
    )

    set_summary = (
        set_counts.join(treatment_weights)
        .join(control_weights)
    )

    if not (
        set_summary["treatment_rows"] == 1
    ).all():
        raise ValueError(
            f"{sample_name}: not every set has "
            f"exactly one treatment row"
        )

    if not (
        set_summary["total_rows"] >= 2
    ).all():
        raise ValueError(
            f"{sample_name}: a matched set has "
            f"no controls"
        )

    if not np.isclose(
        set_summary["treatment_weight"],
        1.0,
    ).all():
        raise ValueError(
            f"{sample_name}: treatment weights "
            f"do not sum to one per set"
        )

    if not np.isclose(
        set_summary["control_weight"],
        1.0,
    ).all():
        raise ValueError(
            f"{sample_name}: control weights "
            f"do not sum to one per set"
        )

    if df["outcome_win"].isna().any():
        raise ValueError(
            f"{sample_name}: missing win outcomes"
        )

    duplicate_rows = df.duplicated(
        subset=[
            "matched_set_id",
            "treatment",
            "match_id",
            "participant_id",
        ]
    )

    if duplicate_rows.any():
        raise ValueError(
            f"{sample_name}: duplicate observations "
            f"found within matched sets"
        )

    return set_summary


# ============================================================
# SUMMARY
# ============================================================


def weighted_win_rate(
    df,
    treatment,
):
    sample = df[
        df["treatment"] == treatment
    ]

    return np.average(
        sample["outcome_win"].astype(float),
        weights=sample["matching_weight"],
    )


def print_sample_summary(
    df,
    sample_name,
):
    matched_sets = df[
        "matched_set_id"
    ].nunique()

    treatment_rows = (
        df["treatment"] == 1
    ).sum()

    control_rows = (
        df["treatment"] == 0
    ).sum()

    case_win_rate = weighted_win_rate(
        df,
        treatment=1,
    )

    control_win_rate = weighted_win_rate(
        df,
        treatment=0,
    )

    controls_per_set = (
        df[df["treatment"] == 0]
        .groupby("matched_set_id")
        .size()
    )

    log("")
    log("=" * 72)
    log(sample_name.upper())
    log("=" * 72)

    log(
        f"Matched sets: {matched_sets:,}"
    )

    log(
        f"Treatment rows: {treatment_rows:,}"
    )

    log(
        f"Control rows: {control_rows:,}"
    )

    log(
        f"Total matching weight: "
        f"{df['matching_weight'].sum():,.2f}"
    )

    log("")
    log("Controls per matched set:")

    log(
        controls_per_set.value_counts()
        .sort_index()
        .to_string()
    )

    log("")
    log("Treatment lifecycle status:")

    log(
        df.loc[
            df["treatment"] == 1,
            "lifecycle_status",
        ]
        .value_counts()
        .to_string()
    )

    log("")
    log(
        f"Weighted Mejai win rate: "
        f"{case_win_rate:.2%}"
    )

    log(
        f"Weighted control win rate: "
        f"{control_win_rate:.2%}"
    )

    log(
        f"Weighted raw difference: "
        f"{case_win_rate - control_win_rate:+.2%}"
    )


# ============================================================
# SAVE
# ============================================================


def save_sample(
    df,
    output_file,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        output_file,
        index=False,
    )

    log(f"[SAVED] {output_file}")


# ============================================================
# MAIN
# ============================================================


def main():
    log("=" * 72)
    log("BUILD MEJAI MATCHED ANALYSIS DATASETS")
    log("=" * 72)

    controls = load_control_candidates()

    if controls.empty:
        log(
            "[ERROR] No usable matched "
            "control candidates"
        )
        return

    case_research = load_case_research_data()

    if case_research.empty:
        log(
            "[ERROR] No usable case research data"
        )
        return

    log(
        f"Matched control rows loaded: "
        f"{len(controls):,}"
    )

    log(
        f"Unique matched cases loaded: "
        f"{controls['case_id'].nunique():,}"
    )

    matched = build_matched_table(
        controls,
        case_research,
    )

    if matched.empty:
        log(
            "[ERROR] Could not build matched "
            "analysis table"
        )
        return

    primary = filter_sample(
        matched,
        PRIMARY_STATUSES,
    )

    sensitivity = filter_sample(
        matched,
        SENSITIVITY_STATUSES,
    )

    validate_matched_sample(
        primary,
        "Primary",
    )

    validate_matched_sample(
        sensitivity,
        "Sensitivity",
    )

    print_sample_summary(
        primary,
        "Primary matched dataset",
    )

    print_sample_summary(
        sensitivity,
        "Sensitivity matched dataset",
    )

    save_sample(
        primary,
        PRIMARY_OUTPUT_FILE,
    )

    save_sample(
        sensitivity,
        SENSITIVITY_OUTPUT_FILE,
    )

    log("")
    log(
        "[PASSED] MATCHED ANALYSIS "
        "DATASETS CONSTRUCTED"
    )


if __name__ == "__main__":
    main()