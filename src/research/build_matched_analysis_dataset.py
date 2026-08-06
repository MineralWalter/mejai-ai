import numpy as np
import pandas as pd

from src.research.config import (
    PRIMARY_STATUSES,
    V2_CASE_ENRICHED,
    V2_CONTROL_ENRICHED,
    V2_MATCHING_DIR,
)


MAX_CONTROLS_PER_CASE = 3

# Moderate-project compromise:
# - current gold gets one clear hard caliper;
# - recent K/D/A remain ranking variables rather than hard exclusions.
MAX_CURRENT_GOLD_GAP = 750
STATE_DISTANCE_WEIGHT = 1.0
CURRENT_GOLD_DISTANCE_WEIGHT = 1.0
EVENT_DISTANCE_WEIGHT = 2.0

# Extra controls must stay close to the best candidate for that case.
# Both values are measured on standardised matching-distance scales.
MAX_SELECTION_SCORE_ABOVE_BEST = 0.75
MAX_STATE_DISTANCE_ABOVE_BEST = 0.75

PRIMARY_PAIR_FILE = (
    V2_MATCHING_DIR
    / "mejai_selected_controls_primary_variable_ratio.parquet"
)

SENSITIVITY_PAIR_FILE = (
    V2_MATCHING_DIR
    / "mejai_selected_controls_relaxed_variable_ratio.parquet"
)

ROBUSTNESS_PAIR_FILE = (
    V2_MATCHING_DIR
    / "mejai_selected_controls_primary_1to1.parquet"
)

PRIMARY_MATCHED_FILE = (
    V2_MATCHING_DIR
    / "mejai_matched_primary.parquet"
)

SENSITIVITY_MATCHED_FILE = (
    V2_MATCHING_DIR
    / "mejai_matched_relaxed.parquet"
)

ROBUSTNESS_MATCHED_FILE = (
    V2_MATCHING_DIR
    / "mejai_matched_primary_1to1_robustness.parquet"
)

PRIMARY_UNMATCHED_FILE = (
    V2_MATCHING_DIR
    / "mejai_unmatched_primary.csv"
)

SENSITIVITY_UNMATCHED_FILE = (
    V2_MATCHING_DIR
    / "mejai_unmatched_relaxed.csv"
)

SUMMARY_FILE = (
    V2_MATCHING_DIR
    / "matching_summary.csv"
)

EVENT_FEATURES = [
    "kills_last_5m",
    "deaths_last_5m",
    "assists_last_5m",
]

DARK_SEAL_FEATURE = (
    "dark_seal_purchased_before_observation"
)


def log(message=""):
    print(message)


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


def load_inputs():
    if not V2_CASE_ENRICHED.exists():
        raise FileNotFoundError(
            f"Enriched case file not found: "
            f"{V2_CASE_ENRICHED}\n"
            "Run the purchase-decision feature "
            "builder first."
        )

    if not V2_CONTROL_ENRICHED.exists():
        raise FileNotFoundError(
            f"Enriched control pool not found: "
            f"{V2_CONTROL_ENRICHED}\n"
            "Run the purchase-decision feature "
            "builder first."
        )

    cases = pd.read_parquet(
        V2_CASE_ENRICHED,
        engine="pyarrow",
    )

    candidates = pd.read_parquet(
        V2_CONTROL_ENRICHED,
        engine="pyarrow",
    )

    required_case_columns = [
        "case_id",
        "match_id",
        "participant_id",
        "purchase_timestamp",
        "lifecycle_status",
        "outcome_win",
        DARK_SEAL_FEATURE,
        *EVENT_FEATURES,
    ]

    required_candidate_columns = [
        "case_id",
        "match_id",
        "mejai_participant_id",
        "mejai_purchase_timestamp",
        "mejai_lifecycle_status",
        "control_match_id",
        "control_participant_id",
        "control_snapshot_timestamp",
        "control_snapshot_age_ms",
        "control_match_state_score",
        "mejai_player_current_gold",
        "control_player_current_gold",
        "outcome_win",
        f"mejai_{DARK_SEAL_FEATURE}",
        f"control_{DARK_SEAL_FEATURE}",
        *[
            f"mejai_{column}"
            for column in EVENT_FEATURES
        ],
        *[
            f"control_{column}"
            for column in EVENT_FEATURES
        ],
    ]

    missing_case_columns = [
        column
        for column in required_case_columns
        if column not in cases.columns
    ]

    missing_candidate_columns = [
        column
        for column in required_candidate_columns
        if column not in candidates.columns
    ]

    if missing_case_columns:
        raise ValueError(
            "Enriched case file is missing columns: "
            f"{missing_case_columns}"
        )

    if missing_candidate_columns:
        raise ValueError(
            "Enriched control pool is missing columns: "
            f"{missing_candidate_columns}"
        )

    cases = cases.copy()
    candidates = candidates.copy()

    cases["case_id"] = (
        cases["case_id"].astype(str)
    )

    candidates["case_id"] = (
        candidates["case_id"].astype(str)
    )

    cases["match_id"] = (
        cases["match_id"].astype(str)
    )

    candidates["match_id"] = (
        candidates["match_id"].astype(str)
    )

    candidates["control_match_id"] = (
        candidates["control_match_id"]
        .astype(str)
    )

    cases["lifecycle_status"] = (
        normalise_status(
            cases["lifecycle_status"]
        )
    )

    candidates["mejai_lifecycle_status"] = (
        normalise_status(
            candidates[
                "mejai_lifecycle_status"
            ]
        )
    )

    cases["outcome_win"] = (
        normalise_boolean(
            cases["outcome_win"]
        )
    )

    candidates["outcome_win"] = (
        normalise_boolean(
            candidates["outcome_win"]
        )
    )

    numeric_case_columns = [
        "participant_id",
        "purchase_timestamp",
        DARK_SEAL_FEATURE,
        *EVENT_FEATURES,
    ]

    numeric_candidate_columns = [
        "mejai_participant_id",
        "mejai_purchase_timestamp",
        "control_participant_id",
        "control_snapshot_timestamp",
        "control_snapshot_age_ms",
        "control_match_state_score",
        "mejai_player_current_gold",
        "control_player_current_gold",
        f"mejai_{DARK_SEAL_FEATURE}",
        f"control_{DARK_SEAL_FEATURE}",
        *[
            f"mejai_{column}"
            for column in EVENT_FEATURES
        ],
        *[
            f"control_{column}"
            for column in EVENT_FEATURES
        ],
    ]

    for column in numeric_case_columns:
        cases[column] = pd.to_numeric(
            cases[column],
            errors="coerce",
        )

    for column in numeric_candidate_columns:
        candidates[column] = pd.to_numeric(
            candidates[column],
            errors="coerce",
        )

    if cases[
        required_case_columns
    ].isna().any(axis=None):
        missing_counts = (
            cases[required_case_columns]
            .isna()
            .sum()
        )

        raise ValueError(
            "Enriched case file contains "
            "missing required values:\n"
            + missing_counts[
                missing_counts > 0
            ].to_string()
        )

    if candidates[
        required_candidate_columns
    ].isna().any(axis=None):
        missing_counts = (
            candidates[
                required_candidate_columns
            ]
            .isna()
            .sum()
        )

        raise ValueError(
            "Enriched control pool contains "
            "missing required values:\n"
            + missing_counts[
                missing_counts > 0
            ].to_string()
        )

    integer_case_columns = [
        "participant_id",
        "purchase_timestamp",
        DARK_SEAL_FEATURE,
        *EVENT_FEATURES,
    ]

    integer_candidate_columns = [
        "mejai_participant_id",
        "mejai_purchase_timestamp",
        "control_participant_id",
        "control_snapshot_timestamp",
        "control_snapshot_age_ms",
        f"mejai_{DARK_SEAL_FEATURE}",
        f"control_{DARK_SEAL_FEATURE}",
        *[
            f"mejai_{column}"
            for column in EVENT_FEATURES
        ],
        *[
            f"control_{column}"
            for column in EVENT_FEATURES
        ],
    ]

    for column in integer_case_columns:
        cases[column] = (
            cases[column].astype(int)
        )

    for column in integer_candidate_columns:
        candidates[column] = (
            candidates[column].astype(int)
        )

    if cases["case_id"].duplicated().any():
        raise ValueError(
            "Duplicate case IDs found in "
            "enriched case file"
        )

    unknown_case_ids = (
        set(candidates["case_id"])
        - set(cases["case_id"])
    )

    if unknown_case_ids:
        examples = "\n".join(
            sorted(unknown_case_ids)[:20]
        )

        raise ValueError(
            "Control pool contains case IDs "
            "absent from the case dataset:\n"
            + examples
        )

    candidates = (
        candidates.drop_duplicates(
            subset=[
                "case_id",
                "control_match_id",
                "control_participant_id",
                "control_snapshot_timestamp",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return (
        cases.reset_index(drop=True),
        candidates,
    )


def safe_scale(series):
    scale = float(
        pd.to_numeric(
            series,
            errors="coerce",
        ).std(ddof=0)
    )

    if (
        not np.isfinite(scale)
        or scale <= 0
    ):
        return 1.0

    return scale


def add_selection_scores(candidates):
    output = candidates.copy()

    output["dark_seal_history_match"] = (
        output[
            f"mejai_{DARK_SEAL_FEATURE}"
        ]
        == output[
            f"control_{DARK_SEAL_FEATURE}"
        ]
    )

    output["current_gold_gap"] = (
        output["mejai_player_current_gold"]
        - output["control_player_current_gold"]
    ).abs()

    for feature in EVENT_FEATURES:
        output[f"{feature}_gap"] = (
            output[f"mejai_{feature}"]
            - output[f"control_{feature}"]
        ).abs()

    state_scale = safe_scale(
        output["control_match_state_score"]
    )

    current_gold_scale = safe_scale(
        output["current_gold_gap"]
    )

    event_scales = {
        feature: safe_scale(
            output[f"{feature}_gap"]
        )
        for feature in EVENT_FEATURES
    }

    output[
        "state_distance_standardised"
    ] = (
        output["control_match_state_score"]
        / state_scale
    )

    output[
        "current_gold_distance_standardised"
    ] = (
        output["current_gold_gap"]
        / current_gold_scale
    )

    event_gap_columns = []

    for feature in EVENT_FEATURES:
        column = (
            f"{feature}_gap_standardised"
        )

        output[column] = (
            output[f"{feature}_gap"]
            / event_scales[feature]
        )

        event_gap_columns.append(column)

    output[
        "event_distance_standardised"
    ] = (
        output[event_gap_columns]
        .mean(axis=1)
    )

    output["selection_score"] = (
        STATE_DISTANCE_WEIGHT
        * output[
            "state_distance_standardised"
        ]
        + CURRENT_GOLD_DISTANCE_WEIGHT
        * output[
            "current_gold_distance_standardised"
        ]
        + EVENT_DISTANCE_WEIGHT
        * output[
            "event_distance_standardised"
        ]
    )

    log("")
    log("Selection-score scales:")
    log(
        "  Match-state score standard "
        f"deviation: {state_scale:.6f}"
    )

    log(
        "  Current-gold gap standard "
        f"deviation: {current_gold_scale:.6f}"
    )

    log(
        "  Score weights: "
        f"state={STATE_DISTANCE_WEIGHT:.1f}, "
        f"current_gold={CURRENT_GOLD_DISTANCE_WEIGHT:.1f}, "
        f"recent_events={EVENT_DISTANCE_WEIGHT:.1f}"
    )

    for feature in EVENT_FEATURES:
        log(
            f"  {feature} gap standard "
            f"deviation: "
            f"{event_scales[feature]:.6f}"
        )

    return output


def sort_candidates(candidates):
    return candidates.sort_values(
        [
            "case_id",
            "selection_score",
            "state_distance_standardised",
            "control_snapshot_age_ms",
            "control_match_id",
            "control_participant_id",
            "control_snapshot_timestamp",
        ],
        ascending=True,
        kind="mergesort",
    )


def apply_additional_control_caliper(candidates):
    if candidates.empty:
        return candidates

    output = sort_candidates(
        candidates
    ).copy()

    # Compare every extra control with the same best overall
    # candidate. The first-ranked candidate must always survive.
    best_selection_score = (
        output.groupby("case_id")[
            "selection_score"
        ]
        .transform("first")
    )

    best_state_distance = (
        output.groupby("case_id")[
            "state_distance_standardised"
        ]
        .transform("first")
    )

    output[
        "selection_score_above_best"
    ] = (
        output["selection_score"]
        - best_selection_score
    )

    output[
        "state_distance_above_best"
    ] = (
        output[
            "state_distance_standardised"
        ]
        - best_state_distance
    )

    output = output[
        (
            output[
                "selection_score_above_best"
            ]
            <= MAX_SELECTION_SCORE_ABOVE_BEST
        )
        & (
            output[
                "state_distance_above_best"
            ]
            <= MAX_STATE_DISTANCE_ABOVE_BEST
        )
    ].copy()

    return output


def take_up_to_three(candidates):
    if candidates.empty:
        return candidates

    candidates = (
        apply_additional_control_caliper(
            candidates
        )
    )

    selected = (
        sort_candidates(candidates)
        .groupby(
            "case_id",
            sort=False,
            group_keys=False,
        )
        .head(MAX_CONTROLS_PER_CASE)
        .copy()
    )

    selected["control_rank"] = (
        selected.groupby("case_id")
        .cumcount()
        + 1
    )

    selected["controls_in_set"] = (
        selected.groupby("case_id")[
            "case_id"
        ]
        .transform("size")
        .astype(int)
    )

    selected["matching_weight"] = (
        1.0
        / selected["controls_in_set"]
    )

    return selected.reset_index(drop=True)


def select_primary(cases, candidates):
    target_case_ids = set(
        cases.loc[
            cases[
                "lifecycle_status"
            ].isin(PRIMARY_STATUSES),
            "case_id",
        ]
    )

    eligible = candidates[
        candidates["case_id"].isin(
            target_case_ids
        )
        & candidates[
            "dark_seal_history_match"
        ]
        & (
            candidates[
                "current_gold_gap"
            ]
            <= MAX_CURRENT_GOLD_GAP
        )
    ].copy()

    selected = take_up_to_three(
        eligible
    )

    selected["selection_sample"] = (
        "PRIMARY_VARIABLE_1_TO_3_"
        "EXACT_DARK_SEAL"
    )

    selected[
        "selection_used_fallback"
    ] = False

    return selected, target_case_ids

def select_sensitivity(cases, candidates):
    target_case_ids = set(
        cases.loc[
            cases["lifecycle_status"].isin(
                PRIMARY_STATUSES
            ),
            "case_id",
        ]
    )

    # Apply every mandatory eligibility rule first.
    caliper_eligible = candidates[
        candidates["case_id"].isin(
            target_case_ids
        )
        & (
            candidates["current_gold_gap"]
            <= MAX_CURRENT_GOLD_GAP
        )
    ].copy()

    # An exact-history candidate only counts as available
    # when it also passes the current-gold caliper.
    exact_available_by_case = (
        caliper_eligible.groupby("case_id")[
            "dark_seal_history_match"
        ]
        .any()
    )

    exact_available = (
        caliper_eligible["case_id"]
        .map(exact_available_by_case)
        .fillna(False)
    )

    eligible = caliper_eligible[
        caliper_eligible[
            "dark_seal_history_match"
        ]
        | ~exact_available
    ].copy()

    selected = take_up_to_three(
        eligible
    )

    selected["selection_sample"] = (
        "RELAXED_VARIABLE_1_TO_3_"
        "EXACT_PREFERRED"
    )

    selected[
        "selection_used_fallback"
    ] = (
        ~selected[
            "dark_seal_history_match"
        ]
    )

    invalid_fallback = selected[
        selected[
            "selection_used_fallback"
        ]
        & selected["case_id"].map(
            exact_available_by_case
        ).fillna(False)
    ]

    if not invalid_fallback.empty:
        raise ValueError(
            "Sensitivity matching used a "
            "Dark Seal mismatch even though "
            "an eligible exact-history "
            "candidate was available"
        )

    return selected, target_case_ids

def select_primary_1to1(
    primary_selected,
):
    if primary_selected.empty:
        return primary_selected.copy()

    selected = (
        sort_candidates(
            primary_selected
        )
        .drop_duplicates(
            subset=["case_id"],
            keep="first",
        )
        .copy()
    )

    selected["control_rank"] = 1
    selected["controls_in_set"] = 1
    selected["matching_weight"] = 1.0
    selected["selection_sample"] = (
        "PRIMARY_1_TO_1_"
        "EXACT_DARK_SEAL"
    )

    selected[
        "selection_used_fallback"
    ] = False

    return selected.reset_index(drop=True)


def comparable_feature_suffixes(
    selected,
):
    mejai_suffixes = {
        column[len("mejai_"):]
        for column in selected.columns
        if column.startswith("mejai_")
    }

    control_suffixes = {
        column[len("control_"):]
        for column in selected.columns
        if column.startswith("control_")
    }

    excluded = {
        "participant_id",
        "purchase_timestamp",
        "snapshot_timestamp",
        "snapshot_age_ms",
        "match_state_score",
        "time_seconds",
        "group",
        "lifecycle_status",
        "match_id",
    }

    return sorted(
        (
            mejai_suffixes
            & control_suffixes
        )
        - excluded
    )


def build_long_matched_dataset(
    cases,
    selected,
):
    if selected.empty:
        return pd.DataFrame()

    selected = selected.copy()

    case_source = (
        sort_candidates(selected)
        .drop_duplicates(
            subset=["case_id"],
            keep="first",
        )
        .copy()
    )

    case_lookup = cases.set_index(
        "case_id",
        drop=False,
    )

    missing_case_ids = (
        set(case_source["case_id"])
        - set(case_lookup.index)
    )

    if missing_case_ids:
        examples = "\n".join(
            sorted(missing_case_ids)[:20]
        )

        raise ValueError(
            "Selected controls contain "
            "unknown case IDs:\n"
            + examples
        )

    selected_cases = (
        case_lookup.loc[
            case_source["case_id"]
        ]
        .reset_index(drop=True)
    )

    set_fallback = (
        selected.groupby("case_id")[
            "selection_used_fallback"
        ]
        .any()
        .reindex(case_source["case_id"])
        .to_numpy()
    )

    exact_controls_in_set = (
        selected.groupby("case_id")[
            "dark_seal_history_match"
        ]
        .sum()
        .reindex(case_source["case_id"])
        .astype(int)
        .to_numpy()
    )

    case_rows = pd.DataFrame(
        {
            "matched_set_id": (
                case_source["case_id"]
                .to_numpy()
            ),
            "case_id": (
                case_source["case_id"]
                .to_numpy()
            ),
            "treatment": 1,
            "sample_type": "MEJAI",
            "lifecycle_status": (
                selected_cases[
                    "lifecycle_status"
                ].to_numpy()
            ),
            "match_id": (
                selected_cases[
                    "match_id"
                ].to_numpy()
            ),
            "participant_id": (
                selected_cases[
                    "participant_id"
                ].to_numpy()
            ),
            "observation_timestamp": (
                selected_cases[
                    "purchase_timestamp"
                ].to_numpy()
            ),
            "outcome_win": (
                selected_cases[
                    "outcome_win"
                ].to_numpy()
            ),
            "matching_weight": 1.0,
            "controls_in_set": (
                case_source[
                    "controls_in_set"
                ].to_numpy()
            ),
            "exact_controls_in_set": (
                exact_controls_in_set
            ),
            "selection_sample": (
                case_source[
                    "selection_sample"
                ].to_numpy()
            ),
            "selection_used_fallback": (
                set_fallback
            ),
        }
    )

    optional_case_metadata = [
        "region",
        "team_position",
        "champion_id",
        "champion_name",
        "team_id",
    ]

    for column in optional_case_metadata:
        if column in selected_cases:
            case_rows[column] = (
                selected_cases[
                    column
                ].to_numpy()
            )

    control_rows = pd.DataFrame(
        {
            "matched_set_id": (
                selected["case_id"]
                .to_numpy()
            ),
            "case_id": (
                selected["case_id"]
                .to_numpy()
            ),
            "treatment": 0,
            "sample_type": "CONTROL",
            "lifecycle_status": "CONTROL",
            "match_id": (
                selected[
                    "control_match_id"
                ].to_numpy()
            ),
            "participant_id": (
                selected[
                    "control_participant_id"
                ].to_numpy()
            ),
            "observation_timestamp": (
                selected[
                    "control_snapshot_timestamp"
                ].to_numpy()
            ),
            "outcome_win": (
                selected[
                    "outcome_win"
                ].to_numpy()
            ),
            "matching_weight": (
                selected[
                    "matching_weight"
                ].to_numpy()
            ),
            "controls_in_set": (
                selected[
                    "controls_in_set"
                ].to_numpy()
            ),
            "exact_controls_in_set": (
                selected.groupby("case_id")[
                    "dark_seal_history_match"
                ]
                .transform("sum")
                .astype(int)
                .to_numpy()
            ),
            "selection_sample": (
                selected[
                    "selection_sample"
                ].to_numpy()
            ),
            "selection_used_fallback": (
                selected[
                    "selection_used_fallback"
                ].to_numpy()
            ),
            "control_rank": (
                selected[
                    "control_rank"
                ].to_numpy()
            ),
            "dark_seal_history_match": (
                selected[
                    "dark_seal_history_match"
                ].to_numpy()
            ),
            "selection_score": (
                selected[
                    "selection_score"
                ].to_numpy()
            ),
            "selection_score_above_best": (
                selected[
                    "selection_score_above_best"
                ].to_numpy()
            ),
            "state_distance_standardised": (
                selected[
                    "state_distance_standardised"
                ].to_numpy()
            ),
            "state_distance_above_best": (
                selected[
                    "state_distance_above_best"
                ].to_numpy()
            ),
            "current_gold_gap": (
                selected[
                    "current_gold_gap"
                ].to_numpy()
            ),
            "current_gold_distance_standardised": (
                selected[
                    "current_gold_distance_standardised"
                ].to_numpy()
            ),
            "event_distance_standardised": (
                selected[
                    "event_distance_standardised"
                ].to_numpy()
            ),
            "control_match_state_score": (
                selected[
                    "control_match_state_score"
                ].to_numpy()
            ),
            "control_snapshot_age_ms": (
                selected[
                    "control_snapshot_age_ms"
                ].to_numpy()
            ),
        }
    )

    optional_control_metadata = [
        "team_position",
        "champion_id",
        "champion_name",
        "team_id",
    ]

    for column in optional_control_metadata:
        if column in selected:
            control_rows[column] = (
                selected[column].to_numpy()
            )

    if "region" in selected:
        control_rows["region"] = (
            selected["region"].to_numpy()
        )
    elif "region" in selected_cases:
        region_map = (
            selected_cases.set_index(
                "case_id"
            )["region"]
        )

        control_rows["region"] = (
            selected["case_id"]
            .map(region_map)
            .to_numpy()
        )

    for suffix in comparable_feature_suffixes(
        selected
    ):
        case_rows[suffix] = (
            case_source[
                f"mejai_{suffix}"
            ].to_numpy()
        )

        control_rows[suffix] = (
            selected[
                f"control_{suffix}"
            ].to_numpy()
        )

    for feature in EVENT_FEATURES:
        gap_column = f"{feature}_gap"

        control_rows[gap_column] = (
            selected[gap_column]
            .to_numpy()
        )

    all_columns = list(
        dict.fromkeys(
            [
                *case_rows.columns,
                *control_rows.columns,
            ]
        )
    )

    matched = pd.concat(
        [
            case_rows.reindex(
                columns=all_columns
            ),
            control_rows.reindex(
                columns=all_columns
            ),
        ],
        ignore_index=True,
    )

    matched["treatment"] = (
        matched["treatment"]
        .astype(int)
    )

    matched["outcome_win"] = (
        normalise_boolean(
            matched["outcome_win"]
        )
    )

    matched["matching_weight"] = (
        pd.to_numeric(
            matched["matching_weight"],
            errors="coerce",
        )
    )

    return matched.sort_values(
        [
            "matched_set_id",
            "treatment",
            "control_rank",
        ],
        ascending=[
            True,
            False,
            True,
        ],
        na_position="first",
        kind="mergesort",
    ).reset_index(drop=True)


def validate_selected_controls(
    selected,
    sample_name,
    exact_dark_seal_required,
):
    if selected.empty:
        raise ValueError(
            f"{sample_name}: no controls "
            "were selected"
        )

    control_counts = (
        selected.groupby("case_id")
        .size()
    )

    if not control_counts.between(
        1,
        MAX_CONTROLS_PER_CASE,
        inclusive="both",
    ).all():
        raise ValueError(
            f"{sample_name}: invalid number "
            "of controls in a matched set"
        )

    duplicate_controls = (
        selected.duplicated(
            subset=[
                "case_id",
                "control_match_id",
                "control_participant_id",
                "control_snapshot_timestamp",
            ]
        )
    )

    if duplicate_controls.any():
        raise ValueError(
            f"{sample_name}: duplicate "
            "selected controls found"
        )

    if (
        exact_dark_seal_required
        and not selected[
            "dark_seal_history_match"
        ].all()
    ):
        raise ValueError(
            f"{sample_name}: a selected "
            "control has different Dark "
            "Seal history"
        )

    if selected[
        "selection_score"
    ].isna().any():
        raise ValueError(
            f"{sample_name}: missing "
            "selection scores"
        )

    if (
        selected[
            "selection_score_above_best"
        ]
        > MAX_SELECTION_SCORE_ABOVE_BEST
    ).any():
        raise ValueError(
            f"{sample_name}: an additional "
            "control exceeds the selection "
            "score caliper"
        )

    if (
        selected[
            "state_distance_above_best"
        ]
        > MAX_STATE_DISTANCE_ABOVE_BEST
    ).any():
        raise ValueError(
            f"{sample_name}: an additional "
            "control exceeds the state "
            "distance caliper"
        )

    weight_sums = (
        selected.groupby("case_id")[
            "matching_weight"
        ]
        .sum()
    )

    if not np.isclose(
        weight_sums,
        1.0,
    ).all():
        raise ValueError(
            f"{sample_name}: control "
            "weights do not sum to one "
            "per matched set"
        )


def validate_long_dataset(
    matched,
    selected,
    sample_name,
):
    if matched.empty:
        raise ValueError(
            f"{sample_name}: matched "
            "dataset is empty"
        )

    set_summary = (
        matched.groupby(
            "matched_set_id"
        )
        .agg(
            rows=(
                "treatment",
                "size",
            ),
            treatment_rows=(
                "treatment",
                "sum",
            ),
            treatment_weight=(
                "matching_weight",
                lambda values: values[
                    matched.loc[
                        values.index,
                        "treatment",
                    ]
                    == 1
                ].sum(),
            ),
            control_weight=(
                "matching_weight",
                lambda values: values[
                    matched.loc[
                        values.index,
                        "treatment",
                    ]
                    == 0
                ].sum(),
            ),
        )
    )

    expected_rows = (
        selected.groupby("case_id")
        .size()
        .add(1)
        .reindex(set_summary.index)
    )

    if not set_summary[
        "rows"
    ].eq(expected_rows).all():
        raise ValueError(
            f"{sample_name}: matched-set "
            "row counts do not match the "
            "selected controls"
        )

    if not set_summary[
        "treatment_rows"
    ].eq(1).all():
        raise ValueError(
            f"{sample_name}: every set "
            "must contain exactly one case"
        )

    if not np.isclose(
        set_summary[
            "treatment_weight"
        ],
        1.0,
    ).all():
        raise ValueError(
            f"{sample_name}: treatment "
            "weights are invalid"
        )

    if not np.isclose(
        set_summary[
            "control_weight"
        ],
        1.0,
    ).all():
        raise ValueError(
            f"{sample_name}: control "
            "weights are invalid"
        )

    if matched[
        "outcome_win"
    ].isna().any():
        raise ValueError(
            f"{sample_name}: missing "
            "outcome values"
        )

    if (
        matched[
            "matched_set_id"
        ].nunique()
        != selected[
            "case_id"
        ].nunique()
    ):
        raise ValueError(
            f"{sample_name}: selected and "
            "long-format matched-set counts "
            "differ"
        )


def build_unmatched_table(
    cases,
    target_case_ids,
    selected,
):
    matched_case_ids = set(
        selected["case_id"]
    )

    unmatched_case_ids = (
        target_case_ids
        - matched_case_ids
    )

    columns = [
        "case_id",
        "lifecycle_status",
        DARK_SEAL_FEATURE,
        "reason",
    ]

    if not unmatched_case_ids:
        return pd.DataFrame(
            columns=columns
        )

    output = cases[
        cases["case_id"].isin(
            unmatched_case_ids
        )
    ][
        [
            "case_id",
            "lifecycle_status",
            DARK_SEAL_FEATURE,
        ]
    ].copy()

    output["reason"] = (
        "NO_ELIGIBLE_CONTROL"
    )

    return output.sort_values(
        "case_id"
    ).reset_index(drop=True)


def control_reuse_summary(selected):
    observation_id = (
        selected[
            "control_match_id"
        ].astype(str)
        + "_"
        + selected[
            "control_participant_id"
        ].astype(str)
        + "_"
        + selected[
            "control_snapshot_timestamp"
        ].astype(str)
    )

    reuse = (
        observation_id.value_counts()
    )

    return {
        "unique_control_observations": (
            int(reuse.size)
        ),
        "reused_control_observations": (
            int((reuse > 1).sum())
        ),
        "maximum_control_reuse": (
            int(reuse.max())
            if not reuse.empty
            else 0
        ),
    }


def weighted_win_rate(
    matched,
    treatment,
):
    sample = matched[
        matched["treatment"]
        == treatment
    ]

    return float(
        np.average(
            sample[
                "outcome_win"
            ].astype(float),
            weights=sample[
                "matching_weight"
            ],
        )
    )


def build_sample_summary(
    sample_name,
    target_case_ids,
    selected,
    matched,
):
    matched_case_ids = set(
        selected["case_id"]
    )

    control_counts = (
        selected.groupby("case_id")
        .size()
    )

    case_fallback = (
        selected.groupby("case_id")[
            "selection_used_fallback"
        ]
        .any()
    )

    reuse = control_reuse_summary(
        selected
    )

    case_win_rate = weighted_win_rate(
        matched,
        treatment=1,
    )

    control_win_rate = weighted_win_rate(
        matched,
        treatment=0,
    )

    return {
        "sample": sample_name,
        "target_cases": (
            len(target_case_ids)
        ),
        "matched_cases": (
            len(matched_case_ids)
        ),
        "unmatched_cases": (
            len(
                target_case_ids
                - matched_case_ids
            )
        ),
        "coverage": (
            len(matched_case_ids)
            / len(target_case_ids)
            if target_case_ids
            else np.nan
        ),
        "selected_control_rows": (
            len(selected)
        ),
        "cases_with_1_control": (
            int((control_counts == 1).sum())
        ),
        "cases_with_2_controls": (
            int((control_counts == 2).sum())
        ),
        "cases_with_3_controls": (
            int((control_counts == 3).sum())
        ),
        "exact_dark_seal_control_rows": (
            int(
                selected[
                    "dark_seal_history_match"
                ].sum()
            )
        ),
        "fallback_control_rows": (
            int(
                (
                    ~selected[
                        "dark_seal_history_match"
                    ]
                ).sum()
            )
        ),
        "cases_using_fallback": (
            int(case_fallback.sum())
        ),
        "mean_selection_score": (
            float(
                selected[
                    "selection_score"
                ].mean()
            )
        ),
        "median_selection_score": (
            float(
                selected[
                    "selection_score"
                ].median()
            )
        ),
        "maximum_score_above_best": (
            float(
                selected[
                    "selection_score_above_best"
                ].max()
            )
        ),
        "maximum_state_above_best": (
            float(
                selected[
                    "state_distance_above_best"
                ].max()
            )
        ),
        "mean_current_gold_gap": (
            float(
                selected[
                    "current_gold_gap"
                ].mean()
            )
        ),
        "maximum_current_gold_gap": (
            float(
                selected[
                    "current_gold_gap"
                ].max()
            )
        ),
        "case_win_rate": (
            case_win_rate
        ),
        "control_win_rate": (
            control_win_rate
        ),
        "raw_win_rate_difference": (
            case_win_rate
            - control_win_rate
        ),
        **reuse,
    }


def print_summary(summary):
    log("")
    log("=" * 72)
    log(summary["sample"])
    log("=" * 72)

    log(
        f"Target cases: "
        f"{summary['target_cases']:,}"
    )

    log(
        f"Matched cases: "
        f"{summary['matched_cases']:,}"
    )

    log(
        f"Unmatched cases: "
        f"{summary['unmatched_cases']:,}"
    )

    log(
        f"Coverage: "
        f"{summary['coverage']:.2%}"
    )

    log(
        f"Selected control rows: "
        f"{summary['selected_control_rows']:,}"
    )

    log("")
    log("Controls per matched set:")

    log(
        f"  1 control: "
        f"{summary['cases_with_1_control']:,}"
    )

    log(
        f"  2 controls: "
        f"{summary['cases_with_2_controls']:,}"
    )

    log(
        f"  3 controls: "
        f"{summary['cases_with_3_controls']:,}"
    )

    log("")
    log(
        "Exact Dark Seal-history "
        f"control rows: "
        f"{summary['exact_dark_seal_control_rows']:,}"
    )

    log(
        f"Fallback control rows: "
        f"{summary['fallback_control_rows']:,}"
    )

    log(
        f"Cases using fallback: "
        f"{summary['cases_using_fallback']:,}"
    )

    log("")
    log(
        f"Mean selection score: "
        f"{summary['mean_selection_score']:.6f}"
    )

    log(
        f"Median selection score: "
        f"{summary['median_selection_score']:.6f}"
    )

    log(
        f"Maximum score above best: "
        f"{summary['maximum_score_above_best']:.6f}"
    )

    log(
        f"Maximum state distance above best: "
        f"{summary['maximum_state_above_best']:.6f}"
    )

    log(
        f"Mean current-gold gap: "
        f"{summary['mean_current_gold_gap']:.1f}"
    )

    log(
        f"Maximum current-gold gap: "
        f"{summary['maximum_current_gold_gap']:.1f}"
    )

    log("")
    log(
        f"Unique control observations: "
        f"{summary['unique_control_observations']:,}"
    )

    log(
        f"Reused control observations: "
        f"{summary['reused_control_observations']:,}"
    )

    log(
        f"Maximum control reuse: "
        f"{summary['maximum_control_reuse']:,}"
    )

    log("")
    log(
        f"Case win rate: "
        f"{summary['case_win_rate']:.2%}"
    )

    log(
        f"Control win rate: "
        f"{summary['control_win_rate']:.2%}"
    )

    log(
        f"Raw difference: "
        f"{summary['raw_win_rate_difference']:+.2%}"
    )


def save_outputs(
    primary_selected,
    sensitivity_selected,
    robustness_selected,
    primary_matched,
    sensitivity_matched,
    robustness_matched,
    primary_unmatched,
    sensitivity_unmatched,
    summaries,
):
    V2_MATCHING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    primary_selected.to_parquet(
        PRIMARY_PAIR_FILE,
        index=False,
        engine="pyarrow",
    )

    sensitivity_selected.to_parquet(
        SENSITIVITY_PAIR_FILE,
        index=False,
        engine="pyarrow",
    )

    robustness_selected.to_parquet(
        ROBUSTNESS_PAIR_FILE,
        index=False,
        engine="pyarrow",
    )

    primary_matched.to_parquet(
        PRIMARY_MATCHED_FILE,
        index=False,
        engine="pyarrow",
    )

    sensitivity_matched.to_parquet(
        SENSITIVITY_MATCHED_FILE,
        index=False,
        engine="pyarrow",
    )

    robustness_matched.to_parquet(
        ROBUSTNESS_MATCHED_FILE,
        index=False,
        engine="pyarrow",
    )

    primary_unmatched.to_csv(
        PRIMARY_UNMATCHED_FILE,
        index=False,
    )

    sensitivity_unmatched.to_csv(
        SENSITIVITY_UNMATCHED_FILE,
        index=False,
    )

    pd.DataFrame(summaries).to_csv(
        SUMMARY_FILE,
        index=False,
    )

    log("")
    log(f"[SAVED] {PRIMARY_PAIR_FILE}")
    log(f"[SAVED] {SENSITIVITY_PAIR_FILE}")
    log(f"[SAVED] {ROBUSTNESS_PAIR_FILE}")
    log(f"[SAVED] {PRIMARY_MATCHED_FILE}")
    log(f"[SAVED] {SENSITIVITY_MATCHED_FILE}")
    log(f"[SAVED] {ROBUSTNESS_MATCHED_FILE}")
    log(f"[SAVED] {PRIMARY_UNMATCHED_FILE}")
    log(f"[SAVED] {SENSITIVITY_UNMATCHED_FILE}")
    log(f"[SAVED] {SUMMARY_FILE}")


def main():
    log("=" * 72)
    log(
        "BUILD VERSION 2 VARIABLE-RATIO "
        "MATCHED DATASETS"
    )
    log("=" * 72)

    cases, candidates = load_inputs()

    log(
        f"Enriched case rows loaded: "
        f"{len(cases):,}"
    )

    log(
        f"Enriched candidate rows loaded: "
        f"{len(candidates):,}"
    )

    log(
        f"Candidate-covered cases: "
        f"{candidates['case_id'].nunique():,}"
    )

    candidates = add_selection_scores(
        candidates
    )

    (
        primary_selected,
        primary_target_case_ids,
    ) = select_primary(
        cases,
        candidates,
    )

    (
        sensitivity_selected,
        sensitivity_target_case_ids,
    ) = select_sensitivity(
        cases,
        candidates,
    )

    robustness_selected = (
        select_primary_1to1(
            primary_selected
        )
    )

    validate_selected_controls(
        primary_selected,
        "Primary variable-ratio",
        exact_dark_seal_required=True,
    )

    validate_selected_controls(
        sensitivity_selected,
        "Relaxed variable-ratio",
        exact_dark_seal_required=False,
    )

    validate_selected_controls(
        robustness_selected,
        "Primary 1-to-1 robustness",
        exact_dark_seal_required=True,
    )

    primary_matched = (
        build_long_matched_dataset(
            cases,
            primary_selected,
        )
    )

    sensitivity_matched = (
        build_long_matched_dataset(
            cases,
            sensitivity_selected,
        )
    )

    robustness_matched = (
        build_long_matched_dataset(
            cases,
            robustness_selected,
        )
    )

    validate_long_dataset(
        primary_matched,
        primary_selected,
        "Primary variable-ratio",
    )

    validate_long_dataset(
        sensitivity_matched,
        sensitivity_selected,
        "Relaxed variable-ratio",
    )

    validate_long_dataset(
        robustness_matched,
        robustness_selected,
        "Primary 1-to-1 robustness",
    )

    primary_unmatched = (
        build_unmatched_table(
            cases,
            primary_target_case_ids,
            primary_selected,
        )
    )

    sensitivity_unmatched = (
        build_unmatched_table(
            cases,
            sensitivity_target_case_ids,
            sensitivity_selected,
        )
    )

    summaries = [
        build_sample_summary(
            (
                "PRIMARY_VARIABLE_1_TO_3_"
                "EXACT_DARK_SEAL"
            ),
            primary_target_case_ids,
            primary_selected,
            primary_matched,
        ),
        build_sample_summary(
            (
                "RELAXED_VARIABLE_1_TO_3_"
                "EXACT_PREFERRED"
            ),
            sensitivity_target_case_ids,
            sensitivity_selected,
            sensitivity_matched,
        ),
        build_sample_summary(
            (
                "PRIMARY_1_TO_1_"
                "EXACT_DARK_SEAL"
            ),
            primary_target_case_ids,
            robustness_selected,
            robustness_matched,
        ),
    ]

    for summary in summaries:
        print_summary(summary)

    save_outputs(
        primary_selected,
        sensitivity_selected,
        robustness_selected,
        primary_matched,
        sensitivity_matched,
        robustness_matched,
        primary_unmatched,
        sensitivity_unmatched,
        summaries,
    )

    log("")
    log(
        "[PASSED] VERSION 2 VARIABLE-RATIO "
        "MATCHED DATASETS CONSTRUCTED"
    )


if __name__ == "__main__":
    main()
