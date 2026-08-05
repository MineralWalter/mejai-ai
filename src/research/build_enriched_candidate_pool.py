from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from build_purchase_decision_features import (
    build_event_index,
    derive_observation_features,
    discover_event_files,
    load_relevant_events,
)
from build_carry_state_features import (
    attach_metadata_to_observations,
    build_role_opponent_lookup,
    discover_files,
    load_participant_metadata,
    load_relevant_snapshots,
    nearest_snapshot_before_observation,
)


# ============================================================
# CONFIG
# ============================================================

CASE_INPUT = Path("data/analysis/mejai_research_dataset.parquet")
CONTROL_INPUT = Path("data/analysis/mejai_control_candidates.parquet")

OUTPUT_DIR = Path("data/analysis/enriched_candidate_pool")
CASE_OUTPUT = OUTPUT_DIR / "mejai_case_candidates_enriched.parquet"
CONTROL_OUTPUT = OUTPUT_DIR / "mejai_control_candidates_enriched.parquet"
COMBINED_OUTPUT = OUTPUT_DIR / "mejai_candidate_pool_enriched.parquet"
DIAGNOSTICS_OUTPUT = OUTPUT_DIR / "enriched_candidate_pool_diagnostics.csv"

PRIMARY_STATUSES = {"RETAINED", "SOLD"}
SENSITIVITY_STATUSES = {"RETAINED", "SOLD", "UNDONE"}

ALIASES = {
    "match_id": [
        "match_id",
        "matchId",
        "control_match_id",
        "controlMatchId",
        "candidate_match_id",
        "candidateMatchId",
        "matched_match_id",
    ],
    "participant_id": [
        "participant_id",
        "participantId",
        "control_participant_id",
        "controlParticipantId",
        "candidate_participant_id",
        "candidateParticipantId",
        "matched_participant_id",
    ],
    "observation_timestamp": [
        "observation_timestamp",
        "purchase_timestamp",
        "purchaseTimestamp",
        "timestamp",
        "control_observation_timestamp",
        "control_timestamp",
        "candidate_observation_timestamp",
        "candidate_timestamp",
        "snapshot_timestamp",
        "frame_timestamp",
    ],
    "team_id": ["team_id", "teamId"],
    "team_position": [
        "team_position",
        "teamPosition",
        "position",
        "individual_position",
        "individualPosition",
    ],
    "team_total_gold_diff": [
        "team_total_gold_diff",
        "team_gold_diff",
        "teamGoldDiff",
    ],
    "team_xp_diff": ["team_xp_diff", "teamXpDiff"],
    "lifecycle_status": [
        "lifecycle_status",
        "status",
        "purchase_status",
        "mejai_status",
    ],
    "case_id": ["case_id", "purchase_id", "lifecycle_id"],
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def log(message: str) -> None:
    print(message)


def first_existing(columns, candidates):
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def canonicalize_columns(
    df: pd.DataFrame,
    source_name: str,
    source_type: str,
) -> pd.DataFrame:
    """
    Convert case and control tables into one observation schema.

    The control-candidate table contains both case-side (`mejai_*`) and
    control-side (`control_*`) fields. Control enrichment must explicitly use
    the control-side match, timestamp, and state columns.
    """
    output = df.copy()

    if source_type == "control":
        preferred_mapping = {
            "control_match_id": "match_id",
            "participant_id": "participant_id",
            "control_snapshot_timestamp": "observation_timestamp",
            "team_id": "team_id",
            "team_position": "team_position",
            "control_team_total_gold_diff": "team_total_gold_diff",
            "control_team_xp_diff": "team_xp_diff",
            "control_player_total_gold": "player_total_gold",
            "control_player_xp": "player_xp",
            "control_player_level": "player_level",
            "control_player_minions_killed": "player_minions_killed",
            "control_player_jungle_minions_killed": "player_jungle_minions_killed",
            "control_enemy_total_gold": "enemy_total_gold",
            "control_enemy_xp": "enemy_xp",
            "control_enemy_cs": "enemy_cs",
            "control_team_total_gold": "team_total_gold",
            "control_team_xp": "team_xp",
            "control_team_cs": "team_cs",
            "control_team_cs_diff": "team_cs_diff",
            "control_time_seconds": "control_time_seconds",
        }

        control_match_column = first_existing(
            output.columns,
            [
                "control_match_id",
                "controlMatchId",
                "candidate_match_id",
                "candidateMatchId",
            ],
        )
        control_participant_column = first_existing(
            output.columns,
            [
                "participant_id",
                "participantId",
                "control_participant_id",
                "controlParticipantId",
                "candidate_participant_id",
                "candidateParticipantId",
            ],
        )
        control_timestamp_column = first_existing(
            output.columns,
            [
                "control_snapshot_timestamp",
                "control_observation_timestamp",
                "control_timestamp",
                "candidate_snapshot_timestamp",
                "candidate_observation_timestamp",
            ],
        )

        resolved_required = {
            "control match": control_match_column,
            "control participant": control_participant_column,
            "control timestamp": control_timestamp_column,
        }
        unresolved = [
            label
            for label, column in resolved_required.items()
            if column is None
        ]

        if unresolved:
            raise ValueError(
                f"{source_name} could not resolve: {unresolved}\n"
                f"Available columns: {sorted(map(str, output.columns))}"
            )

        preferred_mapping[control_match_column] = "match_id"
        preferred_mapping[control_participant_column] = "participant_id"
        preferred_mapping[control_timestamp_column] = "observation_timestamp"

        # Drop case-side fields that would otherwise collide with the control
        # observation after renaming. Never drop the resolved control fields.
        protected_sources = {
            control_match_column,
            control_participant_column,
            control_timestamp_column,
        }
        for ambiguous in [
            "match_id",
            "observation_timestamp",
            "team_total_gold_diff",
            "team_xp_diff",
            "player_total_gold",
            "player_xp",
        ]:
            if ambiguous in output.columns and ambiguous not in protected_sources:
                output = output.drop(columns=[ambiguous])

        rename_map = {
            source: target
            for source, target in preferred_mapping.items()
            if source in output.columns and source != target
        }
        output = output.rename(columns=rename_map)

    elif source_type == "case":
        rename_map = {}

        for canonical, candidates in ALIASES.items():
            existing = first_existing(output.columns, candidates)
            if existing is not None and existing != canonical:
                rename_map[existing] = canonical

        output = output.rename(columns=rename_map)

    else:
        raise ValueError(f"Unknown source_type: {source_type}")

    required = ["match_id", "participant_id", "observation_timestamp"]
    missing = [column for column in required if column not in output.columns]

    if missing:
        available_columns = sorted(map(str, output.columns))
        related = {
            column: [
                candidate
                for candidate in available_columns
                if any(
                    token in candidate.lower()
                    for token in column.lower().split("_")
                )
            ][:20]
            for column in missing
        }

        raise ValueError(
            f"{source_name} is missing required columns: {missing}\n"
            f"Potential related columns: {related}\n"
            f"Available columns: {available_columns}"
        )

    output["match_id"] = output["match_id"].astype(str)
    output["participant_id"] = pd.to_numeric(
        output["participant_id"], errors="coerce"
    )
    output["observation_timestamp"] = pd.to_numeric(
        output["observation_timestamp"], errors="coerce"
    )
    output = output.dropna(
        subset=["match_id", "participant_id", "observation_timestamp"]
    )
    output["participant_id"] = output["participant_id"].astype(int)
    output["observation_timestamp"] = output[
        "observation_timestamp"
    ].astype(int)

    if "team_id" in output.columns:
        output["team_id"] = pd.to_numeric(
            output["team_id"], errors="coerce"
        )

    output["observation_id"] = (
        output["match_id"].astype(str)
        + "_"
        + output["participant_id"].astype(str)
        + "_"
        + output["observation_timestamp"].astype(str)
    )

    return output.reset_index(drop=True)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CASE_INPUT.exists():
        raise FileNotFoundError(f"Case input not found: {CASE_INPUT}")

    if not CONTROL_INPUT.exists():
        raise FileNotFoundError(f"Control input not found: {CONTROL_INPUT}")

    cases = canonicalize_columns(
        pd.read_parquet(CASE_INPUT),
        "case dataset",
        source_type="case",
    )
    controls = canonicalize_columns(
        pd.read_parquet(CONTROL_INPUT),
        "control dataset",
        source_type="control",
    )

    cases["candidate_source"] = "CASE"
    cases["treatment"] = 1

    controls["candidate_source"] = "CONTROL"
    controls["treatment"] = 0

    if "case_id" not in cases.columns:
        cases["case_id"] = "CASE_" + cases["observation_id"]

    if "case_id" not in controls.columns:
        controls["case_id"] = pd.NA

    if "lifecycle_status" in cases.columns:
        cases["lifecycle_status"] = cases["lifecycle_status"].astype(str).str.strip().str.upper()
        cases["primary_case_eligible"] = cases["lifecycle_status"].isin(PRIMARY_STATUSES).astype(int)
        cases["sensitivity_case_eligible"] = cases["lifecycle_status"].isin(SENSITIVITY_STATUSES).astype(int)
    else:
        warnings.warn(
            "No lifecycle-status column was found in the case dataset. "
            "All cases will be marked eligible for both analyses."
        )
        cases["primary_case_eligible"] = 1
        cases["sensitivity_case_eligible"] = 1

    controls["primary_case_eligible"] = 0
    controls["sensitivity_case_eligible"] = 0

    return cases, controls


def build_unique_observations(cases: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    columns = ["observation_id", "match_id", "participant_id", "observation_timestamp"]

    for optional in ["team_id", "team_position"]:
        if optional in cases.columns or optional in controls.columns:
            columns.append(optional)

    combined = pd.concat(
        [
            cases.reindex(columns=columns),
            controls.reindex(columns=columns),
        ],
        ignore_index=True,
    )

    combined = combined.sort_values("observation_id", kind="stable")
    combined = combined.drop_duplicates(subset=["observation_id"], keep="first")

    return combined.reset_index(drop=True)


# ============================================================
# COMPACT EVENT FEATURES
# ============================================================

def build_compact_features(observations: pd.DataFrame) -> pd.DataFrame:
    event_files = discover_event_files()
    relevant_match_ids = set(observations["match_id"])

    log(f"Event parquet files found: {len(event_files):,}")
    events = load_relevant_events(event_files, relevant_match_ids)
    event_index = build_event_index(events)
    del events

    return derive_observation_features(observations, event_index)


# ============================================================
# CARRY-STATE FEATURES
# ============================================================

def resolve_state_column(df: pd.DataFrame, canonical: str) -> str:
    candidates = ALIASES[canonical]
    column = first_existing(df.columns, [canonical, *candidates])

    if column is None:
        raise ValueError(
            f"Could not find {canonical} in candidate data. "
            f"Tried: {[canonical, *candidates]}"
        )

    return column


def build_state_lookup(cases: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([cases, controls], ignore_index=True, sort=False)

    gold_column = resolve_state_column(combined, "team_total_gold_diff")
    xp_column = resolve_state_column(combined, "team_xp_diff")

    state = combined[
        [
            "observation_id",
            "match_id",
            "participant_id",
            "observation_timestamp",
            gold_column,
            xp_column,
        ]
    ].copy()

    state = state.rename(
        columns={
            gold_column: "team_total_gold_diff_for_carry",
            xp_column: "team_xp_diff_for_carry",
        }
    )
    state = state.drop_duplicates(subset=["observation_id"], keep="first")

    return state.reset_index(drop=True)


def build_carry_features(
    observations: pd.DataFrame,
    cases: pd.DataFrame,
    controls: pd.DataFrame,
) -> pd.DataFrame:
    relevant_match_ids = set(observations["match_id"])

    participant_files = discover_files("participants", ("participant",))
    snapshot_files = discover_files("snapshots", ("snapshot", "frame"))

    log(f"Participant parquet files found: {len(participant_files):,}")
    log(f"Snapshot parquet files found: {len(snapshot_files):,}")

    metadata = load_participant_metadata(participant_files, relevant_match_ids)
    snapshots = load_relevant_snapshots(snapshot_files, relevant_match_ids)

    observations = attach_metadata_to_observations(observations, metadata)
    opponent_lookup = build_role_opponent_lookup(metadata)

    observations = observations.merge(
        opponent_lookup,
        on=["match_id", "participant_id"],
        how="left",
        validate="many_to_one",
    )

    if observations["opponent_participant_id"].isna().any():
        missing = int(observations["opponent_participant_id"].isna().sum())
        warnings.warn(f"{missing:,} observations have no same-role opponent")

    player_state = nearest_snapshot_before_observation(
        observations,
        snapshots,
        participant_column="participant_id",
        prefix="player",
    )

    opponent_observations = observations.dropna(subset=["opponent_participant_id"]).copy()
    opponent_observations["opponent_participant_id"] = (
        opponent_observations["opponent_participant_id"].astype(int)
    )

    opponent_state = nearest_snapshot_before_observation(
        opponent_observations,
        snapshots,
        participant_column="opponent_participant_id",
        prefix="opponent",
    )

    features = observations[
        [
            "observation_id",
            "team_id",
            "team_position",
            "opponent_participant_id",
        ]
    ].copy()

    features = features.merge(player_state, on="observation_id", how="left", validate="one_to_one")
    features = features.merge(opponent_state, on="observation_id", how="left", validate="one_to_one")
    features = features.merge(
        build_state_lookup(cases, controls),
        on="observation_id",
        how="left",
        validate="one_to_one",
    )

    features["player_gold_diff_vs_role_opponent"] = (
        features["player_total_gold"] - features["opponent_total_gold"]
    )
    features["player_xp_diff_vs_role_opponent"] = (
        features["player_xp"] - features["opponent_xp"]
    )
    features["rest_of_team_gold_diff"] = (
        features["team_total_gold_diff_for_carry"]
        - features["player_gold_diff_vs_role_opponent"]
    )
    features["rest_of_team_xp_diff"] = (
        features["team_xp_diff_for_carry"]
        - features["player_xp_diff_vs_role_opponent"]
    )

    return features[
        [
            "observation_id",
            "team_id",
            "team_position",
            "opponent_participant_id",
            "player_snapshot_timestamp",
            "opponent_snapshot_timestamp",
            "player_total_gold",
            "opponent_total_gold",
            "player_xp",
            "opponent_xp",
            "player_gold_diff_vs_role_opponent",
            "player_xp_diff_vs_role_opponent",
            "rest_of_team_gold_diff",
            "rest_of_team_xp_diff",
        ]
    ]


# ============================================================
# MERGING AND VALIDATION
# ============================================================

def drop_overlapping_feature_columns(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    removable = [
        column
        for column in feature_columns
        if column != "observation_id" and column in df.columns
    ]

    return df.drop(columns=removable) if removable else df


def attach_all_features(
    candidates: pd.DataFrame,
    compact_features: pd.DataFrame,
    carry_features: pd.DataFrame,
) -> pd.DataFrame:
    output = drop_overlapping_feature_columns(candidates.copy(), list(compact_features.columns))
    output = drop_overlapping_feature_columns(output, list(carry_features.columns))

    original_rows = len(output)
    output = output.merge(compact_features, on="observation_id", how="left", validate="many_to_one")
    output = output.merge(carry_features, on="observation_id", how="left", validate="many_to_one")

    if len(output) != original_rows:
        raise ValueError("Feature enrichment changed candidate row count")

    return output


def validate_enriched(df: pd.DataFrame, sample_name: str) -> None:
    required = [
        "dark_seal_purchased_before_observation",
        "kills_last_5m",
        "deaths_last_5m",
        "player_gold_diff_vs_role_opponent",
        "player_xp_diff_vs_role_opponent",
        "rest_of_team_gold_diff",
        "rest_of_team_xp_diff",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{sample_name} is missing enriched features: {missing}")

    coverage = df[required].notna().mean()
    low_coverage = coverage[coverage < 0.95]

    if not low_coverage.empty:
        warnings.warn(
            f"{sample_name} feature coverage below 95%:\n"
            f"{low_coverage.to_string()}"
        )


def build_diagnostics(cases: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    features = [
        "dark_seal_purchased_before_observation",
        "kills_last_5m",
        "deaths_last_5m",
        "assists_last_5m",
        "seconds_since_last_death",
        "player_gold_diff_vs_role_opponent",
        "player_xp_diff_vs_role_opponent",
        "rest_of_team_gold_diff",
        "rest_of_team_xp_diff",
    ]

    rows = []

    for sample_name, sample in [("cases", cases), ("controls", controls)]:
        for feature in features:
            numeric = pd.to_numeric(sample[feature], errors="coerce")

            rows.append(
                {
                    "sample": sample_name,
                    "feature": feature,
                    "rows": len(sample),
                    "non_missing_count": int(numeric.notna().sum()),
                    "non_missing_ratio": float(numeric.notna().mean()),
                    "mean": float(numeric.mean()) if numeric.notna().any() else np.nan,
                    "std": float(numeric.std()) if numeric.notna().any() else np.nan,
                    "minimum": float(numeric.min()) if numeric.notna().any() else np.nan,
                    "maximum": float(numeric.max()) if numeric.notna().any() else np.nan,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    log("=" * 80)
    log("BUILD ENRICHED CANDIDATE POOL")
    log("=" * 80)

    cases, controls = load_inputs()

    log(f"Case rows loaded: {len(cases):,}")
    log(f"Control rows loaded: {len(controls):,}")
    log(f"Primary committed cases: {int(cases['primary_case_eligible'].sum()):,}")
    log(f"Sensitivity cases: {int(cases['sensitivity_case_eligible'].sum()):,}")

    observations = build_unique_observations(cases, controls)
    log(f"Unique observations to enrich: {len(observations):,}")

    compact_features = build_compact_features(observations)
    carry_features = build_carry_features(observations, cases, controls)

    enriched_cases = attach_all_features(cases, compact_features, carry_features)
    enriched_controls = attach_all_features(controls, compact_features, carry_features)

    validate_enriched(enriched_cases, "cases")
    validate_enriched(enriched_controls, "controls")

    combined = pd.concat([enriched_cases, enriched_controls], ignore_index=True, sort=False)
    diagnostics = build_diagnostics(enriched_cases, enriched_controls)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    enriched_cases.to_parquet(CASE_OUTPUT, index=False)
    enriched_controls.to_parquet(CONTROL_OUTPUT, index=False)
    combined.to_parquet(COMBINED_OUTPUT, index=False)
    diagnostics.to_csv(DIAGNOSTICS_OUTPUT, index=False)

    log("")
    log(f"Enriched case rows: {len(enriched_cases):,}")
    log(f"Enriched control rows: {len(enriched_controls):,}")
    log(f"Combined candidate rows: {len(combined):,}")
    log("")
    log(f"[SAVED] {CASE_OUTPUT}")
    log(f"[SAVED] {CONTROL_OUTPUT}")
    log(f"[SAVED] {COMBINED_OUTPUT}")
    log(f"[SAVED] {DIAGNOSTICS_OUTPUT}")
    log("")
    log("[PASSED] ENRICHED CANDIDATE POOL CONSTRUCTED")


if __name__ == "__main__":
    main()