from pathlib import Path

DATA_DIR = Path("data")
PARQUET_DIR = DATA_DIR / "parquet"
ANALYSIS_DIR = DATA_DIR / "analysis"
VALIDATION_DIR = DATA_DIR / "validation"

VALID_MATCH_MANIFEST = VALIDATION_DIR / "valid_match_manifest.parquet"
MEJAI_EVENT_CATALOGUE = ANALYSIS_DIR / "mejai_event_catalogue.json"
LIFECYCLE_FILE = ANALYSIS_DIR / "mejai_purchase_lifecycles.json"

CASE_DATASET = ANALYSIS_DIR / "mejai_research_dataset.parquet"

# Version 1: historical baseline.
V1_CONTROL_POOL = ANALYSIS_DIR / "mejai_control_candidates.parquet"
V1_ENRICHED_DIR = ANALYSIS_DIR / "enriched_candidate_pool"
V1_MATCHED_PRIMARY = ANALYSIS_DIR / "mejai_matched_primary.parquet"
V1_MATCHED_SENSITIVITY = ANALYSIS_DIR / "mejai_matched_sensitivity.parquet"

# Version 2: current analysis.
V2_CONTROL_POOL = ANALYSIS_DIR / "mejai_control_candidate_pool_generalized.parquet"
V2_ENRICHED_DIR = ANALYSIS_DIR / "generalized_candidate_pool"
V2_MATCHING_DIR = ANALYSIS_DIR / "generalized_matching"
V2_OUTCOME_DIR = ANALYSIS_DIR / "generalized_outcome"

V2_CASE_ENRICHED = V2_ENRICHED_DIR / "mejai_cases_enriched.parquet"
V2_CONTROL_ENRICHED = V2_ENRICHED_DIR / "mejai_controls_enriched.parquet"
V2_COMBINED_ENRICHED = V2_ENRICHED_DIR / "mejai_candidate_pool_enriched.parquet"

PRIMARY_STATUSES = {"RETAINED", "SOLD"}
SENSITIVITY_STATUSES = {"RETAINED", "SOLD", "UNDONE"}

RECENT_WINDOW_MS = 5 * 60 * 1000
DARK_SEAL_ITEM_ID = 1082
MEJAI_ITEM_ID = 3041

V2_EVENT_FEATURES = [
    "kills_last_5m",
    "deaths_last_5m",
    "assists_last_5m",
    "dark_seal_purchased_before_observation",
]

V2_CARRY_FEATURES = [
    "player_gold_diff_vs_role_opponent",
    "player_xp_diff_vs_role_opponent",
    "rest_of_team_gold_diff",
    "rest_of_team_xp_diff",
]