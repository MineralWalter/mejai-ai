from pathlib import Path


DATA_DIR = Path("data")
PARQUET_DIR = DATA_DIR / "parquet"
ANALYSIS_DIR = DATA_DIR / "analysis"
VALIDATION_DIR = DATA_DIR / "validation"

VALID_MATCH_MANIFEST = (
    VALIDATION_DIR / "valid_match_manifest.parquet"
)

MEJAI_EVENT_CATALOGUE = (
    ANALYSIS_DIR / "mejai_event_catalogue.json"
)

LIFECYCLE_FILE = (
    ANALYSIS_DIR / "mejai_purchase_lifecycles.json"
)

CASE_DATASET = (
    ANALYSIS_DIR / "mejai_research_dataset.parquet"
)


# ============================================================
# CURRENT ANALYSIS
# ============================================================

V2_CONTROL_POOL = (
    ANALYSIS_DIR
    / "mejai_control_candidate_pool_generalized.parquet"
)

V2_CASE_ENRICHED = (
    ANALYSIS_DIR
    / "mejai_research_dataset_event_enriched.parquet"
)

V2_CONTROL_ENRICHED = (
    ANALYSIS_DIR
    / "mejai_control_candidate_pool_generalized_event_enriched.parquet"
)

V2_MATCHING_DIR = (
    ANALYSIS_DIR / "generalized_matching"
)

V2_OUTCOME_DIR = (
    ANALYSIS_DIR / "generalized_outcome"
)


# ============================================================
# RESEARCH DEFINITIONS
# ============================================================

PRIMARY_STATUSES = {
    "RETAINED",
    "SOLD",
}

MEJAI_ITEM_ID = 3041