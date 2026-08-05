from __future__ import annotations

import ast
import json
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PRIMARY_INPUT = Path("data/analysis/mejai_matched_primary.parquet")
SENSITIVITY_INPUT = Path("data/analysis/mejai_matched_sensitivity.parquet")
AUDIT_INVENTORY_FILE = Path(
    "data/analysis/purchase_feature_audit/parquet_file_inventory.csv"
)
PARQUET_ROOT = Path("data/parquet")

OUTPUT_DIR = Path("data/analysis")
PRIMARY_OUTPUT = OUTPUT_DIR / "mejai_matched_primary_features.parquet"
SENSITIVITY_OUTPUT = OUTPUT_DIR / "mejai_matched_sensitivity_features.parquet"
DIAGNOSTICS_OUTPUT = OUTPUT_DIR / "purchase_decision_feature_diagnostics.csv"

RECENT_WINDOW_MS = 5 * 60 * 1000
DARK_SEAL_ITEM_ID = 1082
LOG_EVERY_FILES = 25

ALIASES = {
    "match_id": ["match_id", "matchId"],
    "event_type": ["event_type", "type", "event_name"],
    "timestamp": ["timestamp", "event_timestamp"],
    "participant_id": ["participant_id", "participantId"],
    "killer_id": ["killer_id", "killerId"],
    "victim_id": ["victim_id", "victimId"],
    "assisting_ids": [
        "assisting_ids",
        "assisting_participant_ids",
        "assistingParticipantIds",
        "assists",
    ],
    "item_id": ["item_id", "itemId"],
}


# ============================================================
# HELPERS
# ============================================================

def log(message: str) -> None:
    print(message)


def first_existing(columns, candidates):
    available = set(columns)
    return next((candidate for candidate in candidates if candidate in available), None)


def parse_assist_ids(value) -> list[int]:
    """Return assist IDs from arrays, lists, JSON strings, or scalar values."""
    if value is None:
        return []

    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        raw_values = list(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "nan", "null", "[]"}:
            return []

        try:
            parsed = json.loads(stripped)
        except Exception:
            try:
                parsed = ast.literal_eval(stripped)
            except Exception:
                parsed = None

        if isinstance(parsed, (list, tuple, set)):
            raw_values = list(parsed)
        elif parsed is not None:
            raw_values = [parsed]
        else:
            raw_values = [
                part.strip()
                for part in stripped.replace("[", "").replace("]", "").split(",")
                if part.strip()
            ]
    else:
        try:
            if pd.isna(value):
                return []
        except Exception:
            pass
        raw_values = [value]

    output = []
    for raw in raw_values:
        try:
            participant_id = int(float(raw))
        except (TypeError, ValueError):
            continue

        if participant_id > 0:
            output.append(participant_id)

    return output


def count_before(timestamps: np.ndarray, observation_timestamp: int) -> int:
    """Count events strictly before the observation time."""
    if len(timestamps) == 0:
        return 0

    return int(np.searchsorted(timestamps, observation_timestamp, side="left"))


def count_in_recent_window(
    timestamps: np.ndarray, observation_timestamp: int
) -> int:
    """Count events in [observation - 5 minutes, observation)."""
    if len(timestamps) == 0:
        return 0

    left = np.searchsorted(
        timestamps, observation_timestamp - RECENT_WINDOW_MS, side="left"
    )
    right = np.searchsorted(timestamps, observation_timestamp, side="left")
    return int(right - left)


def last_before(timestamps: np.ndarray, observation_timestamp: int) -> float:
    """Return the latest event timestamp strictly before observation."""
    if len(timestamps) == 0:
        return np.nan

    position = np.searchsorted(timestamps, observation_timestamp, side="left") - 1
    return float(timestamps[position]) if position >= 0 else np.nan


def get_array(mapping, key) -> np.ndarray:
    return mapping.get(key, np.empty(0, dtype=np.int64))


def append_timestamp(mapping, key, timestamp) -> None:
    mapping[key].append(int(timestamp))


# ============================================================
# MATCHED DATA
# ============================================================

def load_matched_dataset(path: Path, sample_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{sample_name} input not found: {path}")

    df = pd.read_parquet(path)
    required = [
        "matched_set_id",
        "case_id",
        "treatment",
        "match_id",
        "participant_id",
        "observation_timestamp",
        "outcome_win",
        "matching_weight",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{sample_name} is missing required columns: {missing}")

    df = df.copy()
    df["match_id"] = df["match_id"].astype(str)
    df["participant_id"] = pd.to_numeric(df["participant_id"], errors="coerce")
    df["observation_timestamp"] = pd.to_numeric(
        df["observation_timestamp"], errors="coerce"
    )
    df = df.dropna(
        subset=["match_id", "participant_id", "observation_timestamp"]
    )
    df["participant_id"] = df["participant_id"].astype(int)
    df["observation_timestamp"] = df["observation_timestamp"].astype(int)

    return df.reset_index(drop=True)


def build_observation_table(sensitivity: pd.DataFrame) -> pd.DataFrame:
    observations = sensitivity[
        ["match_id", "participant_id", "observation_timestamp"]
    ].drop_duplicates(
        subset=["match_id", "participant_id", "observation_timestamp"]
    )
    observations = observations.reset_index(drop=True)
    observations["observation_id"] = (
        observations["match_id"].astype(str)
        + "_"
        + observations["participant_id"].astype(str)
        + "_"
        + observations["observation_timestamp"].astype(str)
    )

    return observations


# ============================================================
# EVENT FILE DISCOVERY
# ============================================================

def discover_event_files() -> list[Path]:
    if AUDIT_INVENTORY_FILE.exists():
        inventory = pd.read_csv(AUDIT_INVENTORY_FILE)

        if {"path", "table_type"}.issubset(inventory.columns):
            paths = (
                inventory.loc[inventory["table_type"] == "events", "path"]
                .dropna()
                .astype(str)
                .map(Path)
                .tolist()
            )
            existing = sorted(path for path in paths if path.exists())
            if existing:
                return existing

    if not PARQUET_ROOT.exists():
        raise FileNotFoundError(f"Parquet root not found: {PARQUET_ROOT}")

    paths = sorted(
        path
        for path in PARQUET_ROOT.rglob("*.parquet")
        if "event" in str(path).lower() or "timeline" in str(path).lower()
    )
    if not paths:
        raise FileNotFoundError("No event parquet files found")

    return paths


def parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return [field.name for field in pq.ParquetFile(path).schema_arrow]
    except Exception:
        return list(pd.read_parquet(path).head(0).columns)


def select_event_columns(available_columns):
    selected = {
        canonical: first_existing(available_columns, candidates)
        for canonical, candidates in ALIASES.items()
    }

    if any(selected[name] is None for name in ("match_id", "event_type", "timestamp")):
        return selected, []

    actual_columns = sorted(
        {column for column in selected.values() if column is not None}
    )
    return selected, actual_columns


# ============================================================
# EVENT LOADING
# ============================================================

def normalize_event_frame(raw: pd.DataFrame, selected: dict) -> pd.DataFrame:
    rename_map = {
        actual: canonical
        for canonical, actual in selected.items()
        if actual is not None
    }
    events = raw.rename(columns=rename_map).copy()

    events["match_id"] = events["match_id"].astype(str)
    events["event_type"] = events["event_type"].astype(str).str.strip().str.upper()
    events["timestamp"] = pd.to_numeric(events["timestamp"], errors="coerce")
    events = events.dropna(subset=["match_id", "event_type", "timestamp"])
    events["timestamp"] = events["timestamp"].astype(int)

    for column in ["participant_id", "killer_id", "victim_id", "item_id"]:
        if column not in events.columns:
            events[column] = np.nan
        events[column] = pd.to_numeric(events[column], errors="coerce")

    if "assisting_ids" not in events.columns:
        events["assisting_ids"] = None

    # Tuples preserve assist IDs and remain hashable for drop_duplicates().
    events["assisting_ids"] = events["assisting_ids"].map(
        lambda value: tuple(parse_assist_ids(value))
    )

    return events


def load_relevant_events(
    event_files: list[Path], relevant_match_ids: set[str]
) -> pd.DataFrame:
    relevant_match_ids = set(map(str, relevant_match_ids))
    frames = []
    retained_rows = 0
    skipped = 0

    for index, path in enumerate(event_files, start=1):
        try:
            available = parquet_columns(path)
            selected, columns = select_event_columns(available)
            if not columns:
                skipped += 1
                continue

            raw = pd.read_parquet(path, columns=columns)
            match_column = selected["match_id"]
            raw = raw[raw[match_column].astype(str).isin(relevant_match_ids)]
            if raw.empty:
                continue

            events = normalize_event_frame(raw, selected)
            retained_rows += len(events)
            frames.append(events)

        except Exception as error:
            warnings.warn(f"Could not process event file {path}: {error}")

        if index % LOG_EVERY_FILES == 0 or index == len(event_files):
            log(
                f"Event files processed: {index:,} / {len(event_files):,} | "
                f"retained rows: {retained_rows:,}"
            )

    if not frames:
        raise ValueError("No relevant event rows were loaded")

    events = pd.concat(frames, ignore_index=True)
    events = events.drop_duplicates().sort_values(
        ["match_id", "timestamp"], kind="stable"
    )
    events = events.reset_index(drop=True)

    log(f"Relevant events loaded: {len(events):,}")
    if skipped:
        log(f"Event files skipped for missing required columns: {skipped:,}")

    return events


# ============================================================
# COMPACT EVENT INDEX
# ============================================================

def build_event_index(events: pd.DataFrame) -> dict:
    kills = defaultdict(list)
    deaths = defaultdict(list)
    assists = defaultdict(list)
    dark_seal_purchases = defaultdict(list)

    champion_kills = events[events["event_type"] == "CHAMPION_KILL"]
    for row in champion_kills.itertuples(index=False):
        match_id = str(row.match_id)
        timestamp = int(row.timestamp)

        if not pd.isna(row.killer_id) and int(row.killer_id) > 0:
            append_timestamp(kills, (match_id, int(row.killer_id)), timestamp)

        if not pd.isna(row.victim_id) and int(row.victim_id) > 0:
            append_timestamp(deaths, (match_id, int(row.victim_id)), timestamp)

        for assist_id in parse_assist_ids(row.assisting_ids):
            append_timestamp(assists, (match_id, assist_id), timestamp)

    purchases = events[events["event_type"] == "ITEM_PURCHASED"]
    for row in purchases.itertuples(index=False):
        if pd.isna(row.participant_id) or pd.isna(row.item_id):
            continue

        participant_id = int(row.participant_id)
        if participant_id <= 0 or int(row.item_id) != DARK_SEAL_ITEM_ID:
            continue

        append_timestamp(
            dark_seal_purchases,
            (str(row.match_id), participant_id),
            int(row.timestamp),
        )

    mappings = {
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "dark_seal_purchases": dark_seal_purchases,
    }

    for mapping in mappings.values():
        for key, values in mapping.items():
            mapping[key] = np.sort(np.asarray(values, dtype=np.int64))

    return mappings


# ============================================================
# COMPACT FEATURE CONSTRUCTION
# ============================================================

def derive_observation_features(
    observations: pd.DataFrame, event_index: dict
) -> pd.DataFrame:
    """
    Build only the compact event-derived state needed by the revised analysis.

    Economic carry state is intentionally handled elsewhere. This script only
    provides:
      - Dark Seal eligibility/history
      - recent five-minute kills, deaths, and assists
      - time since the player's last death

    Every event must occur strictly before the observation timestamp.
    """
    rows = []

    for number, row in enumerate(observations.itertuples(index=False), start=1):
        match_id = str(row.match_id)
        participant_id = int(row.participant_id)
        timestamp = int(row.observation_timestamp)
        player_key = (match_id, participant_id)

        kills = get_array(event_index["kills"], player_key)
        deaths = get_array(event_index["deaths"], player_key)
        assists = get_array(event_index["assists"], player_key)
        dark_seal_purchases = get_array(
            event_index["dark_seal_purchases"], player_key
        )
        last_death = last_before(deaths, timestamp)

        rows.append(
            {
                "observation_id": row.observation_id,
                "purchase_time_minutes": timestamp / 60_000.0,
                "dark_seal_purchased_before_observation": int(
                    count_before(dark_seal_purchases, timestamp) > 0
                ),
                "kills_last_5m": count_in_recent_window(kills, timestamp),
                "deaths_last_5m": count_in_recent_window(deaths, timestamp),
                "assists_last_5m": count_in_recent_window(assists, timestamp),
                "seconds_since_last_death": (
                    (timestamp - last_death) / 1000.0
                    if not np.isnan(last_death)
                    else np.nan
                ),
            }
        )

        if number % 10_000 == 0 or number == len(observations):
            log(f"Observations featured: {number:,} / {len(observations):,}")

    return pd.DataFrame(rows)


# ============================================================
# MERGE, VALIDATION, AND DIAGNOSTICS
# ============================================================

def attach_features(matched: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    enriched = matched.copy()
    enriched["observation_id"] = (
        enriched["match_id"].astype(str)
        + "_"
        + enriched["participant_id"].astype(int).astype(str)
        + "_"
        + enriched["observation_timestamp"].astype(int).astype(str)
    )

    original_rows = len(enriched)
    enriched = enriched.merge(
        features, on="observation_id", how="left", validate="many_to_one"
    )

    if len(enriched) != original_rows:
        raise ValueError("Feature merge changed row count")

    return enriched


def validate_features(enriched: pd.DataFrame) -> None:
    required = [
        "purchase_time_minutes",
        "dark_seal_purchased_before_observation",
        "kills_last_5m",
        "deaths_last_5m",
        "assists_last_5m",
        "seconds_since_last_death",
    ]
    missing = [column for column in required if column not in enriched.columns]
    if missing:
        raise ValueError(f"Missing constructed features: {missing}")

    complete_features = [
        "purchase_time_minutes",
        "dark_seal_purchased_before_observation",
        "kills_last_5m",
        "deaths_last_5m",
        "assists_last_5m",
    ]
    if enriched[complete_features].isna().any(axis=None):
        raise ValueError("Unexpected missing values in complete compact features")

    for column in [
        "dark_seal_purchased_before_observation",
        "kills_last_5m",
        "deaths_last_5m",
        "assists_last_5m",
    ]:
        numeric = pd.to_numeric(enriched[column], errors="coerce")
        if (numeric < 0).any():
            raise ValueError(f"Negative value found in {column}")


def build_diagnostics(sensitivity: pd.DataFrame) -> pd.DataFrame:
    features = [
        "purchase_time_minutes",
        "dark_seal_purchased_before_observation",
        "kills_last_5m",
        "deaths_last_5m",
        "assists_last_5m",
        "seconds_since_last_death",
    ]

    rows = []
    for feature in features:
        numeric = pd.to_numeric(sensitivity[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "non_missing_count": int(numeric.notna().sum()),
                "non_missing_ratio": float(numeric.notna().mean()),
                "mean": float(numeric.mean()) if numeric.notna().any() else np.nan,
                "std": float(numeric.std()) if numeric.notna().any() else np.nan,
                "minimum": float(numeric.min()) if numeric.notna().any() else np.nan,
                "maximum": float(numeric.max()) if numeric.notna().any() else np.nan,
            }
        )

    return pd.DataFrame(rows)


def print_summary(
    primary: pd.DataFrame, sensitivity: pd.DataFrame, diagnostics: pd.DataFrame
) -> None:
    log("")
    log("=" * 76)
    log("COMPACT PURCHASE DECISION FEATURE SUMMARY")
    log("=" * 76)
    log(f"Primary rows: {len(primary):,}")
    log(f"Sensitivity rows: {len(sensitivity):,}")
    log(f"Derived feature columns: {len(diagnostics):,}")
    log("")
    log("Feature coverage:")

    display = diagnostics[
        ["feature", "non_missing_ratio", "mean", "minimum", "maximum"]
    ].copy()
    display["non_missing_ratio"] = display["non_missing_ratio"].map(
        lambda value: f"{value:.2%}"
    )
    log(display.to_string(index=False))


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    log("=" * 76)
    log("BUILD COMPACT PURCHASE DECISION FEATURES")
    log("=" * 76)

    primary = load_matched_dataset(PRIMARY_INPUT, "primary")
    sensitivity = load_matched_dataset(SENSITIVITY_INPUT, "sensitivity")

    log(f"Primary rows loaded: {len(primary):,}")
    log(f"Sensitivity rows loaded: {len(sensitivity):,}")

    observations = build_observation_table(sensitivity)
    log(f"Unique observations to feature: {len(observations):,}")

    event_files = discover_event_files()
    log(f"Event parquet files found: {len(event_files):,}")

    relevant_match_ids = set(observations["match_id"])
    events = load_relevant_events(event_files, relevant_match_ids)
    event_index = build_event_index(events)
    del events

    features = derive_observation_features(observations, event_index)
    primary_enriched = attach_features(primary, features)
    sensitivity_enriched = attach_features(sensitivity, features)

    validate_features(primary_enriched)
    validate_features(sensitivity_enriched)
    diagnostics = build_diagnostics(sensitivity_enriched)
    print_summary(primary_enriched, sensitivity_enriched, diagnostics)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    primary_enriched.to_parquet(PRIMARY_OUTPUT, index=False)
    sensitivity_enriched.to_parquet(SENSITIVITY_OUTPUT, index=False)
    diagnostics.to_csv(DIAGNOSTICS_OUTPUT, index=False)

    log("")
    log(f"[SAVED] {PRIMARY_OUTPUT}")
    log(f"[SAVED] {SENSITIVITY_OUTPUT}")
    log(f"[SAVED] {DIAGNOSTICS_OUTPUT}")
    log("")
    log("[PASSED] COMPACT PURCHASE DECISION FEATURES CONSTRUCTED")


if __name__ == "__main__":
    main()
