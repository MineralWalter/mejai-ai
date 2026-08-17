import ast
import json
from collections import defaultdict
import numpy as np
import pandas as pd
from src.research.config import CASE_DATASET, PARQUET_DIR, V2_CASE_ENRICHED, V2_CONTROL_ENRICHED, V2_CONTROL_POOL
RECENT_WINDOW_MS = 5 * 60 * 1000
DARK_SEAL_ITEM_ID = 1082
LOG_EVERY_FILES = 25
FEATURE_COLUMNS = [
    "dark_seal_purchased_before_observation",
    "kills_last_5m",
    "deaths_last_5m",
    "assists_last_5m",
]

def log(message=''):
    print(message)

def load_inputs():
    if not CASE_DATASET.exists():
        raise FileNotFoundError(f'Case dataset not found: {CASE_DATASET}')

    if not V2_CONTROL_POOL.exists():
        raise FileNotFoundError(f'Version 2 control pool not found: {V2_CONTROL_POOL}')

    cases = pd.read_parquet(CASE_DATASET, engine='pyarrow')
    controls = pd.read_parquet(V2_CONTROL_POOL, engine='pyarrow')

    required_case_columns = [
        "case_id",
        "match_id",
        "participant_id",
        "purchase_timestamp",
    ]

    required_control_columns = [
        "case_id",
        "control_match_id",
        "control_participant_id",
        "control_snapshot_timestamp",
    ]

    missing_case_columns = [column for column in required_case_columns if column not in cases.columns]
    missing_control_columns = [column for column in required_control_columns if column not in controls.columns]

    if missing_case_columns:
        raise ValueError(f'Case dataset is missing columns: {missing_case_columns}')
    if missing_control_columns:
        raise ValueError(f'Control pool is missing columns: {missing_control_columns}')

    cases = cases.copy()
    controls = controls.copy()

    cases['case_id'] = cases['case_id'].astype(str)
    cases['match_id'] = cases['match_id'].astype(str)
    cases['participant_id'] = pd.to_numeric(cases['participant_id'], errors='coerce')
    cases['purchase_timestamp'] = pd.to_numeric(cases['purchase_timestamp'], errors='coerce')

    controls['case_id'] = controls['case_id'].astype(str)
    controls['control_match_id'] = controls['control_match_id'].astype(str)
    controls['control_participant_id'] = pd.to_numeric(controls['control_participant_id'], errors='coerce')
    controls['control_snapshot_timestamp'] = pd.to_numeric(controls['control_snapshot_timestamp'], errors='coerce')

    if cases[['participant_id', 'purchase_timestamp']].isna().any(axis=None):
        raise ValueError('Case dataset contains invalid participant IDs or purchase timestamps')

    if controls[
        ["control_participant_id", "control_snapshot_timestamp"]
    ].isna().any(axis=None):
        raise ValueError('Control pool contains invalid participant IDs or observation timestamps')

    cases['participant_id'] = cases['participant_id'].astype(int)
    cases['purchase_timestamp'] = cases['purchase_timestamp'].astype(int)

    controls['control_participant_id'] = controls['control_participant_id'].astype(int)
    controls['control_snapshot_timestamp'] = controls['control_snapshot_timestamp'].astype(int)
    return (cases, controls)

def make_observation_id(match_ids, participant_ids, timestamps):
    return (
        match_ids.astype(str)
        + "_"
        + participant_ids.astype(int).astype(str)
        + "_"
        + timestamps.astype(int).astype(str)
    )

def build_observations(cases, controls):
    case_observations = cases[
        ["match_id", "participant_id", "purchase_timestamp"]
    ].copy()
    case_observations = case_observations.rename(columns={'purchase_timestamp': 'observation_timestamp'})
    control_observations = controls[
        [
            "control_match_id",
            "control_participant_id",
            "control_snapshot_timestamp",
        ]
    ].copy()

    control_observations = control_observations.rename(
        columns={
            "control_match_id": "match_id",
            "control_participant_id": "participant_id",
            "control_snapshot_timestamp": "observation_timestamp",
        }
    )
    observations = pd.concat(
        [case_observations, control_observations],
        ignore_index=True,
    )

    observations["observation_id"] = make_observation_id(
        observations["match_id"],
        observations["participant_id"],
        observations["observation_timestamp"],
    )

    observations = observations.drop_duplicates(subset=['observation_id']).reset_index(drop=True)
    return observations

def find_event_files():
    event_directory = PARQUET_DIR / 'events'
    if not event_directory.exists():
        raise FileNotFoundError(f'Event directory not found: {event_directory}')
    files = sorted(event_directory.glob('*_part_*.parquet'))
    if not files:
        raise FileNotFoundError(f'No event Parquet files found in {event_directory}')
    return files

def parse_assist_ids(value):
    if value is None:
        return []

    if isinstance(value, (list, tuple, set, np.ndarray, pd.Series)):
        raw_values = list(value)

    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {
            "none",
            "nan",
            "null",
            "[]",
        }:
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
    participant_ids = []
    for raw_value in raw_values:
        try:
            participant_id = int(float(raw_value))
        except (TypeError, ValueError):
            continue
        if participant_id > 0:
            participant_ids.append(participant_id)
    return participant_ids

def add_timestamp(mapping, key, timestamp):
    mapping[key].append(int(timestamp))

def build_event_index(event_files, relevant_match_ids):
    kills = defaultdict(list)
    deaths = defaultdict(list)
    assists = defaultdict(list)
    dark_seal_purchases = defaultdict(list)
    relevant_match_ids = {
        str(match_id)
        for match_id in relevant_match_ids
    }
    columns = [
        "match_id",
        "event_type",
        "timestamp",
        "participant_id",
        "killer_id",
        "victim_id",
        "assisting_ids",
        "item_id",
    ]

    retained_rows = 0
    for file_number, filepath in enumerate(event_files, start=1):
        try:
            events = pd.read_parquet(filepath, columns=columns, engine='pyarrow')
        except Exception as error:
            raise RuntimeError(f'Could not read event file {filepath}: {error}') from error
        events['match_id'] = events['match_id'].astype(str)
        events = events[
            events["match_id"].isin(relevant_match_ids)
        ]

        if events.empty:
            continue

        events["event_type"] = (
            events["event_type"].astype(str).str.strip().str.upper()
        )

        events['timestamp'] = pd.to_numeric(events['timestamp'], errors='coerce')
        events = events.dropna(
            subset=["match_id", "event_type", "timestamp"]
        )

        events['timestamp'] = events['timestamp'].astype(int)
        retained_rows += len(events)
        champion_kills = events[
            events["event_type"] == "CHAMPION_KILL"
        ]
        for event in champion_kills.itertuples(index=False):
            match_id = str(event.match_id)
            timestamp = int(event.timestamp)
            if pd.notna(event.killer_id) and int(event.killer_id) > 0:
                add_timestamp(kills, (match_id, int(event.killer_id)), timestamp)

            if pd.notna(event.victim_id) and int(event.victim_id) > 0:
                add_timestamp(deaths, (match_id, int(event.victim_id)), timestamp)

            for assist_id in parse_assist_ids(event.assisting_ids):
                add_timestamp(assists, (match_id, assist_id), timestamp)
        purchases = events[
            events["event_type"] == "ITEM_PURCHASED"
        ]

        for event in purchases.itertuples(index=False):
            if pd.isna(event.participant_id) or pd.isna(event.item_id):
                continue
            participant_id = int(event.participant_id)
            if participant_id <= 0 or int(event.item_id) != DARK_SEAL_ITEM_ID:
                continue
            add_timestamp(dark_seal_purchases, (str(event.match_id), participant_id), int(event.timestamp))
        if file_number % LOG_EVERY_FILES == 0 or file_number == len(event_files):
            log(
                f"Event files processed: {file_number:,} / {len(event_files):,} | "
                f"retained rows: {retained_rows:,}"
            )
    event_index = {'kills': kills, 'deaths': deaths, 'assists': assists, 'dark_seal_purchases': dark_seal_purchases}
    for mapping in event_index.values():
        for key, timestamps in mapping.items():
            mapping[key] = np.sort(
                np.asarray(timestamps, dtype=np.int64)
            )
    return event_index

def get_timestamps(mapping, key):
    return mapping.get(
        key,
        np.empty(0, dtype=np.int64),
    )

def count_before(timestamps, observation_timestamp):
    return int(
        np.searchsorted(
            timestamps,
            observation_timestamp,
            side="left",
        )
    )

def count_in_recent_window(timestamps, observation_timestamp):
    window_start = observation_timestamp - RECENT_WINDOW_MS

    left = np.searchsorted(
        timestamps,
        window_start,
        side="left",
    )

    right = np.searchsorted(
        timestamps,
        observation_timestamp,
        side="left",
    )
    return int(right - left)

def build_features(observations, event_index):
    rows = []
    for number, observation in enumerate(
        observations.itertuples(index=False),
        start=1,
    ):
        match_id = str(observation.match_id)
        participant_id = int(observation.participant_id)
        observation_timestamp = int(observation.observation_timestamp)
        key = (match_id, participant_id)
        kills = get_timestamps(event_index['kills'], key)
        deaths = get_timestamps(event_index['deaths'], key)
        assists = get_timestamps(event_index['assists'], key)

        dark_seal_purchases = get_timestamps(
            event_index["dark_seal_purchases"],
            key,
        )

        rows.append(
            {
                "observation_id": observation.observation_id,
                "dark_seal_purchased_before_observation": int(
                    count_before(dark_seal_purchases, observation_timestamp) > 0),

                "kills_last_5m": count_in_recent_window(kills,observation_timestamp,),
                "deaths_last_5m": count_in_recent_window(deaths,observation_timestamp,),
                "assists_last_5m": count_in_recent_window(assists,observation_timestamp,),
            }
        )
        if number % 10000 == 0 or number == len(observations):
            log(
                f"Observations featured: {number:,} / {len(observations):,}"
            )
    return pd.DataFrame(rows)

def enrich_cases(cases, features):
    output = cases.copy()
    output["observation_id"] = make_observation_id(
        output["match_id"],
        output["participant_id"],
        output["purchase_timestamp"],
    )

    original_rows = len(output)
    output = output.merge(features, on='observation_id', how='left', validate='many_to_one')

    if len(output) != original_rows:
        raise ValueError('Case feature merge changed row count')
    return output.drop(columns=['observation_id'])

def enrich_controls(controls, enriched_cases, features):
    output = controls.copy()
    mejai_features = enriched_cases[
        ["case_id", *FEATURE_COLUMNS]
    ].copy()
    mejai_features = mejai_features.rename(columns={column: f'mejai_{column}' for column in FEATURE_COLUMNS})
    original_rows = len(output)
    output = output.merge(mejai_features, on='case_id', how='left', validate='many_to_one')

    if len(output) != original_rows:
        raise ValueError('Case-side feature merge changed control-pool row count')
    output["observation_id"] = make_observation_id(
        output["control_match_id"],
        output["control_participant_id"],
        output["control_snapshot_timestamp"],
    )

    control_features = features.rename(columns={column: f'control_{column}' for column in FEATURE_COLUMNS})
    output = output.merge(control_features, on='observation_id', how='left', validate='many_to_one')

    if len(output) != original_rows:
        raise ValueError('Control-side feature merge changed control-pool row count')
    return output.drop(columns=['observation_id'])

def validate_outputs(original_cases, original_controls, enriched_cases, enriched_controls):

    if len(enriched_cases) != len(original_cases):
        raise ValueError('Enriched case row count changed')

    if len(enriched_controls) != len(original_controls):
        raise ValueError('Enriched control row count changed')
    case_features = FEATURE_COLUMNS

    control_features = [
        f"control_{column}"
        for column in FEATURE_COLUMNS
    ]

    mejai_features = [
        f"mejai_{column}"
        for column in FEATURE_COLUMNS
    ]
    if enriched_cases[case_features].isna().any(axis=None):
        raise ValueError('Missing event features in enriched cases')

    if enriched_controls[control_features].isna().any(axis=None):
        raise ValueError('Missing control event features in enriched control pool')

    if enriched_controls[mejai_features].isna().any(axis=None):
        raise ValueError('Missing Mejai-side event features in enriched control pool')

    for column in FEATURE_COLUMNS:
        if (enriched_cases[column] < 0).any():
            raise ValueError(f'Negative value found in {column}')

    for prefix in ['mejai_', 'control_']:
        for column in FEATURE_COLUMNS:
            full_column = prefix + column
            if (enriched_controls[full_column] < 0).any():
                raise ValueError(f'Negative value found in {full_column}')

    for dataframe, column in [
        (enriched_cases, "dark_seal_purchased_before_observation"),
        (enriched_controls, "mejai_dark_seal_purchased_before_observation"),
        (enriched_controls, "control_dark_seal_purchased_before_observation"),
    ]:
        if not dataframe[column].isin([0, 1]).all():
            raise ValueError(f'{column} contains values other than 0 or 1')

def print_summary(enriched_cases, enriched_controls):
    log('')
    log('=' * 70)
    log('PURCHASE DECISION FEATURE SUMMARY')
    log('=' * 70)

    log(f'Enriched case rows: {len(enriched_cases):,}')
    log(f'Enriched control rows: {len(enriched_controls):,}')

    log('')
    log('Case feature averages:')
    log(
        enriched_cases[FEATURE_COLUMNS].mean().to_string()
    )

    log('')
    log('Control feature averages:')
    log(
        enriched_controls[
            [f"control_{column}" for column in FEATURE_COLUMNS]
        ].mean().to_string()
    )

def save_outputs(enriched_cases, enriched_controls):
    V2_CASE_ENRICHED.parent.mkdir(parents=True, exist_ok=True)
    enriched_cases.to_parquet(V2_CASE_ENRICHED, index=False, engine='pyarrow')
    enriched_controls.to_parquet(V2_CONTROL_ENRICHED, index=False, engine='pyarrow')

    log('')
    log(f'[SAVED] {V2_CASE_ENRICHED}')
    log(f'[SAVED] {V2_CONTROL_ENRICHED}')

def main():
    log('=' * 70)
    log('BUILD PURCHASE DECISION FEATURES')
    log('=' * 70)

    cases, controls = load_inputs()
    log(f'Case rows loaded: {len(cases):,}')
    log(f'Control candidate rows loaded: {len(controls):,}')

    observations = build_observations(cases, controls)
    log(f'Unique observations to feature: {len(observations):,}')

    event_files = find_event_files()
    log(f'Event files found: {len(event_files):,}')

    relevant_match_ids = set(observations['match_id'])
    event_index = build_event_index(event_files, relevant_match_ids)
    features = build_features(observations, event_index)
    enriched_cases = enrich_cases(cases, features)
    enriched_controls = enrich_controls(controls, enriched_cases, features)

    validate_outputs(cases, controls, enriched_cases, enriched_controls)
    print_summary(enriched_cases, enriched_controls)
    save_outputs(enriched_cases, enriched_controls)

    log('')
    log('[PASSED] PURCHASE DECISION FEATURES CONSTRUCTED')
if __name__ == '__main__':
    main()