from pathlib import Path
import json
import pandas as pd

PARQUET_DIR = Path("data/parquet")
OUTPUT_DIR = Path("data/analysis")
OUTPUT_FILE = OUTPUT_DIR / "mejai_event_catalogue.json"

LANES = ["sea", "asia", "europe", "americas"]


def log(message):
    print(message)


def find_event_files():
    files = []

    for lane in LANES:
        lane_files = sorted((PARQUET_DIR / "events").glob(f"{lane}_part_*.parquet"))
        files.extend(lane_files)

    return files


def inspect_event_types(files):
    counts = {}

    for filepath in files:
        try:
            df = pd.read_parquet(filepath, columns=["event_type"])
        except Exception as e:
            log(f"[ERROR] Could not read {filepath}: {e}")
            continue

        for event_type, count in df["event_type"].value_counts(dropna=False).items():
            key = str(event_type)
            counts[key] = counts.get(key, 0) + int(count)

    return dict(sorted(counts.items(), key=lambda x: x[0]))

def inspect_item_events(files):
    item_events = []

    for filepath in files:
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            log(f"[ERROR] Could not read {filepath}: {e}")
            continue

        required_columns = ["item_id","before_item_id","after_item_id"]

        available_columns = [column for column in required_columns if column in df.columns]

        if not available_columns:
            continue

        filtered = df.copy()

        filtered["source_file"] = str(filepath)
        item_events.append(filtered)

    if not item_events:
        return pd.DataFrame()

    return pd.concat(item_events, ignore_index=True)

MEJAI_ITEM_ID = 3041 # Id like to change later if it turns out wrong

def find_mejai_candidates(df):
    if df.empty:
        return pd.DataFrame()

    candidate_mask = (
        (df["item_id"] == MEJAI_ITEM_ID)
        | (df["before_item_id"] == MEJAI_ITEM_ID)
        | (df["after_item_id"] == MEJAI_ITEM_ID)
    )

    return df[candidate_mask].copy()


def build_catalogue(df):
    if df.empty:
        return []

    catalogue = []

    for _, row in df.iterrows():
        record = {}

        for column in df.columns:
            value = row[column]

            if pd.isna(value):
                record[column] = None
            elif hasattr(value, "tolist"):
                record[column] = value.tolist()
            else:
                record[column] = value.item() if hasattr(value, "item") else value

        catalogue.append(record)

    return catalogue


def main():
    log("===========================================")
    log("         MEJAI EVENT DISCOVERY")
    log("===========================================")

    files = find_event_files()

    log(f"Event files found: {len(files)}")

    if not files:
        log("[ERROR] No event Parquet files found")
        return

    log("")
    log("========== EVENT TYPES ==========")

    event_type_counts = inspect_event_types(files)

    for event_type, count in event_type_counts.items():
        log(f"{event_type}: {count:,}")

    log("")
    log("========== SEARCHING FOR MEJAI ==========")

    item_events = inspect_item_events(files)

    log(f"Item-related rows found: {len(item_events):,}")

    mejai_events = find_mejai_candidates(item_events)

    log(f"Mejai candidate events: {len(mejai_events):,}")

    if mejai_events.empty:
        log("")
        log("[WARNING] No Mejai candidates found.")
        log("The item ID is probably numeric and needs to be identified from Riot data.")
        return

    log("")
    log("========== MEJAI CANDIDATES ==========")

    display_columns = [
        column
        for column in [
            "match_id",
            "timestamp",
            "event_type",
            "participant_id",
            "item_id",
            "before_item_id",
            "after_item_id",
            "source_file"
        ]
        if column in mejai_events.columns
    ]

    log(mejai_events[display_columns].to_string(index=False))

    catalogue = build_catalogue(mejai_events)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(catalogue, file, indent=2, ensure_ascii=False)

    log("")
    log(f"Catalogue written to: {OUTPUT_FILE}")
    log(f"Total candidate events: {len(catalogue):,}")
    
    log("========== MEJAI EVENT TYPES ==========")

    event_type_counts = mejai_events["event_type"].value_counts(dropna=False)

    for event_type, count in event_type_counts.items():
        log(f"{event_type}: {count:,}")

if __name__ == "__main__":
    main()
