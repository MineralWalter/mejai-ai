from pathlib import Path
import json
import pandas as pd

LIFECYCLE_FILE = Path("data/analysis/mejai_purchase_lifecycles.json")
CATALOGUE_FILE = Path("data/analysis/mejai_event_catalogue.json")
OUTPUT_DIR = Path("data/analysis")
OUTPUT_FILE = OUTPUT_DIR / "mejai_lifecycle_validation.json"

MEJAI_ITEM_ID = 3041

def log(message):
    print(message)

def load_json(filepath):
    if not filepath.exists():
        log(f"[ERROR] File not found: {filepath}")
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        log(f"[ERROR] Could not read {filepath}: {e}")
        return None


def prepare_dataframe(data):
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

    if "purchase_timestamp" in df.columns:
        df["purchase_timestamp"] = pd.to_numeric(df["purchase_timestamp"], errors="coerce")

    if "sale_timestamp" in df.columns:
        df["sale_timestamp"] = pd.to_numeric(df["sale_timestamp"], errors="coerce")

    if "undo_timestamp" in df.columns:
        df["undo_timestamp"] = pd.to_numeric(df["undo_timestamp"], errors="coerce")

    if "participant_id" in df.columns:
        df["participant_id"] = pd.to_numeric(df["participant_id"], errors="coerce")

    return df


def build_event_lookup(catalogue):
    lookup = {}

    required_columns = [
        "match_id",
        "participant_id",
        "timestamp",
        "event_type"
    ]

    missing = [
        column
        for column in required_columns
        if column not in catalogue.columns
    ]

    if missing:
        log(f"[ERROR] Catalogue missing columns: {missing}")
        return lookup

    for _, row in catalogue.iterrows():
        key = (
            row["match_id"],
            int(row["participant_id"]),
            int(row["timestamp"]),
            row["event_type"]
        )

        lookup[key] = row.to_dict()

    return lookup


def validate_undone(lifecycles, catalogue, results):
    log("")
    log("========== UNDONE VALIDATION ==========")

    undone = lifecycles[
        lifecycles["status"] == "UNDONE"
    ].copy()

    results["summary"]["undone_episodes"] = len(undone)

    if undone.empty:
        log("No UNDONE episodes found")
        return

    valid = 0
    invalid = 0
    undo_gaps = []

    for _, lifecycle in undone.iterrows():
        match_id = lifecycle["match_id"]
        participant_id = int(lifecycle["participant_id"])
        purchase_timestamp = int(lifecycle["purchase_timestamp"])

        events = catalogue[
            (catalogue["match_id"] == match_id)
            & (catalogue["participant_id"] == participant_id)
            & (catalogue["event_type"] == "ITEM_UNDO")
            & (catalogue["timestamp"] >= purchase_timestamp)
            & (catalogue["before_item_id"] == MEJAI_ITEM_ID)
        ].sort_values("timestamp")

        if events.empty:
            invalid += 1

            results["errors"].append({
                "type": "UNDONE_WITHOUT_UNDO_EVENT",
                "match_id": match_id,
                "participant_id": participant_id,
                "purchase_timestamp": purchase_timestamp
            })

            continue

        undo = events.iloc[0]
        undo_timestamp = int(undo["timestamp"])
        gap = undo_timestamp - purchase_timestamp

        valid += 1
        undo_gaps.append(gap)

    results["summary"]["undone_valid"] = valid
    results["summary"]["undone_invalid"] = invalid

    log(f"UNDONE episodes: {len(undone):,}")
    log(f"Valid undo matches: {valid:,}")
    log(f"Invalid undo matches: {invalid:,}")

    if undo_gaps:
        log(f"Median time to undo: {pd.Series(undo_gaps).median() / 1000:,.1f} seconds")
        log(f"Mean time to undo: {pd.Series(undo_gaps).mean() / 1000:,.1f} seconds")
        log(f"Minimum time to undo: {min(undo_gaps) / 1000:,.1f} seconds")
        log(f"Maximum time to undo: {max(undo_gaps) / 1000:,.1f} seconds")


def validate_sold(lifecycles, catalogue, results):
    log("")
    log("========== SOLD VALIDATION ==========")

    sold = lifecycles[
        lifecycles["status"] == "SOLD"
    ].copy()

    results["summary"]["sold_episodes"] = len(sold)

    if sold.empty:
        log("No SOLD episodes found")
        return

    valid = 0
    invalid = 0

    for _, lifecycle in sold.iterrows():
        match_id = lifecycle["match_id"]
        participant_id = int(lifecycle["participant_id"])
        purchase_timestamp = int(lifecycle["purchase_timestamp"])
        sale_timestamp = int(lifecycle["sale_timestamp"])

        events = catalogue[
            (catalogue["match_id"] == match_id)
            & (catalogue["participant_id"] == participant_id)
            & (catalogue["event_type"] == "ITEM_SOLD")
            & (catalogue["timestamp"] == sale_timestamp)
        ]

        if events.empty:
            invalid += 1

            results["errors"].append({
                "type": "SOLD_WITHOUT_SALE_EVENT",
                "match_id": match_id,
                "participant_id": participant_id,
                "purchase_timestamp": purchase_timestamp,
                "sale_timestamp": sale_timestamp
            })

            continue

        valid += 1

    results["summary"]["sold_valid"] = valid
    results["summary"]["sold_invalid"] = invalid

    log(f"SOLD episodes: {len(sold):,}")
    log(f"Valid sale matches: {valid:,}")
    log(f"Invalid sale matches: {invalid:,}")


def validate_timing(lifecycles, results):
    log("")
    log("========== TIMING VALIDATION ==========")

    invalid = lifecycles[
        (
            lifecycles["status"] == "SOLD"
        )
        & (
            lifecycles["sale_timestamp"]
            <= lifecycles["purchase_timestamp"]
        )
    ]

    results["summary"]["invalid_sale_timing"] = len(invalid)

    if invalid.empty:
        log("[PASS] All SOLD episodes have sale after purchase")
        return

    log(f"[WARNING] {len(invalid):,} SOLD episodes have invalid timing")

    for _, row in invalid.head(20).iterrows():
        results["warnings"].append({
            "type": "INVALID_SALE_TIMING",
            "match_id": row["match_id"],
            "participant_id": int(row["participant_id"]),
            "purchase_timestamp": int(row["purchase_timestamp"]),
            "sale_timestamp": int(row["sale_timestamp"])
        })


def validate_counts(lifecycles, catalogue, results):
    log("")
    log("========== COUNT VALIDATION ==========")

    purchase_count = len(
        catalogue[
            catalogue["event_type"] == "ITEM_PURCHASED"
        ]
    )

    lifecycle_count = len(lifecycles)

    results["summary"]["catalogue_purchase_events"] = purchase_count
    results["summary"]["lifecycle_episodes"] = lifecycle_count

    log(f"Catalogue purchases: {purchase_count:,}")
    log(f"Lifecycle episodes: {lifecycle_count:,}")

    if purchase_count == lifecycle_count:
        log("[PASS] Purchase event count matches lifecycle episode count")
    else:
        log("[WARNING] Purchase event count does not match lifecycle episode count")

        results["warnings"].append({
            "type": "PURCHASE_LIFECYCLE_COUNT_MISMATCH",
            "purchase_events": purchase_count,
            "lifecycle_episodes": lifecycle_count
        })


def inspect_unresolved_sequences(catalogue, results):
    log("")
    log("========== SEQUENCE INSPECTION ==========")

    suspicious = []

    grouped = catalogue.groupby(
        ["match_id", "participant_id"],
        sort=False
    )

    for (match_id, participant_id), group in grouped:
        group = group.sort_values("timestamp")

        mejai_purchases = group[
            group["event_type"] == "ITEM_PURCHASED"
        ]

        if len(mejai_purchases) <= 1:
            continue

        for i in range(1, len(mejai_purchases)):
            previous = mejai_purchases.iloc[i - 1]
            current = mejai_purchases.iloc[i]

            between = group[
                (group["timestamp"] > previous["timestamp"])
                & (group["timestamp"] < current["timestamp"])
            ]

            event_types = between["event_type"].tolist()

            if "ITEM_UNDO" not in event_types:
                suspicious.append({
                    "match_id": match_id,
                    "participant_id": int(participant_id),
                    "first_purchase": int(previous["timestamp"]),
                    "second_purchase": int(current["timestamp"]),
                    "events_between": event_types
                })

    results["summary"]["unresolved_second_purchases"] = len(suspicious)

    log(f"Second-purchase sequences without an intervening undo: {len(suspicious):,}")

    if suspicious:
        results["warnings"].extend(suspicious[:100])

        log("")
        log("First 20 suspicious sequences:")

        for case in suspicious[:20]:
            log(
                f"{case['match_id']} participant={case['participant_id']} "
                f"first={case['first_purchase']} "
                f"second={case['second_purchase']} "
                f"between={case['events_between']}"
            )


def main():
    log("===========================================")
    log("       MEJAI LIFECYCLE VALIDATION")
    log("===========================================")

    lifecycle_data = load_json(LIFECYCLE_FILE)
    catalogue_data = load_json(CATALOGUE_FILE)

    if lifecycle_data is None or catalogue_data is None:
        log("[ERROR] Required input files could not be loaded")
        return

    lifecycles = prepare_dataframe(lifecycle_data)
    catalogue = prepare_dataframe(catalogue_data)

    if lifecycles.empty:
        log("[ERROR] Lifecycle data is empty")
        return

    if catalogue.empty:
        log("[ERROR] Catalogue data is empty")
        return

    log(f"Lifecycle episodes: {len(lifecycles):,}")
    log(f"Catalogue events: {len(catalogue):,}")

    results = {
        "summary": {},
        "errors": [],
        "warnings": []
    }

    validate_counts(lifecycles, catalogue, results)
    validate_undone(lifecycles, catalogue, results)
    validate_sold(lifecycles, catalogue, results)
    validate_timing(lifecycles, results)
    inspect_unresolved_sequences(catalogue, results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    log("")
    log(f"Validation report written to: {OUTPUT_FILE}")
    log("")
    log("===========================================")

    if results["errors"]:
        log(f"[FAILED] {len(results['errors']):,} validation errors found")
    else:
        log("[PASSED] No lifecycle validation errors found")

    log(f"Warnings: {len(results['warnings']):,}")


if __name__ == "__main__":
    main()