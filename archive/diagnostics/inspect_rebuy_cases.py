'''
Repurchase Case Inspection for Mejai Events
Im taking EUW1_7927742887 as an example.
it seems there are 2 buy event consecutively, without a sale in between.

'''

from pathlib import Path
import json
import pandas as pd


CATALOGUE_FILE = Path("data/analysis/mejai_event_catalogue.json")
OUTPUT_DIR = Path("data/analysis")
OUTPUT_FILE = OUTPUT_DIR / "mejai_repurchase_cases.json"

MEJAI_ITEM_ID = 3041


def log(message):
    print(message)


def load_catalogue():
    if not CATALOGUE_FILE.exists():
        log(f"[ERROR] Catalogue not found: {CATALOGUE_FILE}")
        return pd.DataFrame()

    try:
        with open(CATALOGUE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as e:
        log(f"[ERROR] Could not read catalogue: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def find_repurchase_cases(df):
    if df.empty:
        return []

    required_columns = [
        "match_id",
        "participant_id",
        "timestamp",
        "event_type"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        log(f"[ERROR] Missing columns: {missing}")
        return []

    df = df.copy()

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    df["participant_id"] = pd.to_numeric(
        df["participant_id"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "match_id",
            "participant_id",
            "timestamp"
        ]
    )

    df = df.sort_values(
        ["match_id", "participant_id", "timestamp"]
    )

    cases = []

    grouped = df.groupby(
        ["match_id", "participant_id"],
        sort=False
    )

    for (match_id, participant_id), group in grouped:
        group = group.sort_values("timestamp").reset_index(drop=True)

        purchase_count = 0

        for index, row in group.iterrows():
            if row["event_type"] != "ITEM_PURCHASED":
                continue

            purchase_count += 1

            if purchase_count <= 1:
                continue

            previous_purchase = group.iloc[:index]

            previous_purchase = previous_purchase[
                previous_purchase["event_type"] == "ITEM_PURCHASED"
            ]

            if previous_purchase.empty:
                continue

            previous_purchase = previous_purchase.iloc[-1]

            previous_purchase_timestamp = int(
                previous_purchase["timestamp"]
            )

            current_timestamp = int(row["timestamp"])

            events_since_previous_purchase = group[
                (group["timestamp"] >= previous_purchase_timestamp)
                & (group["timestamp"] <= current_timestamp)
            ]

            if not any(
                events_since_previous_purchase["event_type"] == "ITEM_SOLD"
            ):
                cases.append({
                    "match_id": match_id,
                    "participant_id": int(participant_id),
                    "first_purchase_timestamp": previous_purchase_timestamp,
                    "second_purchase_timestamp": current_timestamp,
                    "gap_seconds": (current_timestamp - previous_purchase_timestamp) / 1000,
                    "events_between": events_since_previous_purchase.to_dict("records")
                })

    return cases


def print_cases(cases):
    log("")
    log("========== REPURCHASE CASES ==========")
    log(f"Cases found: {len(cases):,}")

    if not cases:
        return

    for number, case in enumerate(cases[:50], start=1):
        log("")
        log(f"--- CASE {number} ---")
        log(f"Match: {case['match_id']}")
        log(f"Participant: {case['participant_id']}")
        log(f"First purchase: {case['first_purchase_timestamp']}")
        log(f"Second purchase: {case['second_purchase_timestamp']}")
        log(f"Gap: {case['gap_seconds']:.3f} seconds")
        log("Events:")

        events = case["events_between"]

        for event in events:
            log(
                f"  {event.get('timestamp')} "
                f"{event.get('event_type')} "
                f"item={event.get('item_id')} "
                f"before={event.get('before_item_id')} "
                f"after={event.get('after_item_id')}"
            )


def save_cases(cases):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            cases,
            file,
            indent=2,
            ensure_ascii=False
        )


def main():
    log("===========================================")
    log("       MEJAI REPURCHASE INSPECTION")
    log("===========================================")

    df = load_catalogue()

    if df.empty:
        log("[ERROR] No catalogue data available")
        return

    log(f"Catalogue events: {len(df):,}")

    cases = find_repurchase_cases(df)

    print_cases(cases)

    save_cases(cases)

    log("")
    log(f"Inspection report written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
