'''
Discovered 429 really weird case of purchased again be4 sale, need to look into it more ( SOLVED)
'''
from pathlib import Path
import json
import pandas as pd


INPUT_FILE = Path("data/analysis/mejai_event_catalogue.json")
OUTPUT_DIR = Path("data/analysis")
OUTPUT_FILE = OUTPUT_DIR / "mejai_purchase_lifecycles.json"


def log(message):
    print(message)


def load_catalogue():
    if not INPUT_FILE.exists():
        log(f"[ERROR] Catalogue not found: {INPUT_FILE}")
        return pd.DataFrame()

    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as e:
        log(f"[ERROR] Could not read catalogue: {e}")
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def prepare_events(df):
    required_columns = [
        "match_id",
        "participant_id",
        "timestamp",
        "event_type"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        log(f"[ERROR] Missing columns: {missing_columns}")
        return pd.DataFrame()

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
            "timestamp",
            "event_type"
        ]
    )

    df = df.sort_values(
        ["match_id", "participant_id", "timestamp"]
    ).reset_index(drop=True)

    return df

def reconstruct_lifecycles(df):
    lifecycles = []

    grouped = df.groupby(["match_id", "participant_id"], sort=False)

    for (match_id, participant_id), group in grouped:
        group = group.sort_values("timestamp")

        open_purchase = None

        for _, event in group.iterrows():
            event_type = event["event_type"]
            timestamp = int(event["timestamp"])

            if event_type == "ITEM_PURCHASED":
                if open_purchase is not None:
                    lifecycles.append({
                        "match_id": match_id,
                        "participant_id": int(participant_id),
                        "purchase_timestamp": open_purchase["timestamp"],
                        "sale_timestamp": None,
                        "time_held_ms": None,
                        "status": "PURCHASED_AGAIN_BEFORE_SALE"
                    })

                open_purchase = {
                    "timestamp": timestamp
                }

            elif event_type == "ITEM_UNDO":
                if open_purchase is not None:
                    before_item_id = event.get("before_item_id")

                    if pd.notna(before_item_id) and int(before_item_id) == 3041:
                        lifecycles.append({
                            "match_id": match_id,
                            "participant_id": int(participant_id),
                            "purchase_timestamp": open_purchase["timestamp"],
                            "sale_timestamp": None,
                            "time_held_ms": None,
                            "status": "UNDONE"
                        })

                        open_purchase = None

            elif event_type == "ITEM_SOLD":
                if open_purchase is not None:
                    time_held = timestamp - open_purchase["timestamp"]

                    lifecycles.append({
                        "match_id": match_id,
                        "participant_id": int(participant_id),
                        "purchase_timestamp": open_purchase["timestamp"],
                        "sale_timestamp": timestamp,
                        "time_held_ms": time_held,
                        "status": "SOLD"
                    })

                    open_purchase = None

        if open_purchase is not None:
            lifecycles.append({
                "match_id": match_id,
                "participant_id": int(participant_id),
                "purchase_timestamp": open_purchase["timestamp"],
                "sale_timestamp": None,
                "time_held_ms": None,
                "status": "RETAINED"
            })

    return lifecycles

def save_lifecycles(lifecycles):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            lifecycles,
            file,
            indent=2,
            ensure_ascii=False
        )


def print_summary(lifecycles):
    if not lifecycles:
        log("[WARNING] No lifecycles reconstructed")
        return

    df = pd.DataFrame(lifecycles)

    log("")
    log("========== LIFECYCLE SUMMARY ==========")
    log(f"Total purchase episodes: {len(df):,}")

    status_counts = df["status"].value_counts()

    for status, count in status_counts.items():
        log(f"{status}: {count:,}")

    sold = df[
        df["status"] == "SOLD"
    ].copy()

    if not sold.empty:
        log("")
        log("========== SOLD PURCHASES ==========")
        log(f"Sold episodes: {len(sold):,}")
        log(f"Median time held: {sold['time_held_ms'].median() / 1000:,.1f} seconds")
        log(f"Mean time held: {sold['time_held_ms'].mean() / 1000:,.1f} seconds")
        log(f"Minimum time held: {sold['time_held_ms'].min() / 1000:,.1f} seconds")
        log(f"Maximum time held: {sold['time_held_ms'].max() / 1000:,.1f} seconds")

    log("")
    log("========== EXAMPLE LIFECYCLES ==========")

    display_columns = [
        "match_id",
        "participant_id",
        "purchase_timestamp",
        "sale_timestamp",
        "time_held_ms",
        "status"
    ]

    log(
        df[display_columns]
        .head(20)
        .to_string(index=False)
    )


def main():
    log("===========================================")
    log("       MEJAI PURCHASE RECONSTRUCTION")
    log("===========================================")

    df = load_catalogue()

    if df.empty:
        log("[ERROR] No Mejai catalogue data found")
        return

    log(f"Catalogue events: {len(df):,}")

    df = prepare_events(df)

    if df.empty:
        log("[ERROR] No usable events after preparation")
        return

    log(f"Usable events: {len(df):,}")

    lifecycles = reconstruct_lifecycles(df)

    log(f"Purchase episodes reconstructed: {len(lifecycles):,}")

    print_summary(lifecycles)

    save_lifecycles(lifecycles)

    log("")
    log(f"Lifecycle catalogue written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
