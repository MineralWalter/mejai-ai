from pathlib import Path
import pandas as pd


PARQUET_DIR = Path("data/parquet/events")

MATCH_ID = "EUW1_7929051957"
PARTICIPANT_ID = 8

MEJAI_ITEM_ID = 3041


def log(message):
    print(message)


def find_event_file(match_id):
    for filepath in PARQUET_DIR.glob("*.parquet"):
        try:
            df = pd.read_parquet(
                filepath,
                columns=["match_id"]
            )

            if match_id in set(df["match_id"].dropna()):
                return filepath

        except Exception:
            continue

    return None


def load_match_events(filepath, match_id):
    try:
        df = pd.read_parquet(filepath)
    except Exception as e:
        log(f"[ERROR] Could not read {filepath}: {e}")
        return pd.DataFrame()

    if "match_id" not in df.columns:
        return pd.DataFrame()

    return df[df["match_id"] == match_id].copy()


def main():
    log("===========================================")
    log("          MEJAI CASE INSPECTION")
    log("===========================================")

    log(f"Match: {MATCH_ID}")
    log(f"Participant: {PARTICIPANT_ID}")

    filepath = find_event_file(MATCH_ID)

    if filepath is None:
        log("[ERROR] Match event file not found")
        return

    log(f"Event file: {filepath}")

    df = load_match_events(filepath, MATCH_ID)

    if df.empty:
        log("[ERROR] No events found for match")
        return

    log(f"Total match events: {len(df):,}")

    participant_events = df[
        df["participant_id"] == PARTICIPANT_ID
    ].copy()

    if participant_events.empty:
        log("[ERROR] No participant events found")
        return

    participant_events = participant_events.sort_values(
        "timestamp"
    )

    log(f"Participant events: {len(participant_events):,}")

    log("")
    log("========== FULL PARTICIPANT TIMELINE ==========")

    display_columns = [
        column
        for column in [
            "timestamp",
            "event_type",
            "item_id",
            "before_item_id",
            "after_item_id",
            "participant_id",
            "killer_id",
            "victim_id",
            "assisting_ids",
            "monster_type",
            "monster_sub_type",
            "building_type",
            "lane_type",
            "team_id"
        ]
        if column in participant_events.columns
    ]

    print(
        participant_events[display_columns].to_string(
            index=False
        )
    )

    mejai_events = participant_events[
        (
            participant_events["item_id"] == MEJAI_ITEM_ID
        )
        |
        (
            participant_events["before_item_id"] == MEJAI_ITEM_ID
        )
        |
        (
            participant_events["after_item_id"] == MEJAI_ITEM_ID
        )
    ].copy()

    log("")
    log("========== MEJAI EVENTS ==========")

    if mejai_events.empty:
        log("No Mejai events found")
    else:
        print(
            mejai_events[display_columns].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()