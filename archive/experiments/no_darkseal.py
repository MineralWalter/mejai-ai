import pandas as pd

from src.research.config import CASE_DATASET, PARQUET_DIR


ENRICHED_CASES = (
    CASE_DATASET.parent
    / "mejai_research_dataset_event_enriched.parquet"
)

OUTPUT_FILE = (
    CASE_DATASET.parent
    / "dark_seal_zero_case_audit.csv"
)

DARK_SEAL_ITEM_ID = 1082
MEJAI_ITEM_ID = 3041


def log(message=""):
    print(message)


def main():
    if not ENRICHED_CASES.exists():
        raise FileNotFoundError(
            f"Enriched case file not found: {ENRICHED_CASES}"
        )

    cases = pd.read_parquet(ENRICHED_CASES)

    zero_cases = cases[
        cases["dark_seal_purchased_before_observation"] == 0
    ][
        [
            "case_id",
            "match_id",
            "participant_id",
            "purchase_timestamp",
            "lifecycle_status",
            "champion_name",
            "team_position",
        ]
    ].copy()

    log(f"Cases to audit: {len(zero_cases):,}")

    relevant_match_ids = set(
        zero_cases["match_id"].astype(str)
    )

    event_files = sorted(
        (PARQUET_DIR / "events").glob("*_part_*.parquet")
    )

    if not event_files:
        raise FileNotFoundError(
            f"No event files found in {PARQUET_DIR / 'events'}"
        )

    frames = []

    for number, filepath in enumerate(event_files, start=1):
        events = pd.read_parquet(
            filepath,
            columns=[
                "match_id",
                "event_type",
                "timestamp",
                "participant_id",
                "item_id",
            ],
        )

        events["match_id"] = events["match_id"].astype(str)

        events = events[
            events["match_id"].isin(relevant_match_ids)
        ]

        if events.empty:
            continue

        events["event_type"] = (
            events["event_type"]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        events = events[
            events["event_type"] == "ITEM_PURCHASED"
        ]

        events["participant_id"] = pd.to_numeric(
            events["participant_id"],
            errors="coerce",
        )

        events["timestamp"] = pd.to_numeric(
            events["timestamp"],
            errors="coerce",
        )

        events["item_id"] = pd.to_numeric(
            events["item_id"],
            errors="coerce",
        )

        events = events.dropna(
            subset=[
                "participant_id",
                "timestamp",
                "item_id",
            ]
        )

        events["participant_id"] = (
            events["participant_id"].astype(int)
        )

        events["timestamp"] = events["timestamp"].astype(int)
        events["item_id"] = events["item_id"].astype(int)

        events = events[
            events["item_id"].isin(
                [DARK_SEAL_ITEM_ID, MEJAI_ITEM_ID]
            )
        ]

        if not events.empty:
            frames.append(events)

        if number % 50 == 0 or number == len(event_files):
            log(
                f"Event files checked: "
                f"{number:,} / {len(event_files):,}"
            )

    if frames:
        item_events = pd.concat(
            frames,
            ignore_index=True,
        )
    else:
        item_events = pd.DataFrame(
            columns=[
                "match_id",
                "participant_id",
                "timestamp",
                "item_id",
            ]
        )

    rows = []

    for case in zero_cases.itertuples(index=False):
        player_events = item_events[
            (item_events["match_id"] == str(case.match_id))
            & (
                item_events["participant_id"]
                == int(case.participant_id)
            )
        ]

        dark_seal_times = sorted(
            player_events.loc[
                player_events["item_id"] == DARK_SEAL_ITEM_ID,
                "timestamp",
            ].astype(int)
        )

        mejai_times = sorted(
            player_events.loc[
                player_events["item_id"] == MEJAI_ITEM_ID,
                "timestamp",
            ].astype(int)
        )

        purchase_timestamp = int(case.purchase_timestamp)

        dark_seal_before = [
            timestamp
            for timestamp in dark_seal_times
            if timestamp < purchase_timestamp
        ]

        dark_seal_same_time = [
            timestamp
            for timestamp in dark_seal_times
            if timestamp == purchase_timestamp
        ]

        dark_seal_after = [
            timestamp
            for timestamp in dark_seal_times
            if timestamp > purchase_timestamp
        ]

        if dark_seal_before:
            classification = "PRIOR_DARK_SEAL_FOUND"
        elif dark_seal_same_time:
            classification = "DARK_SEAL_SAME_TIMESTAMP"
        elif dark_seal_after:
            classification = "DARK_SEAL_ONLY_AFTER_MEJAI"
        else:
            classification = "NO_DARK_SEAL_PURCHASE_EVENT"

        rows.append(
            {
                "case_id": case.case_id,
                "match_id": case.match_id,
                "participant_id": case.participant_id,
                "purchase_timestamp": purchase_timestamp,
                "lifecycle_status": case.lifecycle_status,
                "champion_name": case.champion_name,
                "team_position": case.team_position,
                "classification": classification,
                "dark_seal_purchase_timestamps": "|".join(
                    str(timestamp)
                    for timestamp in dark_seal_times
                ),
                "mejai_purchase_timestamps": "|".join(
                    str(timestamp)
                    for timestamp in mejai_times
                ),
            }
        )

    audit = pd.DataFrame(rows)

    log("")
    log("Classification:")
    log(
        audit["classification"]
        .value_counts(dropna=False)
        .to_string()
    )

    log("")
    log("Classification by lifecycle status:")
    log(
        pd.crosstab(
            audit["classification"],
            audit["lifecycle_status"],
            dropna=False,
        ).to_string()
    )

    audit.to_csv(OUTPUT_FILE, index=False)

    log("")
    log(f"[SAVED] {OUTPUT_FILE}")
    log("[PASSED] DARK SEAL ZERO-CASE AUDIT COMPLETE")


if __name__ == "__main__":
    main()