import json
from pathlib import Path

import pandas as pd

from src.research.config import (
    LIFECYCLE_FILE,
    MEJAI_EVENT_CATALOGUE,
    MEJAI_ITEM_ID,
)
from src.research.utils import get_valid_match_ids


VALID_STATUSES = {
    "RETAINED",
    "SOLD",
    "UNDONE",
    "PURCHASED_AGAIN_BEFORE_SALE",
}


# ============================================================
# LOGGING
# ============================================================

def log(message=""):
    print(message)


# ============================================================
# INPUT
# ============================================================

def load_catalogue():
    if not MEJAI_EVENT_CATALOGUE.exists():
        raise FileNotFoundError(
            f"Mejai event catalogue not found: "
            f"{MEJAI_EVENT_CATALOGUE}\n"
            "Run:\n"
            "py -m src.analysis.mejai_events"
        )

    try:
        with open(
            MEJAI_EVENT_CATALOGUE,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except Exception as error:
        raise RuntimeError(
            f"Could not read Mejai event catalogue: {error}"
        ) from error

    if not data:
        return pd.DataFrame()

    return pd.DataFrame(data)


def prepare_events(df):
    required_columns = [
        "match_id",
        "participant_id",
        "timestamp",
        "event_type",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Catalogue is missing required columns: "
            f"{missing_columns}"
        )

    df = df.copy()

    df["match_id"] = (
        df["match_id"]
        .astype(str)
    )

    df["participant_id"] = pd.to_numeric(
        df["participant_id"],
        errors="coerce",
    )

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce",
    )

    df["event_type"] = (
        df["event_type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    for column in [
        "item_id",
        "before_item_id",
        "after_item_id",
    ]:
        if column not in df.columns:
            df[column] = pd.NA

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "match_id",
            "participant_id",
            "timestamp",
            "event_type",
        ]
    )

    df["participant_id"] = (
        df["participant_id"]
        .astype(int)
    )

    df["timestamp"] = (
        df["timestamp"]
        .astype(int)
    )

    valid_match_ids = get_valid_match_ids()

    invalid_match_ids = (
        set(df["match_id"])
        - set(valid_match_ids)
    )

    if invalid_match_ids:
        raise ValueError(
            "The Mejai catalogue contains invalid matches:\n"
            + "\n".join(
                sorted(invalid_match_ids)[:30]
            )
        )

    df = df.sort_values(
        [
            "match_id",
            "participant_id",
            "timestamp",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return df


# ============================================================
# LIFECYCLE RECONSTRUCTION
# ============================================================

def reconstruct_lifecycles(df):
    lifecycles = []

    grouped = df.groupby(
        [
            "match_id",
            "participant_id",
        ],
        sort=False,
        observed=True,
    )

    for (
        match_id,
        participant_id,
    ), group in grouped:
        group = group.sort_values(
            "timestamp",
            kind="stable",
        )

        open_purchase = None

        for event in group.itertuples(
            index=False
        ):
            event_type = event.event_type
            timestamp = int(event.timestamp)

            if event_type == "ITEM_PURCHASED":
                if open_purchase is not None:
                    lifecycles.append(
                        {
                            "match_id": str(match_id),
                            "participant_id": int(
                                participant_id
                            ),
                            "purchase_timestamp": int(
                                open_purchase["timestamp"]
                            ),
                            "sale_timestamp": None,
                            "time_held_ms": None,
                            "status": (
                                "PURCHASED_AGAIN_BEFORE_SALE"
                            ),
                        }
                    )

                open_purchase = {
                    "timestamp": timestamp,
                }

            elif event_type == "ITEM_UNDO":
                if open_purchase is None:
                    continue

                before_item_id = getattr(
                    event,
                    "before_item_id",
                    None,
                )

                if (
                    pd.notna(before_item_id)
                    and int(before_item_id)
                    == MEJAI_ITEM_ID
                ):
                    lifecycles.append(
                        {
                            "match_id": str(match_id),
                            "participant_id": int(
                                participant_id
                            ),
                            "purchase_timestamp": int(
                                open_purchase["timestamp"]
                            ),
                            "sale_timestamp": None,
                            "time_held_ms": None,
                            "status": "UNDONE",
                        }
                    )

                    open_purchase = None

            elif event_type == "ITEM_SOLD":
                if open_purchase is None:
                    continue

                item_id = getattr(
                    event,
                    "item_id",
                    None,
                )

                if (
                    pd.isna(item_id)
                    or int(item_id)
                    != MEJAI_ITEM_ID
                ):
                    continue

                time_held_ms = (
                    timestamp
                    - int(open_purchase["timestamp"])
                )

                lifecycles.append(
                    {
                        "match_id": str(match_id),
                        "participant_id": int(
                            participant_id
                        ),
                        "purchase_timestamp": int(
                            open_purchase["timestamp"]
                        ),
                        "sale_timestamp": timestamp,
                        "time_held_ms": time_held_ms,
                        "status": "SOLD",
                    }
                )

                open_purchase = None

        if open_purchase is not None:
            lifecycles.append(
                {
                    "match_id": str(match_id),
                    "participant_id": int(
                        participant_id
                    ),
                    "purchase_timestamp": int(
                        open_purchase["timestamp"]
                    ),
                    "sale_timestamp": None,
                    "time_held_ms": None,
                    "status": "RETAINED",
                }
            )

    return lifecycles


# ============================================================
# VALIDATION
# ============================================================

def validate_lifecycles(
    lifecycles,
    events,
):
    if not lifecycles:
        raise ValueError(
            "No purchase lifecycles were reconstructed"
        )

    lifecycle_df = pd.DataFrame(
        lifecycles
    )

    required_columns = {
        "match_id",
        "participant_id",
        "purchase_timestamp",
        "sale_timestamp",
        "time_held_ms",
        "status",
    }

    missing_columns = sorted(
        required_columns
        - set(lifecycle_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Lifecycle output is missing columns: "
            f"{missing_columns}"
        )

    invalid_statuses = sorted(
        set(lifecycle_df["status"])
        - VALID_STATUSES
    )

    if invalid_statuses:
        raise ValueError(
            f"Unknown lifecycle statuses: "
            f"{invalid_statuses}"
        )

    invalid_participants = lifecycle_df[
        ~lifecycle_df["participant_id"].between(
            1,
            10,
        )
    ]

    if not invalid_participants.empty:
        raise ValueError(
            "Lifecycle output contains participant IDs "
            "outside 1–10"
        )

    negative_purchase_times = lifecycle_df[
        "purchase_timestamp"
    ].lt(0)

    if negative_purchase_times.any():
        raise ValueError(
            "Lifecycle output contains negative purchase "
            "timestamps"
        )

    sold = lifecycle_df[
        lifecycle_df["status"] == "SOLD"
    ].copy()

    invalid_sales = sold[
        sold["sale_timestamp"].isna()
        | sold["time_held_ms"].isna()
        | sold["time_held_ms"].lt(0)
        | sold["sale_timestamp"].lt(
            sold["purchase_timestamp"]
        )
    ]

    if not invalid_sales.empty:
        raise ValueError(
            "Invalid SOLD lifecycle rows found:\n"
            + invalid_sales.head(20).to_string(
                index=False
            )
        )

    non_sold = lifecycle_df[
        lifecycle_df["status"] != "SOLD"
    ]

    if non_sold["sale_timestamp"].notna().any():
        raise ValueError(
            "Non-SOLD lifecycle rows contain sale timestamps"
        )

    duplicate_cases = lifecycle_df.duplicated(
        subset=[
            "match_id",
            "participant_id",
            "purchase_timestamp",
        ],
        keep=False,
    )

    if duplicate_cases.any():
        examples = lifecycle_df.loc[
            duplicate_cases,
            [
                "match_id",
                "participant_id",
                "purchase_timestamp",
                "status",
            ],
        ].head(30)

        raise ValueError(
            "Duplicate purchase episodes found:\n"
            + examples.to_string(
                index=False
            )
        )

    valid_match_ids = get_valid_match_ids()

    invalid_lifecycle_matches = (
        set(lifecycle_df["match_id"])
        - set(valid_match_ids)
    )

    if invalid_lifecycle_matches:
        raise ValueError(
            "Lifecycle output contains invalid matches:\n"
            + "\n".join(
                sorted(
                    invalid_lifecycle_matches
                )[:30]
            )
        )

    purchase_event_count = int(
        events["event_type"]
        .eq("ITEM_PURCHASED")
        .sum()
    )

    if len(lifecycle_df) != purchase_event_count:
        raise ValueError(
            "Lifecycle count does not equal the number of "
            "Mejai purchase events:\n"
            f"Purchase events: {purchase_event_count:,}\n"
            f"Lifecycle rows:  {len(lifecycle_df):,}"
        )

    return lifecycle_df


# ============================================================
# OUTPUT
# ============================================================

def save_lifecycles(lifecycles):
    LIFECYCLE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = Path(
        str(LIFECYCLE_FILE) + ".tmp"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            lifecycles,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_path.replace(
        LIFECYCLE_FILE
    )


# ============================================================
# REPORTING
# ============================================================

def print_summary(lifecycle_df):
    log("")
    log("=" * 70)
    log("LIFECYCLE SUMMARY")
    log("=" * 70)

    log(
        f"Total purchase episodes: "
        f"{len(lifecycle_df):,}"
    )

    log(
        f"Unique matches: "
        f"{lifecycle_df['match_id'].nunique():,}"
    )

    unique_players = lifecycle_df[
        [
            "match_id",
            "participant_id",
        ]
    ].drop_duplicates()

    log(
        f"Unique player-match combinations: "
        f"{len(unique_players):,}"
    )

    log("")
    log("Status counts:")

    log(
        lifecycle_df[
            "status"
        ].value_counts().to_string()
    )

    sold = lifecycle_df[
        lifecycle_df["status"] == "SOLD"
    ]

    if not sold.empty:
        log("")
        log("Sold-item holding time:")

        log(
            sold["time_held_ms"]
            .describe()
            .to_string()
        )

    log("")
    log("Example lifecycles:")

    log(
        lifecycle_df[
            [
                "match_id",
                "participant_id",
                "purchase_timestamp",
                "sale_timestamp",
                "time_held_ms",
                "status",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():
    log("=" * 70)
    log("MEJAI PURCHASE RECONSTRUCTION")
    log("=" * 70)

    catalogue = load_catalogue()

    if catalogue.empty:
        log(
            "[ERROR] No Mejai catalogue data found"
        )
        return

    log(
        f"Catalogue events: "
        f"{len(catalogue):,}"
    )

    events = prepare_events(
        catalogue
    )

    log(
        f"Usable eligible events: "
        f"{len(events):,}"
    )

    purchase_event_count = int(
        events["event_type"]
        .eq("ITEM_PURCHASED")
        .sum()
    )

    log(
        f"Mejai purchase events: "
        f"{purchase_event_count:,}"
    )

    lifecycles = reconstruct_lifecycles(
        events
    )

    lifecycle_df = validate_lifecycles(
        lifecycles,
        events,
    )

    print_summary(
        lifecycle_df
    )

    save_lifecycles(
        lifecycles
    )

    log("")
    log(
        f"Lifecycle catalogue written to: "
        f"{LIFECYCLE_FILE}"
    )

    log("")
    log(
        "[PASSED] MEJAI PURCHASE LIFECYCLES "
        "RECONSTRUCTED AND VALIDATED"
    )


if __name__ == "__main__":
    main()