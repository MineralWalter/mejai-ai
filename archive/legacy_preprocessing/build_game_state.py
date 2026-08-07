from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")

REGIONS = [
    "americas",
    "asia",
    "europe",
    "sea",
]
MEJAI_ITEM_ID = 3041
RECENT_EVENT_MILISECONDS = 5 * 60 * 1000

def load_region_data(region):
    """Load the four processed tables for one region."""

    base = DATA_DIR

    matches = pd.read_parquet(
        base / f"{region}_matches.parquet"
    )

    participants = pd.read_parquet(
        base / f"{region}_participants.parquet"
    )

    snapshots = pd.read_parquet(
        base / f"{region}_snapshots.parquet"
    )

    events = pd.read_parquet(
        base / f"{region}_events.parquet"
    )

    return matches, participants, snapshots, events


# ---------------------------------------------------------------------------
# Mejai purchase detection
# ---------------------------------------------------------------------------

def find_mejai_purchases(events):
    """
    Find events that represent a Mejai purchase.

    We check both item_id and after_item_id because different event
    representations may expose item information differently.
    """

    item_id_match = events["item_id"].eq(MEJAI_ITEM_ID)

    after_item_match = events["after_item_id"].eq(MEJAI_ITEM_ID)

    purchases = events[
        item_id_match | after_item_match
    ].copy()

    return purchases


# ---------------------------------------------------------------------------
# Snapshot reconstruction
# ---------------------------------------------------------------------------

def get_snapshot_before_purchase(
    snapshots,
    match_id,
    participant_id,
    purchase_timestamp,
):
    """
    Return the latest snapshot available at or before the purchase.
    """

    player_snapshots = snapshots[
        (snapshots["match_id"] == match_id)
        & (snapshots["participant_id"] == participant_id)
        & (snapshots["timestamp"] <= purchase_timestamp)
    ].copy()

    if player_snapshots.empty:
        return None

    player_snapshots = player_snapshots.sort_values("timestamp")

    return player_snapshots.iloc[-1]


# ---------------------------------------------------------------------------
# Recent events
# ---------------------------------------------------------------------------

def get_recent_events(
    events,
    match_id,
    purchase_timestamp,
):
    """
    Return events occurring during the five minutes before purchase.
    """

    start_time = max(
        0,
        purchase_timestamp - RECENT_EVENT_MILISECONDS
    )

    recent = events[
        (events["match_id"] == match_id)
        & (events["timestamp"] >= start_time)
        & (events["timestamp"] <= purchase_timestamp)
    ].copy()

    return recent.sort_values("timestamp")


# ---------------------------------------------------------------------------
# Player information
# ---------------------------------------------------------------------------

def get_player_info(
    participants,
    match_id,
    participant_id,
):
    """Return the participant's static/final information."""

    player = participants[
        (participants["match_id"] == match_id)
        & (participants["participant_id"] == participant_id)
    ].copy()

    if player.empty:
        return None

    return player.iloc[0]


# ---------------------------------------------------------------------------
# Match information
# ---------------------------------------------------------------------------

def get_match_info(matches, match_id):
    """Return match-level information."""

    match = matches[
        matches["match_id"] == match_id
    ].copy()

    if match.empty:
        return None

    return match.iloc[0]


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

def build_case(
    matches,
    participants,
    snapshots,
    events,
    purchase,
):
    """
    Construct the observable game state surrounding one Mejai purchase.

    IMPORTANT:
    This function intentionally does NOT use final player statistics as
    decision-time features. Final statistics are returned separately only
    as outcome/context information.
    """

    match_id = purchase["match_id"]
    participant_id = int(purchase["participant_id"])
    purchase_timestamp = int(purchase["timestamp"])

    match = get_match_info(matches, match_id)

    player = get_player_info(
        participants,
        match_id,
        participant_id,
    )

    snapshot = get_snapshot_before_purchase(
        snapshots,
        match_id,
        participant_id,
        purchase_timestamp,
    )

    recent_events = get_recent_events(
        events,
        match_id,
        purchase_timestamp,
    )

    return {
        "match": match,
        "player": player,
        "snapshot": snapshot,
        "purchase": purchase,
        "recent_events": recent_events,
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_case(case, case_number):
    """Print a human-readable decision state."""

    match = case["match"]
    player = case["player"]
    snapshot = case["snapshot"]
    purchase = case["purchase"]
    recent_events = case["recent_events"]

    print()
    print("=" * 70)
    print(f"MEJAI CASE #{case_number}")
    print("=" * 70)

    print("\n[ACTION]")

    print(f"Match:       {purchase['match_id']}")
    print(f"Timestamp:   {purchase['timestamp']} seconds")
    print(f"Participant: {int(purchase['participant_id'])}")
    print(f"Item ID:     {purchase.get('item_id')}")

    if "before_item_id" in purchase:
        print(f"Before item: {purchase['before_item_id']}")

    if "after_item_id" in purchase:
        print(f"After item:  {purchase['after_item_id']}")

    print("\n[PLAYER]")

    if player is not None:
        print(f"Champion:    {player.get('champion_name')}")
        print(f"Position:    {player.get('team_position')}")
        print(f"Team:        {player.get('team_id')}")

    print("\n[GAME STATE AT PURCHASE]")

    if snapshot is None:
        print("WARNING: No snapshot found before purchase.")

    else:
        print(f"Snapshot time:      {snapshot['timestamp']}s")
        print(f"Current gold:       {snapshot['current_gold']}")
        print(f"Total gold:         {snapshot['total_gold']}")
        print(f"Gold / second:      {snapshot['gold_per_second']}")
        print(f"Level:              {snapshot['level']}")
        print(f"XP:                 {snapshot['xp']}")
        print(f"Lane CS:            {snapshot['minions_killed']}")
        print(f"Jungle CS:          {snapshot['jungle_minions_killed']}")
        print(
            f"Position:           "
            f"({snapshot['position_x']}, {snapshot['position_y']})"
        )

    print("\n[RECENT EVENTS — LAST 5 MINUTES]")

    if recent_events.empty:
        print("No events found.")

    else:
        print(f"Events found: {len(recent_events)}")

        for _, event in recent_events.tail(15).iterrows():

            event_type = event["event_type"]
            timestamp = event["timestamp"]

            participant = event.get("participant_id")

            print(
                f"  {timestamp:>6}s | "
                f"{event_type:<20} | "
                f"participant={participant}"
            )

    print("\n[FINAL OUTCOME — NOT A DECISION-TIME FEATURE]")

    if player is not None:
        print(f"Game won: {player.get('win')}")

    if match is not None:
        print(f"Game duration: {match.get('game_duration')}")
        print(f"Game version:  {match.get('game_version')}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    print("=" * 70)
    print("       MEJAI GAME-STATE RECONSTRUCTION")
    print("=" * 70)

    total_cases = 0

    for region in REGIONS:

        print(f"\nLoading region: {region}")

        matches, participants, snapshots, events = load_region_data(
            region
        )

        purchases = find_mejai_purchases(events)

        print(
            f"Mejai-related events found: {len(purchases)}"
        )

        # Only inspect the first few cases for now.
        # We don't want to dump thousands of cases.
        for _, purchase in purchases.head(5).iterrows():

            case = build_case(
                matches,
                participants,
                snapshots,
                events,
                purchase,
            )

            total_cases += 1

            print_case(
                case,
                total_cases,
            )

    print()
    print("=" * 70)
    print(f"Cases inspected: {total_cases}")
    print("=" * 70)


if __name__ == "__main__":
    main()