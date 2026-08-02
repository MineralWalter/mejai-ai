from pathlib import Path
import json
import pandas as pd


PARQUET_DIR = Path("data/parquet")
REPORT_DIR = Path("data/validation")
REPORT_FILE = REPORT_DIR / "relationship_validation_report.json"

LANES = ["sea", "asia", "europe", "americas"]
TABLES = ["matches", "participants", "snapshots", "events"]


def log(message):
    print(message)


def get_batch_ids(table, lane):
    directory = PARQUET_DIR / table

    if not directory.exists():
        return []

    batch_ids = []

    for file in directory.glob(f"{lane}_part_*.parquet"):
        try:
            batch_id = int(file.stem.split("_part_")[1])
            batch_ids.append(batch_id)
        except (IndexError, ValueError):
            continue

    return sorted(batch_ids)


def load_table(table, lane, batch_id):
    filepath = (PARQUET_DIR/ table/ f"{lane}_part_{batch_id:05d}.parquet")

    if not filepath.exists():
        return None

    try:
        return pd.read_parquet(filepath)
    except Exception as e:
        log(
            f"[ERROR] Could not read "
            f"{lane} {table} batch {batch_id:05d}: {e}"
        )
        return None

def load_lane_table(table, lane):
    frames = []

    for batch_id in get_batch_ids(table, lane):
        df = load_table(table, lane, batch_id)

        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def add_error(results, message):
    results["errors"].append(message)
    log(f"[ERROR] {message}")


def add_warning(results, message):
    results["warnings"].append(message)
    log(f"[WARNING] {message}")


def validate_match_participants(matches, participants, lane, results):
    log(f"\n[{lane}] MATCH → PARTICIPANTS")

    if matches.empty or participants.empty:
        add_error(results,f"{lane}: matches or participants table is empty")
        return

    match_ids = set(matches["match_id"].dropna())

    participant_match_ids = set(participants["match_id"].dropna())

    # Participants referring to matches that don't exist
    orphan_participants = participant_match_ids - match_ids

    if orphan_participants:
        add_error(results,f"{lane}: {len(orphan_participants)} "f"participant match_ids do not exist in matches")

    # Participant count per match
    counts = (participants.groupby("match_id").size())

    bad_counts = counts[counts != 10]

    if not bad_counts.empty:
        add_warning(results,f"{lane}: {len(bad_counts)} matches do not "f"have exactly 10 participants")

        for match_id, count in bad_counts.items():
            log(f"    {match_id}: "f"{count} participants")

    invalid_ids = participants[~participants["participant_id"].between(1, 10)]

    if not invalid_ids.empty:
        add_warning(results,f"{lane}: {len(invalid_ids)} participants "f"have invalid participant_id values")
    # Expecting participant_id values to be in the range 1-10
    participant_id_counts = (
        participants
        .groupby(["match_id", "participant_id"])
        .size()
    )

    duplicate_participant_ids = (
        participant_id_counts[
            participant_id_counts > 1
        ]
    )

    if not duplicate_participant_ids.empty:
        add_warning(
            results,
            f"{lane}: {len(duplicate_participant_ids)} "
            f"duplicate (match_id, participant_id) pairs"
        )

    # expecting team_id values to be in the range 1-2
def validate_match_snapshots(matches, snapshots, lane, results):
    log(f"\n[{lane}] MATCH → SNAPSHOTS")

    if matches.empty or snapshots.empty:
        add_warning(results,f"{lane}: matches or snapshots table is empty")
        return

    match_ids = set(matches["match_id"].dropna())

    snapshot_match_ids = set(
        snapshots["match_id"].dropna()
    )

    orphan_snapshots = snapshot_match_ids - match_ids

    if orphan_snapshots:
        add_error(
            results,
            f"{lane}: {len(orphan_snapshots)} "
            f"snapshot match_ids do not exist in matches"
        )

    # Snapshot participant references
    participant_keys = set(
        zip(
            snapshots["match_id"],
            snapshots["participant_id"]
        )
    )


def validate_snapshot_participants(participants,snapshots,lane,results):
    log(f"\n[{lane}] SNAPSHOTS → PARTICIPANTS")

    if participants.empty or snapshots.empty:
        return

    valid_keys = set(zip(participants["match_id"],participants["participant_id"]))
    snapshot_keys = set(zip(snapshots["match_id"],snapshots["participant_id"]))
    orphan_keys = snapshot_keys - valid_keys

    if orphan_keys:
        add_error(results,f"{lane}: {len(orphan_keys)} "f"snapshot (match_id, participant_id) "f"pairs do not exist in participants")


def validate_match_events(matches, events, lane, results):
    log(f"\n[{lane}] MATCH → EVENTS")

    if matches.empty or events.empty:
        add_warning(results,f"{lane}: matches or events table is empty")
        return

    match_ids = set(matches["match_id"].dropna())
    event_match_ids = set(events["match_id"].dropna())
    orphan_events = event_match_ids - match_ids

    if orphan_events:
        add_error(results,f"{lane}: {len(orphan_events)} "f"event match_ids do not exist in matches")

def validate_match_winners(matches, participants, lane, results):
    log(f"\n[{lane}] MATCH → WINNERS")

    if matches.empty or participants.empty:
        return

    # Count winners per match
    winner_counts = (participants.groupby("match_id")["win"].sum())

    # Only inspect matches that have 0 winners ( this is added AFTER the fact that one match was aborted due to anticheat triggers)
    zero_winner_matches = winner_counts[winner_counts == 0]
    for match_id in zero_winner_matches.index:

        match_row = matches[matches["match_id"] == match_id]

        if match_row.empty:
            add_error(results,f"{lane}: {match_id} has 0 winners "f"but match does not exist")
            continue

        end_result = match_row.iloc[0]["end_of_game_result"]

        # Now that I know about the existence of legitimate abnormal termination
        if end_result == "Abort_AntiCheatExit":

            message = (f"{lane}: {match_id} has 0 winners "f"because end_of_game_result="f"{end_result}")
            log(f"[INFO] {message}")
            results.setdefault("expected_anomalies", []).append(message)

        else:
            message = (f"{lane}: {match_id} has 0 winners "f"with end_of_game_result="f"{end_result}")
            add_warning(results, message)

    # More than 5 winners is never expected
    too_many_winners = winner_counts[winner_counts > 5]
    for match_id, count in too_many_winners.items():

        add_warning(results,f"{lane}: {match_id} has "f"{count} winners")


def validate_team_structure(participants, lane, results):
    log(f"\n[{lane}] TEAM STRUCTURE")

    if participants.empty:
        return

    team_counts = (participants.groupby(["match_id", "team_id"]).size())

    bad_team_counts = team_counts[
        team_counts != 5]

    if not bad_team_counts.empty:
        add_warning(results,f"{lane}: {len(bad_team_counts)} "f"(match_id, team_id) groups do not "f"contain exactly 5 players")

    # Each match should normally have two teams
    team_per_match = (participants.groupby("match_id")["team_id"].nunique())

    bad_team_numbers = team_per_match[
        team_per_match != 2
    ]

    if not bad_team_numbers.empty:
        add_warning(results,f"{lane}: {len(bad_team_numbers)} matches "f"do not contain exactly 2 teams")


def validate_all_lanes():
    results = {
        "errors": [],
        "warnings": [],
        "expected_anomalies": [],
        "summary": {}
    }

    log("=======================================")
    log("       CROSS-TABLE VALIDATION")
    log("=======================================")

    for lane in LANES:

        log(f"\n\n========== {lane.upper()} ==========")

        matches = load_lane_table("matches",lane)
        participants = load_lane_table("participants",lane)
        snapshots = load_lane_table("snapshots",lane)
        events = load_lane_table("events",lane)

        results["summary"][lane] = {
            "matches": len(matches),
            "participants": len(participants),
            "snapshots": len(snapshots),
            "events": len(events)
        }

        log(f"Matches:       {len(matches):,}")
        log(f"Participants:  {len(participants):,}")
        log(f"Snapshots:     {len(snapshots):,}")
        log(f"Events:        {len(events):,}")

        validate_match_participants(matches,participants,lane,results)
        validate_match_snapshots(matches,snapshots,lane,results)
        validate_snapshot_participants(participants,snapshots,lane,results)
        validate_match_events(matches,events,lane,results)
        validate_match_winners(matches,participants,lane,results)
        validate_team_structure(participants,lane,results)

    REPORT_DIR.mkdir(parents=True,exist_ok=True)

    with open(REPORT_FILE,"w",encoding="utf-8") as f:
        json.dump(results,f,indent=2)

    log(f"\nReport written to: {REPORT_FILE}")
    log("\n=======================================")

    if results["errors"]:
        log(f"[FAILED] "f"{len(results['errors'])} critical errors")
    else:
        log("[PASSED] No critical cross-table errors")

    log(f"Warnings: {len(results['warnings'])}")


if __name__ == "__main__":
    validate_all_lanes()