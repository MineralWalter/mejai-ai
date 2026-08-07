from pathlib import Path
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PARQUET_DIR = Path("data/parquet")
OUTPUT_DIR = Path("data/processed")

LANES = ["sea", "asia", "europe", "americas"]


# ============================================================
# HELPERS
# ============================================================

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


def load_parquet(table, lane, batch_id):
    filepath = PARQUET_DIR / table / f"{lane}_part_{batch_id:05d}.parquet"

    if not filepath.exists():
        return None

    try:
        return pd.read_parquet(filepath)
    except Exception as e:
        log(f"[ERROR] Could not read {lane} {table} batch {batch_id:05d}: {e}")
        return None


def load_lane_table(table, lane):
    frames = []

    for batch_id in get_batch_ids(table, lane):
        df = load_parquet(table, lane, batch_id)

        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


# ============================================================
# MATCH FILTERING
# ============================================================

def identify_usable_matches(matches, participants):
    """
    A match is considered usable when:

    - the game completed normally
    - it has exactly 10 participants
    - it has exactly 5 winners
    - it has exactly 2 teams
    - participant IDs are exactly 1-10
    """

    if matches.empty or participants.empty:
        return set(), {}

    participant_counts = participants.groupby("match_id").size()
    winner_counts = participants.groupby("match_id")["win"].sum()
    team_counts = participants.groupby("match_id")["team_id"].nunique()

    participant_id_valid = participants.groupby("match_id")["participant_id"].apply(
        lambda x: set(x) == set(range(1, 11))
    )

    usable_matches = set()
    exclusion_reasons = {}

    for _, match in matches.iterrows():
        match_id = match["match_id"]
        reasons = []

        end_result = match["end_of_game_result"]

        if end_result != "GameComplete":
            reasons.append(f"end_of_game_result={end_result}")

        participant_count = participant_counts.get(match_id, 0)

        if participant_count != 10:
            reasons.append(f"participant_count={participant_count}")

        winner_count = winner_counts.get(match_id, 0)

        if winner_count != 5:
            reasons.append(f"winner_count={winner_count}")

        team_count = team_counts.get(match_id, 0)

        if team_count != 2:
            reasons.append(f"team_count={team_count}")

        if not participant_id_valid.get(match_id, False):
            reasons.append("invalid participant_id structure")

        if reasons:
            exclusion_reasons[match_id] = reasons
        else:
            usable_matches.add(match_id)

    return usable_matches, exclusion_reasons


# ============================================================
# PROCESS LANE
# ============================================================

def process_lane(lane):
    log("")
    log("=" * 50)
    log(f"PREPARING {lane.upper()}")
    log("=" * 50)

    matches = load_lane_table("matches", lane)
    participants = load_lane_table("participants", lane)
    snapshots = load_lane_table("snapshots", lane)
    events = load_lane_table("events", lane)

    if matches.empty:
        log(f"[WARNING] No match data found for {lane}")
        return

    log(f"Raw matches: {len(matches):,}")
    log(f"Raw participants: {len(participants):,}")
    log(f"Raw snapshots: {len(snapshots):,}")
    log(f"Raw events: {len(events):,}")

    usable_matches, exclusion_reasons = identify_usable_matches(matches, participants)

    log(f"Usable matches: {len(usable_matches):,}")
    log(f"Excluded matches: {len(exclusion_reasons):,}")

    if exclusion_reasons:
        log("")
        log("Exclusion summary:")

        reason_counts = {}

        for reasons in exclusion_reasons.values():
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            log(f"  {count:,} × {reason}")

    matches_clean = matches[matches["match_id"].isin(usable_matches)].copy()
    participants_clean = participants[participants["match_id"].isin(usable_matches)].copy()
    snapshots_clean = snapshots[snapshots["match_id"].isin(usable_matches)].copy()
    events_clean = events[events["match_id"].isin(usable_matches)].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matches_clean.to_parquet(OUTPUT_DIR / f"{lane}_matches.parquet", index=False)
    participants_clean.to_parquet(OUTPUT_DIR / f"{lane}_participants.parquet", index=False)
    snapshots_clean.to_parquet(OUTPUT_DIR / f"{lane}_snapshots.parquet", index=False)
    events_clean.to_parquet(OUTPUT_DIR / f"{lane}_events.parquet", index=False)

    log("")
    log("Saved:")
    log(f"  matches: {len(matches_clean):,}")
    log(f"  participants: {len(participants_clean):,}")
    log(f"  snapshots: {len(snapshots_clean):,}")
    log(f"  events: {len(events_clean):,}")


# ============================================================
# MAIN
# ============================================================

def main():
    log("===========================================")
    log("          RESEARCH DATA PREPARATION")
    log("===========================================")

    for lane in LANES:
        process_lane(lane)

    log("")
    log("===========================================")
    log("PREPARATION COMPLETE")
    log("===========================================")


if __name__ == "__main__":
    main()