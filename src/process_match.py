from src.api.riot_client import get_match, get_timeline

from src.extractors.extract_match import extract_match, extract_participants

from src.extractors.extract_snapshots import extract_snapshots

from src.extractors.extract_events import extract_events


def process_match(match_id: str) -> dict | None:
    """
    Fetch and process one Riot match.

    Returns:
        {
            "match": dict,
            "participants": list,
            "snapshots": list,
            "events": list
        }

    Returns None if processing fails.
    """
    match_json = get_match(match_id)

    if match_json is None:
        print(f"Failed match fetch: {match_id}")
        return None

    try:
        match_data = extract_match(match_json)
        participant_data = extract_participants(match_json)

    except Exception as e:
        print(f"Extraction failed {match_id}: {e}")
        return None
    timeline_json = get_timeline(match_id)

    # Timeline may fail, still keep match information

    if timeline_json is None:

        print(
            f"No timeline: {match_id}"
        )

        return {
            "match": match_data,
            "participants": participant_data,
            "snapshots": [],
            "events": []
        }
    try:
        snapshot_data = extract_snapshots(timeline_json,match_id)
        event_data = extract_events(timeline_json,match_id)

    except Exception as e:
        print(f"extraction failed  {match_id}: {e}")

        snapshot_data = []
        event_data = []

    return {
        "match": match_data,
        "participants": participant_data,
        "snapshots": snapshot_data,
        "events": event_data
    }