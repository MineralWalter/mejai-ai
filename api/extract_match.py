def extract_match(match_json: dict) -> dict:
    """Extract match-level metadata from Riot Match-V5."""

    metadata = match_json["metadata"]
    info = match_json["info"]

    return {

        # IDs
        "match_id": metadata["matchId"],

        # Game
        "game_creation": info.get("gameCreation"),
        "game_start_timestamp": info.get("gameStartTimestamp"),
        "game_duration": info.get("gameDuration"),
        "game_end_timestamp": info.get("gameEndTimestamp"),

        # Version
        "game_version": info.get("gameVersion"),

        # Queue / Map
        "queue_id": info.get("queueId"),
        "map_id": info.get("mapId"),

        # Mode
        "game_mode": info.get("gameMode"),
        "game_type": info.get("gameType"),

        # Platform
        "platform_id": info.get("platformId"),

        # Result
        "end_of_game_result": info.get("endOfGameResult")
    }

def extract_participants(match_json: dict) -> list[dict]:
    """Extract participant-level metadata."""

    match_id = match_json["metadata"]["matchId"]

    participants = []

    for p in match_json["info"]["participants"]:

        participants.append({

            # Keys
            "match_id": match_id,
            "participant_id": p["participantId"],

            # Identity
            "puuid": p["puuid"],
            "team_id": p["teamId"],

            # Champion
            "champion_id": p["championId"],
            "champion_name": p["championName"],

            # Side
            "win": p["win"]
        })

    return participants