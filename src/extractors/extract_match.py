def extract_match(match_json: dict) -> dict:
    """Extract match-level metadata from Riot Match-V5."""

    metadata = match_json["metadata"]
    info = match_json["info"]

    return {

        # IDs
        "match_id": metadata["matchId"],
        "game_id": info.get("gameId"),
        
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

            # Role / Champion
            "team_position": p.get("teamPosition"),
            "champion_id": p.get("championId"),
            "champion_name": p.get("championName"),

            # Result
            "win": p.get("win"),

            # Economy
            "gold_earned": p.get("goldEarned"),
            "gold_spent": p.get("goldSpent"),

            # Progression
            "champ_level": p.get("champLevel"),
            "champ_experience": p.get("champExperience"),

            # Combat
            "kills": p.get("kills"),
            "deaths": p.get("deaths"),
            "assists": p.get("assists"),

            "damage_dealt_to_champions": p.get(
                "totalDamageDealtToChampions"
            ),

            "damage_taken": p.get(
                "totalDamageTaken"
            ),

            # Items
            "item0": p.get("item0"),
            "item1": p.get("item1"),
            "item2": p.get("item2"),
            "item3": p.get("item3"),
            "item4": p.get("item4"),
            "item5": p.get("item5"),
            "item6": p.get("item6"),
        })

    return participants