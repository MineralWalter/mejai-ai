def extract_snapshots(timeline_json: dict,match_id: str) -> list[dict]: # Extract player state snapshots

    snapshots = []
    frames = timeline_json["info"]["frames"]

    for frame in frames:

        timestamp = frame["timestamp"]

        participant_frames = frame["participantFrames"]

        for participant_id, player in participant_frames.items():
            position = player.get("position", {})
            snapshots.append({

                # Keys
                "match_id": match_id,
                "timestamp": timestamp,
                "participant_id": int(participant_id),

                # Economy
                "current_gold": player.get("currentGold"),
                "total_gold": player.get("totalGold"),
                "gold_per_second": player.get("goldPerSecond"),

                # Progression
                "level": player.get("level"),
                "xp": player.get("xp"),

                # Farming
                "minions_killed": player.get("minionsKilled"),
                "jungle_minions_killed": player.get("jungleMinionsKilled"),

                # Position
                "position_x": position.get("x"),
                "position_y": position.get("y")
            })

    return snapshots