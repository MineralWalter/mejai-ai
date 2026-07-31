def extract_events(timeline_json: dict, match_id: str) -> list[dict]:
    """Extract important gameplay events from Riot timeline."""

    events = []
    frames = timeline_json.get("info", {}).get("frames", [])

    for frame in frames:

        frame_timestamp = frame["timestamp"]
        for event in frame.get("events", []):

            event_type = event.get("type")
            if event_type not in {"ITEM_PURCHASED","ITEM_SOLD","ITEM_UNDO","CHAMPION_KILL","BUILDING_KILL","ELITE_MONSTER_KILL"}:
                continue
            
            position = event.get("position", {})
            base = {
                "match_id": match_id,
                "timestamp": event.get("timestamp",frame_timestamp),
                "event_type": event_type,
                "position_x": position.get("x"),
                "position_y": position.get("y"),
            }

            if event_type in {"ITEM_PURCHASED","ITEM_SOLD","ITEM_UNDO"}:
                base.update({

                    "participant_id": event.get("participantId"),
                    "item_id": event.get("itemId"),

                    # Mainly for ITEM_UNDO
                    "before_item_id": event.get("beforeId"),
                    "after_item_id": event.get("afterId"),})
                
            elif event_type == "CHAMPION_KILL":

                base.update({
                    "killer_id": event.get("killerId"),
                    "victim_id": event.get("victimId"),
                    "assisting_ids": event.get("assistingParticipantIds"),
                }) # Might as well get every kill event instead of only keeping ones with Mejai

            elif event_type == "BUILDING_KILL":

                base.update({
                    "killer_id": event.get("killerId"),
                    "building_type": event.get("buildingType"),
                    "lane_type": event.get("laneType"),
                    "team_id": event.get("teamId"),
                })
            elif event_type == "ELITE_MONSTER_KILL":

                base.update({
                    "killer_id": event.get("killerId"),
                    "monster_type": event.get("monsterType"),
                    "monster_sub_type": event.get("monsterSubType"),
                    "item_id": (
                        int(event["itemId"])
                        if event.get("itemId") is not None
                        else None
                    ),
                })

            events.append(base)


    return events