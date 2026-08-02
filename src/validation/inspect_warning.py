from src.api.riot_client import get_match

MATCH_ID = "EUW1_7921883844"

data = get_match(MATCH_ID)

if data is None:
    print("Could not fetch match")
else:
    for participant in data["info"]["participants"]:
        print(
            participant["participantId"],
            participant["teamId"],
            participant["championName"],
            participant["win"]
        )

# Went directly to RIOT API to fetch match data
# Turns out the match was aborted because anticheat triggers
# endOfGameResult = "Abort_AntiCheatExit"