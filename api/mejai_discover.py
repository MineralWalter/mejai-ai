import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from requests.exceptions import RequestException

load_dotenv()

API_KEY = os.getenv("RIOT_API_KEY")

if not API_KEY:
    raise RuntimeError("RIOT_API_KEY missing")

HEADERS = {"X-Riot-Token": API_KEY}

REQUEST_DELAY = 1.25
CHECKPOINT_EVERY = 100
MEJAI_ID = 3041

MATCHES_FILE = "matches_checkpoint.csv"
OUTPUT_FILE = "mejai_purchases.csv"
PROCESSED_FILE = "processed_matches.csv"


def riot_request(url):

    while True:

        try:
            r = requests.get(url, headers=HEADERS, timeout=30)

        except RequestException as e:
            print("Connection error:", e)
            print("Retrying in 10 seconds...")
            time.sleep(10)
            continue

        if r.status_code == 200:
            time.sleep(REQUEST_DELAY)
            return r

        elif r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 10))
            print(f"Rate limited. Waiting {wait}s")
            time.sleep(wait)

        else:
            print("Request failed:", r.status_code, url)
            print(r.text)
            time.sleep(REQUEST_DELAY)
            return None


def get_routing(match_id):

    if match_id.startswith(("KR_", "JP1_")):
        return "asia"

    elif match_id.startswith(("EUW1_", "EUN1_", "TR1_", "RU_")):
        return "europe"

    elif match_id.startswith(("NA1_", "BR1_", "LA1_", "LA2_")):
        return "americas"

    elif match_id.startswith(("VN2_", "SG2_", "TW2_", "OC1_", "PH2_", "TH2_")):
        return "sea"

    else:
        return None


def save_checkpoint():

    pd.DataFrame(mejai_purchases).to_csv(
        OUTPUT_FILE,
        index=False
    )

    pd.DataFrame(
        list(processed_matches),
        columns=["match_id"]
    ).to_csv(
        PROCESSED_FILE,
        index=False
    )

    print("Checkpoint saved")


matches = pd.read_csv(MATCHES_FILE)

mejai_purchases = []
processed_matches = set()

if os.path.exists(OUTPUT_FILE):
    mejai_purchases = pd.read_csv(OUTPUT_FILE).to_dict("records")
    print("Loaded existing Mejai purchases:", len(mejai_purchases))

if os.path.exists(PROCESSED_FILE):
    processed_matches = set(pd.read_csv(PROCESSED_FILE)["match_id"])
    print("Loaded processed matches:", len(processed_matches))

print("Scanning timelines")
print("Matches loaded:", len(matches))

for i, row in matches.iterrows():

    match_id = row["match_id"]

    if match_id in processed_matches:
        continue

    routing = get_routing(match_id)

    if routing is None:
        print("Unknown routing:", match_id)
        processed_matches.add(match_id)
        continue

    print(f"[{i+1}/{len(matches)}] {match_id}")

    url = (
        f"https://{routing}.api.riotgames.com/"
        f"lol/match/v5/matches/{match_id}/timeline"
    )

    r = riot_request(url)

    if r is None:
        continue

    timeline = r.json()

    found = False

    for frame_index, frame in enumerate(timeline["info"]["frames"]):

        for event in frame["events"]:

            if (
                event.get("type") == "ITEM_PURCHASED"
                and event.get("itemId") == MEJAI_ID
            ):

                mejai_purchases.append(
                    {
                        "match_id": match_id,
                        "routing": routing,
                        "participantId": event["participantId"],
                        "timestamp": event["timestamp"],
                        "frame": frame_index
                    }
                )

                found = True

    if found:
        print("Mejai purchase found")

    processed_matches.add(match_id)

    if len(processed_matches) % CHECKPOINT_EVERY == 0:
        save_checkpoint()

save_checkpoint()

print("======================")
print("SCAN COMPLETE")
print("Matches scanned:", len(processed_matches))
print("Mejai purchases:", len(mejai_purchases))