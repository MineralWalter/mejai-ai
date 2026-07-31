import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from urllib.parse import quote

load_dotenv()

API_KEY = os.getenv("RIOT_API_KEY")

if not API_KEY:
    raise RuntimeError("RIOT_API_KEY missing")

HEADERS = {"X-Riot-Token": API_KEY}

QUEUE = 420
MATCHES_PER_PLAYER = 20

REQUEST_DELAY = 1.3
CHECKPOINT_EVERY = 50

SEEDS = [
    {"game_name": "Vincent","tag": "2604V","platform": "vn2","match_routing": "sea", "account_routing":"asia"},
    {"game_name": "Kaga","tag": "7L3","platform": "euw1","match_routing": "europe", "account_routing":"europe"},
    {"game_name": "Xiang","tag": "God","platform": "kr","match_routing": "asia", "account_routing":"asia"},
    {"game_name": "was","tag": "10000","platform": "na1","match_routing": "americas", "account_routing":"americas"}
]

match_ids = set()
known_puuids = set()

def riot_request(url, params=None):
    while True:
        r = requests.get(url,headers=HEADERS,params=params)
        time.sleep(REQUEST_DELAY)

        if r.status_code == 200:
            return r

        elif r.status_code == 429:
            wait = int(r.headers.get("Retry-After",10))

            print(f"Rate limited. Waiting {wait}s")
            time.sleep(wait)

        else:
            print(f"Request failed {r.status_code}: {url}")
            print(r.text)
            return None


def save_checkpoint():
    pd.DataFrame(list(match_ids),columns=["match_id","routing","platform"]).to_csv("match_ids_checkpoint.csv",index=False)
    pd.DataFrame(list(known_puuids),columns=["puuid","routing","platform"]).to_csv("puuids_checkpoint.csv",index=False)

# Riot ID -> PUUID -> Match IDs

for seed in SEEDS:
    name = seed["game_name"]
    tag = seed["tag"]
    account_routing = seed["account_routing"]
    match_routing = seed["match_routing"]
    platform = seed["platform"]
    print(f"\nLooking up {name}#{tag} ({platform})")

    url = (f"https://{account_routing}.api.riotgames.com/"f"riot/account/v1/accounts/"f"by-riot-id/"f"{quote(name)}/"f"{quote(tag)}")

    r = riot_request(url)

    if r is None:
        continue

    puuid = r.json()["puuid"]
    known_puuids.add((puuid,account_routing,platform))
    url = (f"https://{match_routing}.api.riotgames.com/"f"lol/match/v5/matches/"f"by-puuid/{puuid}/ids")

    params = {"queue": QUEUE,"count": MATCHES_PER_PLAYER}
    r = riot_request(url,params)

    if r is None:
        continue

    ids = r.json()

    print(f"Found {len(ids)} matches")

    for match_id in ids:
        match_ids.add((match_id,match_routing,platform))

    time.sleep(REQUEST_DELAY)

# Match -> Participants

print("\nDownloading match details")


for i, (match_id, routing, platform) in enumerate(match_ids, 1):

    print(f"[{i}/{len(match_ids)}] {match_id}")
    url = (f"https://{routing}.api.riotgames.com/"f"lol/match/v5/matches/{match_id}")

    r = riot_request(url)
    if r is None:
        continue

    match = r.json()

    for puuid in match["metadata"]["participants"]:

        known_puuids.add((puuid,routing,platform))

    if i % CHECKPOINT_EVERY == 0:
        print("Saving checkpoint...")
        save_checkpoint()
        
    time.sleep(REQUEST_DELAY)

pd.DataFrame(list(match_ids),columns=["match_id","routing","platform"]).to_csv("match_ids.csv",index=False)

pd.DataFrame(list(known_puuids),columns=["puuid","routing","platform"]).to_csv("puuids.csv",index=False)

print(f"Unique matches: {len(match_ids)}")
print(f"Unique players: {len(known_puuids)}")