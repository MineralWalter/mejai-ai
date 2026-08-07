import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from requests.exceptions import RequestException
from threading import Thread, Lock, Event

load_dotenv()

API_KEY = os.getenv("RIOT_API_KEY")

if not API_KEY:
    raise RuntimeError("RIOT_API_KEY missing")

HEADERS = {"X-Riot-Token": API_KEY}

QUEUE = 420
MATCHES_PER_PLAYER = 20

REQUEST_DELAY = 1.25
CHECKPOINT_EVERY = 100
MATCH_TARGET = 100000

PLAYERS_FILE = "checkpoints/players_checkpoint.csv"
MATCHES_FILE = "checkpoints/matches_checkpoint.csv"
PROCESSED_FILE = "checkpoints/processed_checkpoint.csv"
QUEUE_FILE = "checkpoints/queue_checkpoint.csv"

stop_event = Event()


def riot_request(url, params=None):

    while True:

        if stop_event.is_set():
            return None

        try:
            r = requests.get(url,headers=HEADERS,params=params,timeout=30)

        except RequestException as e:
            print("Connection error:", e)
            time.sleep(10)
            continue

        if r.status_code == 200:
            time.sleep(REQUEST_DELAY)
            return r

        elif r.status_code == 401:
            print("INVALID RIOT API KEY")
            stop_event.set()
            return None

        elif r.status_code == 429:
            wait = int(r.headers.get("Retry-After", 10))
            print(f"Rate limited. Waiting {wait}s")
            time.sleep(wait)

        else:
            print(f"Request failed {r.status_code}: {url}")
            print(r.text)
            return None


def save_checkpoint():

    pd.DataFrame(list(players),columns=["puuid","platform","match_routing"]).to_csv(PLAYERS_FILE,index=False)

    pd.DataFrame(list(matches),columns=["match_id"]).to_csv(MATCHES_FILE,index=False)

    pd.DataFrame(list(processed_players),columns=["puuid"]).to_csv(PROCESSED_FILE,index=False)

    pd.DataFrame(list(queue),columns=["puuid","platform","match_routing"]).to_csv(QUEUE_FILE,index=False)

    print("Checkpoint saved |","Matches:",len(matches),"| Players:",len(players),"| Queue:",len(queue))

# Load existing state

players = set()
matches = set()
queue = set()
processed_players = set()


if os.path.exists(PLAYERS_FILE):

    df = pd.read_csv(PLAYERS_FILE)

    players = set(map(tuple, df.values))

    print("Loaded players:", len(players))


if os.path.exists(MATCHES_FILE):

    df = pd.read_csv(MATCHES_FILE)

    matches = set(df["match_id"])

    print("Loaded matches:", len(matches))


if os.path.exists(PROCESSED_FILE):

    df = pd.read_csv(PROCESSED_FILE)

    processed_players = set(df["puuid"])

    print("Loaded processed players:", len(processed_players))


if os.path.exists(QUEUE_FILE):

    df = pd.read_csv(QUEUE_FILE)

    queue = set(map(tuple, df.values))

    print("Loaded queue:", len(queue))

print("Starting BFS crawler")
print("Current matches:", len(matches))
print("Target:", MATCH_TARGET)

lock = Lock()


def build_lanes():

    lanes = {
        "sea": [],
        "asia": [],
        "europe": [],
        "americas": []
    }

    for player in queue:

        puuid, platform, routing = player # idk if this is needed, probs

        if routing in lanes:
            lanes[routing].append(player)

    return lanes

def worker(routing, player_queue):

    while True:

        if stop_event.is_set():
            break

        with lock:

            if len(matches) >= MATCH_TARGET:
                break

            if len(player_queue) == 0:
                break

            puuid, platform, match_routing = player_queue.pop()

            if puuid in processed_players:
                continue


        print(f"[{routing}] Processing player {puuid[:8]}...")


        # Player -> Match IDs

        url = (f"https://{routing}.api.riotgames.com/"f"lol/match/v5/matches/"f"by-puuid/{puuid}/ids")

        params = {"queue": QUEUE,"count": MATCHES_PER_PLAYER}

        r = riot_request(url,params)

        if r is None:
            break

        match_ids = r.json()


        # Match -> Players

        for match_id in match_ids:

            with lock:

                if len(matches) >= MATCH_TARGET:
                    break

                if match_id in matches:
                    continue

                matches.add(match_id)


            detail_url = (f"https://{match_routing}.api.riotgames.com/"f"lol/match/v5/matches/{match_id}")

            detail = riot_request(detail_url)

            if detail is None:
                continue


            data = detail.json()

            participants = (data["metadata"]["participants"])


            with lock:

                for participant in participants:

                    new_player = (participant,platform,match_routing)

                    if new_player not in players:

                        players.add(new_player)

                        queue.add(new_player)


        with lock:

            processed_players.add(puuid)

            if len(processed_players) % CHECKPOINT_EVERY == 0:
                save_checkpoint()



# Main BFS loop

while len(matches) < MATCH_TARGET:

    lanes = build_lanes()

    threads = []

    for routing in lanes:

        t = Thread(
            target=worker,
            args=(routing,lanes[routing])
        )

        t.start()
        threads.append(t)


    for t in threads:
        t.join()

    save_checkpoint()

    if sum(len(x) for x in lanes.values()) == 0:
        print("Queue empty")
        break

save_checkpoint()

print("BFS COMPLETE")
print("Matches:", len(matches))
print("Players:", len(players))
print("Processed:", len(processed_players))