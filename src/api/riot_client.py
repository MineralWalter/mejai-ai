import os
import time
import requests
from dotenv import load_dotenv
from requests.exceptions import RequestException
from threading import Event

load_dotenv()

API_KEY = os.getenv("RIOT_API_KEY")

if not API_KEY:
    raise RuntimeError("RIOT_API_KEY missing")

HEADERS = {
    "X-Riot-Token": API_KEY
}

REQUEST_DELAY = 1.5

stop_event = Event()


def riot_request(url: str, params=None) -> dict | None:

    while True:

        if stop_event.is_set():
            return None

        try:

            r = requests.get(url,headers=HEADERS,params=params,timeout=10)

        except RequestException as e:

            print("Connection error:", e)
            time.sleep(10)
            continue

        if r.status_code == 200:

            time.sleep(REQUEST_DELAY)
            return r.json()

        elif r.status_code == 401:

            print("INVALID RIOT API KEY")
            stop_event.set()
            return None

        elif r.status_code == 429:

            wait = int(r.headers.get("Retry-After",10))

            print(f"Rate limited. Waiting {wait}s")
            time.sleep(wait)

        else:
            print(f"Request failed {r.status_code}: {url}")
            print(r.text)
            return None

def infer_routing(match_id: str) -> str | None:

    if match_id.startswith(("KR_","JP1_")):
        return "asia"

    elif match_id.startswith(("EUW1_","EUN1_","TR1_","RU_")): # Just in case
        return "europe"

    elif match_id.startswith(("NA1_","BR1_","LA1_","LA2_")): # Just in case
        return "americas"

    elif match_id.startswith(("VN2_","SG2_","TW2_","OC1_","PH2_","TH2_")):
        return "sea"

    return None

def get_match(match_id: str) -> dict | None:

    routing = infer_routing(match_id)
    if routing is None:

        print(f"Unknown routing: {match_id}")

        return None

    url = (f"https://{routing}.api.riotgames.com/"f"lol/match/v5/matches/"f"{match_id}")
    return riot_request(url)

def get_timeline(match_id: str) -> dict | None:

    routing = infer_routing(match_id)
    if routing is None:

        print(f"Unknown routing: {match_id}")
        return None

    url = (f"https://{routing}.api.riotgames.com/"f"lol/match/v5/matches/"f"{match_id}/timeline")
    return riot_request(url)