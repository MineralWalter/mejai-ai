import os
import pandas as pd

PARQUET_DIR = "data/parquet"

LANES = ["sea", "asia", "europe", "americas"]
FOLDERS = ["matches", "participants", "snapshots", "events"]

EXPECTED_MATCHES = 100


def check_batch(lane, batch_id):
    results = {}

    for folder in FOLDERS:
        filepath = os.path.join(
            PARQUET_DIR,
            folder,
            f"{lane}_part_{batch_id:05d}.parquet"
        )

        if not os.path.exists(filepath):
            results[folder] = None
            continue

        try:
            df = pd.read_parquet(filepath)
            results[folder] = len(df)
        except Exception as e:
            results[folder] = f"ERROR: {e}"

    return results


def main():
    print("Checking parquet batches...\n")

    for lane in LANES:
        match_dir = os.path.join(PARQUET_DIR, "matches")

        batch_ids = []

        for filename in os.listdir(match_dir):
            prefix = f"{lane}_part_"

            if filename.startswith(prefix) and filename.endswith(".parquet"):
                batch_id = int(
                    filename[len(prefix):-8]
                )
                batch_ids.append(batch_id)

        if not batch_ids:
            print(f"[{lane}] No batches found")
            continue

        print(f"========== {lane.upper()} ==========")

        faulty = 0

        for batch_id in sorted(batch_ids):
            results = check_batch(lane, batch_id)

            match_count = results["matches"]

            problems = []

            if match_count is None:
                problems.append("matches file missing")
            elif match_count != EXPECTED_MATCHES:
                problems.append(
                    f"matches={match_count}"
                )

            for folder in FOLDERS:
                if results[folder] is None:
                    problems.append(f"{folder}=MISSING")
                elif isinstance(results[folder], str):
                    problems.append(
                        f"{folder}={results[folder]}"
                    )

            if not problems:
                # Check that all files have sensible row counts
                expected_participants = match_count * 10

                if results["participants"] != expected_participants:
                    problems.append(
                        f"participants={results['participants']} "
                        f"(expected {expected_participants})"
                    )

            if problems:
                faulty += 1

                print(
                    f"[FAULTY] Batch {batch_id:03d}: "
                    + " | ".join(problems)
                )

        if faulty == 0:
            print("No faulty batches found.")

        print(f"Faulty batches: {faulty}\n")


if __name__ == "__main__":
    main()