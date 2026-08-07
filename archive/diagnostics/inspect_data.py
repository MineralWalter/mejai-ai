import os
import json
import pandas as pd


PARQUET_DIR = "data/parquet"

LANES = ["sea","asia","europe","americas"]

TABLES = ["matches","participants","snapshots","events"]

OUTPUT_FILE = "data/validation/data_profile.json"


def log(message):
    print(message)


def get_batch_ids(table, lane):
    folder = os.path.join(PARQUET_DIR, table)

    if not os.path.exists(folder):
        return []

    batch_ids = []

    prefix = f"{lane}_part_"
    suffix = ".parquet"

    for filename in os.listdir(folder):
        if filename.startswith(prefix) and filename.endswith(suffix):
            batch_string = filename[len(prefix):-len(suffix)]

            try:
                batch_ids.append(int(batch_string))
            except ValueError:
                continue

    return sorted(batch_ids)


def load_parquet(table, lane, batch_id):
    filepath = os.path.join(
        PARQUET_DIR,
        table,
        f"{lane}_part_{batch_id:05d}.parquet")

    if not os.path.exists(filepath):
        return None

    try:
        return pd.read_parquet(filepath)

    except Exception as e:
        log(f"[ERROR] Could not read "f"{lane} {table} batch {batch_id:05d}: {e}")
        return None


def make_json_safe(value):
    """
    Convert values into something JSON can store.
    """

    if isinstance(value, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(v)
            for v in value
        ]

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value


def profile_column(df, column):
    series = df[column]

    null_count = int(series.isna().sum())

    # Get a representative non-null value.
    non_null = series.dropna()

    example = None

    if len(non_null) > 0:
        example = make_json_safe(non_null.iloc[0])

    # Detect list/array-like values.
    nested = False

    for value in non_null.head(100):
        if isinstance(value, (list, tuple)):
            nested = True
            break

        # numpy arrays
        if hasattr(value, "ndim") and getattr(value, "ndim", 0) > 0:
            nested = True
            break

    result = {
        "dtype": str(series.dtype),
        "rows": int(len(series)),
        "nulls": null_count,
        "null_percentage": round((null_count / len(series)) * 100,3) if len(series) else 0,
        "example": example,
        "nested_or_array": nested
    }

    return result


def inspect_dataset():
    profile = {
        "tables": {},
        "summary": {
            "lanes": LANES,
            "tables": TABLES
        }
    }

    for table in TABLES:

        log("")
        log("================================================ ")
        log(f"TABLE: {table.upper()}")
        log("================================================ ")

        table_profile = {
            "total_rows": 0,
            "batches": 0,
            "columns": {}
        }

        for lane in LANES:

            batch_ids = get_batch_ids(table,lane)

            if not batch_ids:
                continue

            log(f"[{lane}] "f"{len(batch_ids)} batches")
            for batch_id in batch_ids:

                df = load_parquet(table,lane,batch_id)
                if df is None:
                    continue

                if df.empty:
                    log(
                        f"[WARNING] "
                        f"{lane} batch {batch_id:05d} is empty"
                    )
                    continue

                table_profile["total_rows"] += len(df)
                table_profile["batches"] += 1

                # Record schema only once.
                for column in df.columns:

                    if column not in table_profile["columns"]:
                        table_profile["columns"][column] = (
                            profile_column(df, column)
                        )

                    else:
                        # Update aggregate null counts.
                        existing = table_profile["columns"][column]

                        null_count = int(df[column].isna().sum())

                        existing["rows"] += len(df)
                        existing["nulls"] += null_count

                        # Update null percentage later.
                        existing["null_percentage"] = round((existing["nulls"]/ existing["rows"]) * 100,3)

                log(
                    f"    batch {batch_id:05d}: "
                    f"{len(df)} rows"
                )

        profile["tables"][table] = table_profile

        log("")
        log(f"TOTAL {table}: "f"{table_profile['total_rows']} rows")

        log("COLUMNS:")
        for column, info in table_profile["columns"].items():

            nested_marker = ""

            if info["nested_or_array"]:
                nested_marker = " [NESTED/ARRAY]"

            log(f"  {column:<30} "f"{info['dtype']:<15} "f"nulls={info['nulls']}"f"{nested_marker}")
    return profile


def save_profile(profile):

    os.makedirs(os.path.dirname(OUTPUT_FILE),exist_ok=True)

    with open(OUTPUT_FILE,"w",encoding="utf-8") as file:
        
        json.dump(profile,file,indent=4,ensure_ascii=False)

    log("")
    log(f"Profile written to: {OUTPUT_FILE}")

def main():

    log("================================================ ")
    log("          PARQUET DATA PROFILER")
    log("================================================ ")

    profile = inspect_dataset()

    save_profile(profile)

    log("[DONE] Dataset profiling complete.")


if __name__ == "__main__":
    main()