from pathlib import Path
import json
import numpy as np
import pandas as pd

PARQUET_DIR = Path("data/parquet")
OUTPUT_DIR = Path("data/analysis")
OUTPUT_FILE = OUTPUT_DIR / "feature_schema.json"

LANES = ["sea", "asia", "europe", "americas"]
TABLES = ["matches", "participants", "snapshots", "events"]


def log(message):
    print(message)


def find_files(table, lane):
    directory = PARQUET_DIR / table
    return sorted(directory.glob(f"{lane}_part_*.parquet"))


def safe_unique_count(series):
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        try:
            return int(
                series.map(
                    lambda value: repr(value)
                ).nunique(dropna=True)
            )
        except Exception:
            return None


def safe_examples(series, limit=3):
    examples = []

    for value in series.dropna().head(20):
        try:
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif hasattr(value, "tolist"):
                value = value.tolist()

            if hasattr(value, "item"):
                value = value.item()

            examples.append(value)

            if len(examples) >= limit:
                break

        except Exception:
            examples.append(str(value))

    return examples


def inspect_column(series):
    row_count = len(series)

    null_count = int(series.isna().sum())

    if row_count:
        null_percentage = (
            null_count / row_count
        ) * 100
    else:
        null_percentage = 0.0

    return {
        "dtype": str(series.dtype),
        "null_count": null_count,
        "null_percentage": round(
            null_percentage,
            2
        ),
        "unique_count": safe_unique_count(series),
        "examples": safe_examples(series)
    }


def inspect_table(table, lane):
    files = find_files(table, lane)

    if not files:
        log(f"[WARNING] No files found for {lane}/{table}")
        return None

    sample_file = files[0]

    try:
        df = pd.read_parquet(sample_file)
    except Exception as e:
        log(f"[ERROR] Could not read {sample_file}: {e}")
        return None

    schema = {}

    for column in df.columns:
        schema[column] = inspect_column(
            df[column]
        )

    return {
        "table": table,
        "region": lane,
        "file_count": len(files),
        "sample_file": str(sample_file),
        "sample_rows": len(df),
        "column_count": len(df.columns),
        "columns": schema
    }


def inspect_all_tables():
    results = []

    for lane in LANES:
        for table in TABLES:
            log("")
            log(f"========== {lane.upper()} / {table.upper()} ==========")

            result = inspect_table(
                table,
                lane
            )

            if result is None:
                continue

            log(f"Files: {result['file_count']:,}")
            log(f"Sample file: {result['sample_file']}")
            log(f"Rows in sample: {result['sample_rows']:,}")
            log(f"Columns: {result['column_count']:,}")

            log("")
            log("---------- SCHEMA ----------")

            for column, details in result["columns"].items():
                log(
                    f"{column}: "
                    f"dtype={details['dtype']}, "
                    f"null={details['null_percentage']:.2f}%, "
                    f"unique={details['unique_count']}, "
                    f"examples={details['examples']}"
                )

            results.append(result)

    return results


def build_feature_catalogue(results):
    catalogue = {}

    for result in results:
        table = result["table"]
        region = result["region"]

        if table not in catalogue:
            catalogue[table] = {}

        catalogue[table][region] = result["columns"]

    return catalogue


def save_schema(results):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    catalogue = build_feature_catalogue(
        results
    )

    report = {
        "tables": TABLES,
        "regions": LANES,
        "schema": catalogue
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    log("")
    log(f"Schema report written to: {OUTPUT_FILE}")


def main():
    log("===========================================")
    log("       RESEARCH FEATURE SCHEMA INSPECTION")
    log("===========================================")

    results = inspect_all_tables()

    if not results:
        log("[ERROR] No table data could be inspected")
        return

    save_schema(results)

    log("")
    log("===========================================")
    log("[PASSED] FEATURE SCHEMA INSPECTION COMPLETE")
    log("===========================================")


if __name__ == "__main__":
    main()