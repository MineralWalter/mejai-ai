import json
import math
import os
from pathlib import Path
import pandas as pd
PARQUET_DIR = Path('data/parquet')
REPORT_DIR = Path('data/validation')
REPORT_FILE = REPORT_DIR / 'validation_report.json'
BATCH_SIZE = 100
TABLES = ['matches', 'participants', 'snapshots', 'events']
LANES = ['sea', 'asia', 'europe', 'americas']

def log(message):
    print(message)

def parquet_path(table, lane, batch_id):
    return PARQUET_DIR / table / f'{lane}_part_{batch_id:05d}.parquet'

def get_batch_ids(table, lane):
    directory = PARQUET_DIR / table
    if not directory.exists():
        return []
    batch_ids = []
    prefix = f'{lane}_part_'
    suffix = '.parquet'
    for file in directory.glob(f'{lane}_part_*.parquet'):
        name = file.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        batch_string = name[len(prefix):-len(suffix)]
        try:
            batch_ids.append(int(batch_string))
        except ValueError:
            pass
    return sorted(batch_ids)

def load_parquet(table, lane, batch_id):
    path = parquet_path(table, lane, batch_id)
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:
        log(f'[ERROR] Could not read {table}/{lane}/batch {batch_id}: {e}')
        return None

def validate_batches(results):
    log('\n========== BATCH VALIDATION ==========')
    for lane in LANES:
        log(f'\n--- {lane.upper()} ---')
        table_batches = {table: get_batch_ids(table, lane) for table in TABLES}
        all_batches = sorted(set().union(*table_batches.values()))

        if not all_batches:
            results['errors'].append(f'No Parquet batches found for {lane}')
            continue
        expected_batches = list(range(min(all_batches), max(all_batches) + 1))
        missing_batch_numbers = sorted(set(expected_batches) - set(all_batches))

        if missing_batch_numbers:
            message = f'{lane}: missing batch numbers {missing_batch_numbers}'
            log(f'[ERROR] {message}')
            results['errors'].append(message)
        for batch_id in all_batches:
            missing_tables = []
            for table in TABLES:
                if batch_id not in table_batches[table]:
                    missing_tables.append(table)
            if missing_tables:
                message = f'{lane} batch {batch_id:05d}: missing tables: {missing_tables}'
                log(f'[ERROR] {message}')
                results['errors'].append(message)
        for batch_id in all_batches:
            counts = {}
            for table in TABLES:
                df = load_parquet(table, lane, batch_id)
                if df is not None:
                    counts[table] = len(df)

            results['batch_counts'].append({'lane': lane, 'batch_id': batch_id, 'counts': counts})
            match_count = counts.get('matches')
            if match_count is None:
                continue

            if match_count == 0:
                message = f'{lane} batch {batch_id:05d}: ZERO matches'
                log(f'[ERROR] {message}')
                results['errors'].append(message)
            elif match_count > BATCH_SIZE:
                message = f'{lane} batch {batch_id:05d}: {match_count} matches (more than batch size {BATCH_SIZE})'
                log(f'[ERROR] {message}')
                results['errors'].append(message)
            elif match_count < BATCH_SIZE:
                # The final batch for a region can legitimately be smaller.
                log(f'[INFO] {lane} batch {batch_id:05d}: {match_count} matches')

def validate_schema(results):
    log('\n========== SCHEMA VALIDATION ==========')
    for lane in LANES:
        for table in TABLES:
            batch_ids = get_batch_ids(table, lane)
            for batch_id in batch_ids:
                path = parquet_path(table, lane, batch_id)
                try:
                    df = pd.read_parquet(path)
                except Exception as e:
                    message = f'{lane} {table} batch {batch_id:05d}: unreadable Parquet: {e}'
                    log(f'[ERROR] {message}')
                    results['errors'].append(message)
                    continue
                if not isinstance(df, pd.DataFrame):
                    message = f'{lane} {table} batch {batch_id:05d}: invalid DataFrame'
                    log(f'[ERROR] {message}')
                    results['errors'].append(message)
                results["schemas"].append(
                    {
                        "lane": lane,
                        "table": table,
                        "batch_id": batch_id,
                        "columns": list(df.columns),
                        "dtypes": {
                            column: str(dtype)
                            for column, dtype in df.dtypes.items()
                        },
                    }
                )

def find_match_id_column(df):
    candidates = ['match_id', 'game_id', 'id']
    for column in candidates:
        if column in df.columns:
            return column
    return None

def validate_match_ids(results):
    log('\n========== MATCH ID VALIDATION ==========')
    for lane in LANES:
        for batch_id in get_batch_ids('matches', lane):
            df = load_parquet('matches', lane, batch_id)
            if df is None:
                continue

            match_column = find_match_id_column(df)

            if match_column is None:
                message = f'{lane} batch {batch_id:05d}: could not find match ID column'
                log(f'[WARNING] {message}')
                results['warnings'].append(message)
                continue

            duplicate_count = df[match_column].duplicated().sum()

            if duplicate_count:
                message = f'{lane} batch {batch_id:05d}: {duplicate_count} duplicate match IDs'
                log(f'[ERROR] {message}')
                results['errors'].append(message)

def validate_participants(results):
    log('\n========== PARTICIPANT VALIDATION ==========')
    for lane in LANES:
        for batch_id in get_batch_ids('matches', lane):
            matches = load_parquet('matches', lane, batch_id)
            participants = load_parquet('participants', lane, batch_id)
            if matches is None or participants is None:
                continue

            match_column = find_match_id_column(matches)
            participant_match_column = find_match_id_column(participants)

            if match_column is None or participant_match_column is None:
                log(f'[WARNING] {lane} batch {batch_id:05d}: could not determine match ID columns')
                continue

            participant_counts = participants.groupby(participant_match_column).size()
            bad_counts = participant_counts[participant_counts != 10]

            if not bad_counts.empty:
                message = f'{lane} batch {batch_id:05d}: {len(bad_counts)} matches do not have exactly 10 participants'
                log(f'[WARNING] {message}')
                results['warnings'].append(message)

            match_ids = set(matches[match_column].dropna())
            participant_match_ids = set(participants[participant_match_column].dropna())
            orphan_ids = participant_match_ids - match_ids

            if orphan_ids:
                message = (
                    f"{lane} batch {batch_id:05d}: "
                    f"{len(orphan_ids)} participant match IDs have no matching match row"
                )
                log(f'[ERROR] {message}')
                results['errors'].append(message)

def validate_nulls_and_duplicates(results):
    log('\n========== NULL / DUPLICATE VALIDATION ==========')
    for lane in LANES:
        for table in TABLES:
            for batch_id in get_batch_ids(table, lane):
                df = load_parquet(table, lane, batch_id)
                if df is None or df.empty:
                    continue

                # Identify nested or array-like columns before other checks.
                nested_columns = []
                for column in df.columns:
                    sample = df[column].dropna().head(20)
                    if any((isinstance(value, (list, tuple, dict)) or hasattr(value, 'shape') for value in sample)):
                        nested_columns.append(column)

                if nested_columns:
                    message = f'{lane} {table} batch {batch_id:05d}: nested/array columns: {nested_columns}'
                    log(f'[INFO] {message}')

                null_columns = [column for column in df.columns if df[column].isna().all()]
                if null_columns:
                    message = f'{lane} {table} batch {batch_id:05d}: completely-null columns: {null_columns}'
                    log(f'[WARNING] {message}')
                    results['warnings'].append(message)

def build_summary(results):
    summary = {
        "total_batches": 0,
        "total_errors": len(results["errors"]),
        "total_warnings": len(results["warnings"]),
        "tables": {},
    }
    for table in TABLES:
        summary['tables'][table] = {'files': 0, 'rows': 0}
        for lane in LANES:
            for batch_id in get_batch_ids(table, lane):
                df = load_parquet(table, lane, batch_id)
                if df is None:
                    continue
                summary['tables'][table]['files'] += 1
                summary['tables'][table]['rows'] += len(df)

    # Count unique region/batch combinations.
    batch_pairs = {(item['lane'], item['batch_id']) for item in results['batch_counts']}
    summary['total_batches'] = len(batch_pairs)
    return summary

def save_report(results):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results['summary'] = build_summary(results)

    with open(REPORT_FILE, 'w', encoding='utf-8') as file:
        json.dump(results, file, indent=4, default=str)
    log(f'\nValidation report written to: {REPORT_FILE}')

def main():
    log('========================================')
    log('       PARQUET DATA VALIDATION')
    log('========================================')

    if not PARQUET_DIR.exists():
        log(f'[ERROR] Parquet directory does not exist: {PARQUET_DIR}')
        return
    results = {'errors': [], 'warnings': [], 'batch_counts': [], 'schemas': [], 'summary': {}}

    validate_batches(results)
    validate_schema(results)
    validate_match_ids(results)
    validate_participants(results)
    validate_nulls_and_duplicates(results)

    save_report(results)

    log('\n========================================')
    log('             FINAL RESULT')
    log('========================================')

    if results['errors']:
        log(f"[FAILED] {len(results['errors'])} errors found")

    else:
        log('[PASSED] No critical errors found')

    if results['warnings']:
        log(f"[WARNING] {len(results['warnings'])} warnings found")
    log('========================================')
if __name__ == '__main__':
    main()