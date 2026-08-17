from pathlib import Path
import pandas as pd
INPUT_FILE = Path('data/analysis/mejai_research_dataset.parquet')

def log(message=''):
    print(message)

def load_dataset():
    if not INPUT_FILE.exists():
        log(f'[ERROR] Research dataset not found: {INPUT_FILE}')
        return pd.DataFrame()
    try:
        df = pd.read_parquet(INPUT_FILE, engine='pyarrow')
    except Exception as error:
        log(f'[ERROR] Could not read research dataset: {error}')
        return pd.DataFrame()
    return df

def validate_structure(df):
    log('')
    log('========== DATASET STRUCTURE ==========')

    log(f'Rows: {len(df):,}')
    log(f'Columns: {len(df.columns):,}')

    log('')
    log('Columns:')
    
    for column in df.columns:
        log(f'  - {column}')
    log('')
    log('Dtypes:')
    for column, dtype in df.dtypes.items():
        log(f'  {column}: {dtype}')

def validate_required_columns(df):
    required_columns = [
        'case_id',
        'match_id',
        'participant_id',
        'region',
        'purchase_timestamp',
        'purchase_time_seconds',
        'lifecycle_status',
        'snapshot_timestamp',
        'snapshot_age_ms',
        'player_current_gold',
        'player_total_gold',
        'player_level',
        'player_xp',
        'player_minions_killed',
        'player_jungle_minions_killed',
        'team_total_gold_sum',
        'enemy_total_gold_sum',
        'team_total_gold_diff',
        'team_current_gold_diff',
        'team_xp_diff',
        'team_cs_diff',
        'outcome_win',
    ]
    log('')
    log('========== REQUIRED COLUMNS ==========')

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        log('[ERROR] Missing required columns:')
        for column in missing:
            log(f'  - {column}')
        return False
    log('[PASSED] All required columns present')
    return True

def validate_duplicates(df):
    log('')
    log('========== DUPLICATE CHECK ==========')

    if 'case_id' not in df.columns:
        log('[SKIPPED] case_id missing')
        return

    duplicate_cases = df['case_id'].duplicated().sum()
    duplicate_rows = df.duplicated().sum()

    log(f'Duplicate case_id rows: {duplicate_cases:,}')
    log(f'Completely duplicate rows: {duplicate_rows:,}')
    if duplicate_cases == 0:
        log('[PASSED] No duplicate case IDs')
    else:
        log('[WARNING] Duplicate case IDs found')

def validate_missing_values(df):
    log('')
    log('========== MISSING VALUE CHECK ==========')
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        log('[PASSED] No missing values')
        return
    log('Columns containing missing values:')
    for column, count in missing.items():
        percentage = count / len(df) * 100
        log(f'{column}: {count:,} ({percentage:.2f}%)')

def validate_lifecycle_status(df):
    log('')
    log('========== LIFECYCLE STATUS ==========')

    if 'lifecycle_status' not in df.columns:
        log('[SKIPPED] lifecycle_status missing')
        return

    counts = df['lifecycle_status'].value_counts(dropna=False)

    for status, count in counts.items():
        percentage = count / len(df) * 100
        log(f'{status}: {count:,} ({percentage:.2f}%)')

def validate_regions(df):
    log('')
    log('========== REGION DISTRIBUTION ==========')

    if 'region' not in df.columns:
        log('[SKIPPED] region missing')
        return

    counts = df['region'].value_counts(dropna=False)

    for region, count in counts.items():
        percentage = count / len(df) * 100
        log(f'{region}: {count:,} ({percentage:.2f}%)')

def validate_snapshot_age(df):
    log('')
    log('========== SNAPSHOT AGE ==========')

    if 'snapshot_age_ms' not in df.columns:
        log('[SKIPPED] snapshot_age_ms missing')
        return
    ages = df['snapshot_age_ms'] / 1000
    log(ages.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
    log('')

    invalid_negative = (ages < 0).sum()
    over_60 = (ages > 60).sum()
    log(f'Negative snapshot ages: {invalid_negative:,}')
    log(f'Snapshot ages > 60 seconds: {over_60:,}')

    if invalid_negative == 0:
        log('[PASSED] No future snapshots')
    if over_60 == 0:
        log('[PASSED] All snapshots within expected 60-second cadence')

def validate_temporal_order(df):
    log('')
    log('========== TEMPORAL ORDER ==========')
    required = ['snapshot_timestamp', 'purchase_timestamp']
    if not all((column in df.columns for column in required)):
        log('[SKIPPED] Required timestamp columns missing')
        return
    invalid = (df['snapshot_timestamp'] > df['purchase_timestamp']).sum()
    equal = (df['snapshot_timestamp'] == df['purchase_timestamp']).sum()
    log(f'Snapshot after purchase: {invalid:,}')
    log(f'Snapshot exactly at purchase: {equal:,}')
    if invalid == 0:
        log('[PASSED] All snapshots occur at or before purchase')

def validate_purchase_times(df):
    log('')
    log('========== PURCHASE TIMESTAMPS ==========')
    if 'purchase_time_seconds' not in df.columns:
        log('[SKIPPED] purchase_time_seconds missing')
        return
    values = df['purchase_time_seconds']
    log(values.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_string())
    negative = (values < 0).sum()
    log(f'Negative purchase times: {negative:,}')

def validate_player_state(df):
    log('')
    log('========== PLAYER STATE ==========')
    columns = [
        'player_current_gold',
        'player_total_gold',
        'player_level',
        'player_xp',
        'player_minions_killed',
        'player_jungle_minions_killed',
    ]

    for column in columns:
        if column not in df.columns:
            log(f'[SKIPPED] {column} missing')
            continue
        values = df[column]

        log('')
        log(column)
        log(values.describe().to_string())
        negative = (values < 0).sum()
        if negative > 0:
            log(f'[WARNING] Negative values: {negative:,}')

def validate_team_state(df):
    log('')
    log('========== TEAM / ENEMY STATE ==========')
    columns = [
        'team_total_gold_sum',
        'enemy_total_gold_sum',
        'team_current_gold_diff',
        'team_total_gold_diff',
        'team_xp_diff',
        'team_cs_diff',
    ]
    for column in columns:
        if column not in df.columns:
            log(f'[SKIPPED] {column} missing')
            continue

        values = df[column]
        log('')
        log(column)
        log(values.describe().to_string())

def validate_team_consistency(df):
    log('')
    log('========== TEAM CONSISTENCY ==========')
    required = [
        'match_id',
        'team_id',
        'team_total_gold_sum',
        'enemy_total_gold_sum',
        'team_current_gold_sum',
        'enemy_current_gold_sum',
        'team_current_gold_diff',
        'team_total_gold_diff',
        'team_xp_diff',
        'team_cs_diff',
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        log('[SKIPPED] Required columns missing:')
        for column in missing:
            log(f'  - {column}')
        return
    match_team_cases = df.groupby(['match_id', 'team_id']).size()
    log('Research cases per match/team:')
    log(match_team_cases.describe().to_string())

    calculated_total_gold_diff = df['team_total_gold_sum'] - df['enemy_total_gold_sum']
    total_gold_mismatch = (calculated_total_gold_diff != df['team_total_gold_diff']).sum()
    log(f'Team total-gold difference mismatches: {total_gold_mismatch:,}')

    calculated_current_gold_diff = df['team_current_gold_sum'] - df['enemy_current_gold_sum']
    current_gold_mismatch = (calculated_current_gold_diff != df['team_current_gold_diff']).sum()
    log(f'Team current-gold difference mismatches: {current_gold_mismatch:,}')

    team_case_counts = df.groupby(['match_id', 'team_id'])['case_id'].nunique()
    duplicate_case_groups = (team_case_counts != df.groupby(['match_id', 'team_id']).size()).sum()
    log(f'Match/team groups with inconsistent case representation: {duplicate_case_groups:,}')

    if total_gold_mismatch == 0 and current_gold_mismatch == 0 and (duplicate_case_groups == 0):
        log('[PASSED] Team-level features are internally consistent')
    else:
        log('[FAILED] Team-level feature consistency problems detected')

def validate_outcomes(df):
    log('')
    log('========== OUTCOME CHECK ==========')
    if 'outcome_win' in df.columns:
        log('outcome_win distribution:')
        log(df['outcome_win'].value_counts(dropna=False).to_string())

    if 'outcome_game_duration' in df.columns:
        duration = df['outcome_game_duration']
        log('Game duration seconds:')
        log(duration.describe().to_string())
        invalid = (duration <= 0).sum()
        log(f'Non-positive game durations: {invalid:,}')

def validate_status_vs_outcome(df):
    log('')
    log('========== STATUS VS OUTCOME ==========')
    if not {'lifecycle_status', 'outcome_win'}.issubset(df.columns):
        log('[SKIPPED] Required columns missing')
        return
    table = pd.crosstab(df['lifecycle_status'], df['outcome_win'], margins=True)
    log(table.to_string())

def final_summary(df):
    log('')
    log('========== FINAL VALIDATION ==========')

    checks = []
    if 'case_id' in df.columns:
        checks.append(('duplicate_case_ids', df['case_id'].duplicated().sum() == 0))
    if {'snapshot_timestamp', 'purchase_timestamp'}.issubset(df.columns):
        checks.append(('temporal_order', (df['snapshot_timestamp'] <= df['purchase_timestamp']).all()))
    if 'snapshot_age_ms' in df.columns:
        checks.append(('snapshot_age_non_negative', (df['snapshot_age_ms'] >= 0).all()))

    passed = 0
    for name, result in checks:
        if result:
            log(f'[PASSED] {name}')
            passed += 1
        else:
            log(f'[FAILED] {name}')

    log('')
    log(f'Validation checks passed: {passed}/{len(checks)}')

def main():
    log('===========================================')
    log('       RESEARCH DATASET VALIDATION')
    log('===========================================')
    df = load_dataset()
    if df.empty:
        log('[ERROR] Dataset is empty')
        return
    validate_structure(df)
    if not validate_required_columns(df):
        log('[ERROR] Required columns missing. Stopping validation.')
        return
    
    validate_duplicates(df)
    validate_missing_values(df)
    validate_lifecycle_status(df)
    validate_regions(df)
    validate_snapshot_age(df)
    validate_temporal_order(df)
    validate_purchase_times(df)
    validate_player_state(df)
    validate_team_state(df)
    validate_team_consistency(df)
    validate_outcomes(df)
    validate_status_vs_outcome(df)
    final_summary(df)

    log('')
    log('===========================================')
    log('       VALIDATION COMPLETE')
    log('===========================================')
if __name__ == '__main__':
    main()