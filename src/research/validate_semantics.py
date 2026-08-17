from pathlib import Path
import pandas as pd
INPUT_FILE = Path("data/analysis/mejai_research_dataset.parquet")

def log(message=''):
    print(message)

def load_dataset():
    if not INPUT_FILE.exists():
        log(f'  Research dataset not found: {INPUT_FILE}')
        return pd.DataFrame()
    try:
        df = pd.read_parquet(INPUT_FILE, engine='pyarrow')
    except Exception as error:
        log(f'  Could not read research dataset: {error}')
        return pd.DataFrame()
    return df

def validate_team_gold(df):
    log('')
    log('========== TEAM GOLD SEMANTICS ==========')

    required = [
        "team_current_gold_sum",
        "enemy_current_gold_sum",
        "team_total_gold_sum",
        "enemy_total_gold_sum",
        "team_current_gold_diff",
        "team_total_gold_diff",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        log('[SKIPPED] Required columns missing:')
        for column in missing:
            log(f'  - {column}')
        return

    calculated_current_diff = df['team_current_gold_sum'] - df['enemy_current_gold_sum']
    current_diff_mismatch = (calculated_current_diff != df['team_current_gold_diff']).sum()

    log(f'Current-gold difference mismatches: {current_diff_mismatch:,}')
    calculated_total_diff = df['team_total_gold_sum'] - df['enemy_total_gold_sum']
    total_diff_mismatch = (calculated_total_diff != df['team_total_gold_diff']).sum()

    log(f'Total-gold difference mismatches: {total_diff_mismatch:,}')
    team_current_exceeds_total = (df['team_current_gold_sum'] > df['team_total_gold_sum']).sum()
    enemy_current_exceeds_total = (df['enemy_current_gold_sum'] > df['enemy_total_gold_sum']).sum()

    log(f'Team current gold > team total gold: {team_current_exceeds_total:,}')
    log(f'Enemy current gold > enemy total gold: {enemy_current_exceeds_total:,}')
    if (
        current_diff_mismatch == 0
        and total_diff_mismatch == 0
        and team_current_exceeds_total == 0
        and enemy_current_exceeds_total == 0
    ):
        log('[PASSED] Team gold relationships are internally consistent')
    else:
        log('[FAILED] Team gold relationship problems detected')

def validate_xp(df):
    log('')
    log('========== XP SEMANTICS ==========')
    required = [
        "team_xp_sum",
        "enemy_xp_sum",
        "team_xp_diff",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        log('[SKIPPED] Required XP columns missing')
        for column in missing:
            log(f'  - {column}')
        return
    calculated_diff = df['team_xp_sum'] - df['enemy_xp_sum']
    mismatch = (calculated_diff != df['team_xp_diff']).sum()

    log(f'XP difference mismatches: {mismatch:,}')

    negative_team_xp = (df['team_xp_sum'] < 0).sum()
    negative_enemy_xp = (df['enemy_xp_sum'] < 0).sum()

    log(f'Negative team XP totals: {negative_team_xp:,}')
    log(f'Negative enemy XP totals: {negative_enemy_xp:,}')
    if mismatch == 0 and negative_team_xp == 0 and (negative_enemy_xp == 0):
        log('[PASSED] XP features are internally consistent')
    else:
        log('[FAILED] XP feature consistency problems detected')

def validate_cs(df):
    log('')
    log('========== CS SEMANTICS ==========')

    required = [
        "team_minions_killed_sum",
        "enemy_minions_killed_sum",
        "team_jungle_minions_killed_sum",
        "enemy_jungle_minions_killed_sum",
        "team_cs_diff",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        log('[SKIPPED] Required CS columns missing')
        for column in missing:
            log(f'  - {column}')
        return
    # team_cs_diff tracks lane minions only; jungle CS is handled separately.
    calculated_lane_cs_diff = df['team_minions_killed_sum'] - df['enemy_minions_killed_sum']
    lane_cs_mismatch = (calculated_lane_cs_diff != df['team_cs_diff']).sum()
    log(f'Lane-CS difference mismatches: {lane_cs_mismatch:,}')

    # Informational only this is not expected to equal team_cs_diff.
    calculated_jungle_cs_diff = df['team_jungle_minions_killed_sum'] - df['enemy_jungle_minions_killed_sum']
    log('Jungle CS difference is tracked separately from team_cs_diff')

    negative_team_lane = (df['team_minions_killed_sum'] < 0).sum()
    negative_enemy_lane = (df['enemy_minions_killed_sum'] < 0).sum()
    negative_team_jungle = (df['team_jungle_minions_killed_sum'] < 0).sum()
    negative_enemy_jungle = (df['enemy_jungle_minions_killed_sum'] < 0).sum()

    log(f'Negative team lane CS: {negative_team_lane:,}')
    log(f'Negative enemy lane CS: {negative_enemy_lane:,}')
    log(f'Negative team jungle CS: {negative_team_jungle:,}')
    log(f'Negative enemy jungle CS: {negative_enemy_jungle:,}')

    total_cs_diff = calculated_lane_cs_diff + calculated_jungle_cs_diff
    total_cs_diff_difference = total_cs_diff - df['team_cs_diff']
    rows_differing_from_total_cs = (total_cs_diff_difference != 0).sum()

    log(f'Rows where total (lane + jungle) CS differs from team_cs_diff: {rows_differing_from_total_cs:,}')
    if (
        lane_cs_mismatch == 0
        and negative_team_lane == 0
        and negative_enemy_lane == 0
        and negative_team_jungle == 0
        and negative_enemy_jungle == 0
    ):
        log('[PASSED] CS features are internally consistent')
    else:
        log('[FAILED] CS feature consistency problems detected')

def validate_player_team_relationship(df):
    log('')
    log('========== PLAYER / TEAM RELATIONSHIP ==========')
    required = [
        "player_total_gold",
        "player_current_gold",
        "player_xp",
        "player_minions_killed",
        "player_jungle_minions_killed",
        "team_total_gold_sum",
        "team_current_gold_sum",
        "team_xp_sum",
        "team_minions_killed_sum",
        "team_jungle_minions_killed_sum",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        log('[SKIPPED] Required player/team columns missing')
        for column in missing:
            log(f'  - {column}')
        return
    player_gold_exceeds_team = (df['player_total_gold'] > df['team_total_gold_sum']).sum()
    player_current_exceeds_team = (df['player_current_gold'] > df['team_current_gold_sum']).sum()
    player_xp_exceeds_team = (df['player_xp'] > df['team_xp_sum']).sum()
    player_lane_cs_exceeds_team = (df['player_minions_killed'] > df['team_minions_killed_sum']).sum()
    player_jungle_cs_exceeds_team = (
        df['player_jungle_minions_killed'] > df['team_jungle_minions_killed_sum']
    ).sum()

    log(f'Player total gold > team total gold: {player_gold_exceeds_team:,}')
    log(f'Player current gold > team current gold: {player_current_exceeds_team:,}')
    log(f'Player XP > team XP: {player_xp_exceeds_team:,}')
    log(f'Player lane CS > team lane CS: {player_lane_cs_exceeds_team:,}')
    log(f'Player jungle CS > team jungle CS: {player_jungle_cs_exceeds_team:,}')

    if (
        player_gold_exceeds_team == 0
        and player_current_exceeds_team == 0
        and player_xp_exceeds_team == 0
        and player_lane_cs_exceeds_team == 0
        and player_jungle_cs_exceeds_team == 0
    ):
        log('[PASSED] Player-level values do not exceed corresponding team totals')
    else:
        log('[FAILED] Player-level values exceed corresponding team totals')

def validate_team_aggregate_magnitudes(df):
    log('')
    log('========== TEAM AGGREGATE MAGNITUDES ==========')
    columns = [
        "team_total_gold_sum",
        "enemy_total_gold_sum",
        "team_xp_sum",
        "enemy_xp_sum",
        "team_minions_killed_sum",
        "enemy_minions_killed_sum",
        "team_jungle_minions_killed_sum",
        "enemy_jungle_minions_killed_sum",
    ]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        log('[SKIPPED] Required aggregate columns missing')
        for column in missing:
            log(f'  - {column}')
        return
    for column in columns:
        values = df[column]
        log('')
        log(column)
        log(values.describe().to_string())

def validate_outcome_namespace(df):
    log('')
    log('========== OUTCOME NAMESPACE CHECK ==========')
    outcome_columns = [column for column in df.columns if column.startswith('outcome_')]
    expected = {
        "outcome_win",
        "outcome_final_gold_earned",
        "outcome_final_gold_spent",
        "outcome_final_champ_level",
        "outcome_final_champ_experience",
        "outcome_final_kills",
        "outcome_final_deaths",
        "outcome_final_assists",
        "outcome_final_damage_dealt_to_champions",
        "outcome_final_damage_taken",
        "outcome_game_duration",
        "outcome_game_result",
    }
    actual = set(outcome_columns)
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)

    log(f'Outcome columns detected: {len(outcome_columns)}')
    if unexpected:
        log('Unexpected outcome columns:')
        for column in unexpected:
            log(f'  - {column}')
    if missing:
        log('Expected outcome columns missing:')
        for column in missing:
            log(f'  - {column}')
    if not unexpected and (not missing):
        log('[PASSED] Outcome columns match expected outcome namespace')

def validate_feature_names(df):
    log('')
    log('FEATURE NAMESPACE CHECK')
    derived_keywords = [
        "diff",
        "sum",
        "per_second",
        "per_minute",
        "ratio",
        "share",
        "rate",
    ]
    candidates = []
    for column in df.columns:
        lower = column.lower()
        if any((keyword in lower for keyword in derived_keywords)):
            candidates.append(column)
    log('Potentially derived columns:')
    if candidates:
        for column in candidates:
            log(f'  - {column}')
    else:
        log('  None detected')
    log('')

def main():
    log('FEATURE SEMANTICS VALIDATION')

    df = load_dataset()
    if df.empty:
        log('  Dataset is empty')
        return
    log(f'Research cases: {len(df):,}')
    log(f'Columns: {len(df.columns):,}')

    validate_team_gold(df)
    validate_xp(df)
    validate_cs(df)
    validate_player_team_relationship(df)
    validate_team_aggregate_magnitudes(df)
    validate_outcome_namespace(df)
    validate_feature_names(df)

    log('')
    log('===========================================')
    log('       FEATURE SEMANTICS CHECK COMPLETE')
    log('===========================================')

if __name__ == '__main__':
    main()