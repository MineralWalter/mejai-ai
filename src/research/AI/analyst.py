from pathlib import Path

import numpy as np
import pandas as pd

from src.research.config import V2_OUTCOME_DIR


MATCHED_SET_EFFECTS_FILE = (
    V2_OUTCOME_DIR
    / "matched_set_effects.parquet"
)

AI_EXPLORATION_DIR = (
    V2_OUTCOME_DIR
    / "ai_exploration"
)

MIN_MATCHED_SETS = 200

ALLOWED_GROUPS = {
    "team_state",
    "player_state",
    "purchase_time_group",
}


def load_matched_set_effects():
    if not MATCHED_SET_EFFECTS_FILE.exists():
        raise FileNotFoundError(
            "Matched-set effects file not found:\n"
            f"{MATCHED_SET_EFFECTS_FILE}\n\n"
            "Run:\n"
            "py -m src.research.AI.generate_report "
            "--evidence-only"
        )

    effects = pd.read_parquet(
        MATCHED_SET_EFFECTS_FILE,
        engine="pyarrow",
    ).copy()

    required_columns = {
        "matched_set_id",
        "case_win",
        "control_win_rate",
        "risk_difference",
        *ALLOWED_GROUPS,
    }

    missing = sorted(
        required_columns
        - set(effects.columns)
    )

    if missing:
        raise ValueError(
            "Matched-set effects file is missing "
            f"required columns: {missing}"
        )

    numeric_columns = [
        "case_win",
        "control_win_rate",
        "risk_difference",
    ]

    for column in numeric_columns:
        effects[column] = pd.to_numeric(
            effects[column],
            errors="coerce",
        )

    if effects[
        numeric_columns
    ].isna().any(axis=None):
        raise ValueError(
            "Matched-set effects contain missing "
            "numeric outcome values"
        )

    return effects.reset_index(drop=True)


def confidence_interval(values):
    values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    mean = float(values.mean())

    if len(values) < 2:
        return mean, np.nan, np.nan

    standard_error = float(
        values.std(ddof=1)
        / np.sqrt(len(values))
    )

    margin = 1.96 * standard_error

    return (
        mean,
        mean - margin,
        mean + margin,
    )


def calculate_interaction(
    effects,
    group_a,
    group_b,
):
    if group_a not in ALLOWED_GROUPS:
        raise ValueError(
            f"Unsupported group: {group_a}"
        )

    if group_b not in ALLOWED_GROUPS:
        raise ValueError(
            f"Unsupported group: {group_b}"
        )

    if group_a == group_b:
        raise ValueError(
            "Interaction groups must differ"
        )

    rows = []

    grouped = (
        effects.dropna(
            subset=[group_a, group_b]
        )
        .groupby(
            [group_a, group_b],
            observed=True,
            sort=False,
        )
    )

    for values, frame in grouped:
        if len(frame) < MIN_MATCHED_SETS:
            continue

        (
            risk_difference,
            ci_low,
            ci_high,
        ) = confidence_interval(
            frame["risk_difference"]
        )

        rows.append(
            {
                group_a: str(values[0]),
                group_b: str(values[1]),
                "matched_sets": int(len(frame)),
                "case_win_rate": float(
                    frame["case_win"].mean()
                ),
                "control_win_rate": float(
                    frame[
                        "control_win_rate"
                    ].mean()
                ),
                "risk_difference": (
                    risk_difference
                ),
                "risk_difference_ci_low": (
                    ci_low
                ),
                "risk_difference_ci_high": (
                    ci_high
                ),
            }
        )

    output = pd.DataFrame(rows)

    if output.empty:
        raise ValueError(
            "No interaction groups met the "
            f"minimum of {MIN_MATCHED_SETS} "
            "matched sets"
        )

    return output.sort_values(
        "risk_difference",
        ascending=False,
    ).reset_index(drop=True)


def main():
    print("=" * 72)
    print("AI ANALYST — DETERMINISTIC TOOL TEST")
    print("=" * 72)

    effects = load_matched_set_effects()

    result = calculate_interaction(
        effects,
        group_a="team_state",
        group_b="purchase_time_group",
    )

    AI_EXPLORATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        AI_EXPLORATION_DIR
        / "team_state_by_purchase_time.csv"
    )

    result.to_csv(
        output_file,
        index=False,
    )

    print("")
    print(result.to_string(index=False))
    print("")
    print(f"[SAVED] {output_file}")
    print("[PASSED] DETERMINISTIC ANALYST TOOL COMPLETED")


if __name__ == "__main__":
    main()