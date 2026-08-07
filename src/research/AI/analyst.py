import json

import numpy as np
import pandas as pd

from src.research.AI.client import ask_model
from src.research.config import V2_OUTCOME_DIR


MATCHED_SET_EFFECTS_FILE = V2_OUTCOME_DIR / "matched_set_effects.parquet"
OUTCOME_GROUPS_FILE = V2_OUTCOME_DIR / "simple_outcome_by_group.csv"
AI_EXPLORATION_DIR = V2_OUTCOME_DIR / "ai_exploration"
DECISION_FILE = AI_EXPLORATION_DIR / "analyst_decision.json"

MIN_MATCHED_SETS = 200

ALLOWED_GROUPS = {
    "team_state",
    "player_state",
    "purchase_time_group",
}

ALLOWED_PAIRS = {
    frozenset({"team_state", "player_state"}),
    frozenset({"team_state", "purchase_time_group"}),
    frozenset({"player_state", "purchase_time_group"}),
}


def load_matched_set_effects():
    if not MATCHED_SET_EFFECTS_FILE.exists():
        raise FileNotFoundError(
            "Matched-set effects file not found:\n"
            f"{MATCHED_SET_EFFECTS_FILE}\n\n"
            "Run:\n"
            "py -m src.research.AI.generate_report --evidence-only"
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

    missing = sorted(required_columns - set(effects.columns))
    if missing:
        raise ValueError(
            "Matched-set effects file is missing required columns: "
            f"{missing}"
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

    if effects[numeric_columns].isna().any(axis=None):
        raise ValueError(
            "Matched-set effects contain missing numeric outcome values"
        )

    return effects.reset_index(drop=True)


def load_primary_summary():
    if not OUTCOME_GROUPS_FILE.exists():
        raise FileNotFoundError(
            "Primary outcome summary not found:\n"
            f"{OUTCOME_GROUPS_FILE}\n\n"
            "Run:\n"
            "py -m src.research.AI.generate_report --evidence-only"
        )

    summary = pd.read_csv(OUTCOME_GROUPS_FILE)

    required = {
        "group_type",
        "group",
        "matched_sets",
        "risk_difference",
    }

    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(
            "Primary outcome summary is missing required columns: "
            f"{missing}"
        )

    return summary


def confidence_interval(values):
    values = pd.to_numeric(values, errors="coerce").dropna()
    mean = float(values.mean())

    if len(values) < 2:
        return mean, np.nan, np.nan

    standard_error = float(
        values.std(ddof=1) / np.sqrt(len(values))
    )
    margin = 1.96 * standard_error

    return mean, mean - margin, mean + margin


def calculate_interaction(effects, group_a, group_b):
    if group_a not in ALLOWED_GROUPS:
        raise ValueError(f"Unsupported group: {group_a}")

    if group_b not in ALLOWED_GROUPS:
        raise ValueError(f"Unsupported group: {group_b}")

    pair = frozenset({group_a, group_b})
    if pair not in ALLOWED_PAIRS:
        raise ValueError(
            "Unsupported interaction pair: "
            f"{group_a} and {group_b}"
        )

    rows = []

    grouped = (
        effects.dropna(subset=[group_a, group_b])
        .groupby(
            [group_a, group_b],
            observed=True,
            sort=False,
        )
    )

    for values, frame in grouped:
        if len(frame) < MIN_MATCHED_SETS:
            continue

        risk_difference, ci_low, ci_high = confidence_interval(
            frame["risk_difference"]
        )

        rows.append(
            {
                group_a: str(values[0]),
                group_b: str(values[1]),
                "matched_sets": int(len(frame)),
                "case_win_rate": float(frame["case_win"].mean()),
                "control_win_rate": float(
                    frame["control_win_rate"].mean()
                ),
                "risk_difference": risk_difference,
                "risk_difference_ci_low": ci_low,
                "risk_difference_ci_high": ci_high,
            }
        )

    output = pd.DataFrame(rows)

    if output.empty:
        raise ValueError(
            "No interaction groups met the minimum of "
            f"{MIN_MATCHED_SETS} matched sets"
        )

    return output.sort_values(
        "risk_difference",
        ascending=False,
    ).reset_index(drop=True)


def build_decision_prompt(summary):
    evidence = summary[
        [
            "group_type",
            "group",
            "matched_sets",
            "risk_difference",
        ]
    ].to_csv(index=False)

    return f"""
You are directing one bounded exploratory analysis after a frozen primary
matched study of Mejai's Soulstealer.

Choose exactly one exploratory interaction that would help investigate an
unanswered question raised by the primary results.

Allowed choices:
- team_state with player_state
- team_state with purchase_time_group
- player_state with purchase_time_group

- Your reason must use only the primary results shown below.
- Do not claim anything about the interaction result before it has been run.
- Explain why the interaction is worth investigating, not what you expect it
  to show.
- Do not use "effect", "impact", "modulates", "causes", or similar causal
  language.
- Phrase uncertainty explicitly, such as "This would test whether..." or
  "This is worth examining because...".

Return only JSON in exactly this shape:
{{
  "action": "interaction",
  "group_a": "team_state",
  "group_b": "purchase_time_group",
  "reason": "One concise evidence-based reason."
}}

Primary results:
{evidence}
""".strip()


def parse_model_json(response):
    text = str(response or "").strip()

    if text.startswith("```"):
        lines = text.splitlines()

        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            "The model did not return valid JSON:\n"
            f"{text}"
        ) from error


def validate_decision(decision):
    required = {
        "action",
        "group_a",
        "group_b",
        "reason",
    }

    missing = sorted(required - set(decision))
    if missing:
        raise ValueError(
            "AI decision is missing fields: "
            f"{missing}"
        )

    if decision["action"] != "interaction":
        raise ValueError(
            "Only the interaction action is allowed in this step"
        )

    group_a = str(decision["group_a"]).strip()
    group_b = str(decision["group_b"]).strip()

    if frozenset({group_a, group_b}) not in ALLOWED_PAIRS:
        raise ValueError(
            "AI selected an unsupported pair: "
            f"{group_a} and {group_b}"
        )
    
    reason = str(decision["reason"]).strip()
    
    forbidden_reason_terms = [
        "effect",
        "impact",
        "modulates",
        "causes",
        "caused",
    ]

    lowered_reason = reason.lower()

    if any(term in lowered_reason for term in forbidden_reason_terms):
        raise ValueError("AI decision reason uses unsupported"
                         "causal or interaction-effect language" )
    
    if not reason:
        raise ValueError("AI decision reason is empty")

    return {
        "action": "interaction",
        "group_a": group_a,
        "group_b": group_b,
        "reason": reason,
    }


def safe_output_name(group_a, group_b):
    return f"{group_a}_by_{group_b}.csv"


def main(): 
    print("=" * 72)
    print("AI ANALYST — MODEL-DIRECTED EXPLORATION")
    print("=" * 72)

    effects = load_matched_set_effects()
    summary = load_primary_summary()

    prompt = build_decision_prompt(summary)

    print("")
    print("[AI] Choosing one approved exploratory interaction...")

    response = ask_model(prompt)
    decision = validate_decision(parse_model_json(response))

    print(
        "[AI] Selected: "
        f"{decision['group_a']} × {decision['group_b']}"
    )
    print(f"[AI] Reason: {decision['reason']}")

    result = calculate_interaction(
        effects,
        group_a=decision["group_a"],
        group_b=decision["group_b"],
    )

    AI_EXPLORATION_DIR.mkdir(parents=True, exist_ok=True)

    result_file = (
        AI_EXPLORATION_DIR
        / safe_output_name(
            decision["group_a"],
            decision["group_b"],
        )
    )

    decision_record = {
        **decision,
        "minimum_matched_sets": MIN_MATCHED_SETS,
        "result_file": str(result_file),
    }

    DECISION_FILE.write_text(
        json.dumps(decision_record, indent=2) + "\n",
        encoding="utf-8",
    )

    result.to_csv(result_file, index=False)

    print("")
    print(result.to_string(index=False))
    print("")
    print(f"[SAVED] {DECISION_FILE}")
    print(f"[SAVED] {result_file}")
    print(
        "[PASSED] AI SELECTED AND PYTHON EXECUTED "
        "ONE EXPLORATORY ANALYSIS"
    )


if __name__ == "__main__":
    main()