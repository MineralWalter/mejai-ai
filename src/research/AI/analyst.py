import json

import numpy as np
import pandas as pd

from src.research.AI.client import ask_model
from src.research.config import V2_OUTCOME_DIR


MATCHED_SET_EFFECTS_FILE = V2_OUTCOME_DIR / "matched_set_effects.parquet"
OUTCOME_GROUPS_FILE = V2_OUTCOME_DIR / "simple_outcome_by_group.csv"

AI_EXPLORATION_DIR = V2_OUTCOME_DIR / "ai_exploration"
TRACE_FILE = AI_EXPLORATION_DIR / "analyst_trace.json"
EXPLORATORY_MEMO_FILE = AI_EXPLORATION_DIR / "ai_exploratory_memo.md"

MIN_MATCHED_SETS = 200
MAX_ANALYSIS_STEPS = 2

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

    effects = pd.read_parquet(MATCHED_SET_EFFECTS_FILE, engine="pyarrow").copy()

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

    numeric_columns = ["case_win", "control_win_rate", "risk_difference"]

    for column in numeric_columns:
        effects[column] = pd.to_numeric(effects[column], errors="coerce")

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

    required_columns = {
        "group_type",
        "group",
        "matched_sets",
        "risk_difference",
    }

    missing = sorted(required_columns - set(summary.columns))
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

    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    margin = 1.96 * standard_error

    return mean, mean - margin, mean + margin


def calculate_joint_subgroups(effects, group_a, group_b):
    if group_a not in ALLOWED_GROUPS:
        raise ValueError(f"Unsupported group: {group_a}")

    if group_b not in ALLOWED_GROUPS:
        raise ValueError(f"Unsupported group: {group_b}")

    pair = frozenset({group_a, group_b})
    if pair not in ALLOWED_PAIRS:
        raise ValueError(
            "Unsupported joint subgroup pair: "
            f"{group_a} and {group_b}"
        )

    rows = []

    grouped = (
        effects.dropna(subset=[group_a, group_b])
        .groupby([group_a, group_b], observed=True, sort=False)
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
                "control_win_rate": float(frame["control_win_rate"].mean()),
                "risk_difference": risk_difference,
                "risk_difference_ci_low": ci_low,
                "risk_difference_ci_high": ci_high,
            }
        )

    output = pd.DataFrame(rows)

    if output.empty:
        raise ValueError(
            "No joint subgroup combinations met the minimum of "
            f"{MIN_MATCHED_SETS} matched sets"
        )

    return output.sort_values(
        "risk_difference",
        ascending=False,
    ).reset_index(drop=True)


def primary_evidence_text(summary):
    return summary[
        ["group_type", "group", "matched_sets", "risk_difference"]
    ].to_csv(index=False)


def previous_results_text(trace):
    completed = [
        step
        for step in trace
        if step.get("action") == "joint_subgroup"
    ]

    if not completed:
        return "No exploratory joint subgroup analysis has been run yet."

    sections = []

    for step in completed:
        sections.append(
            "\n".join(
                [
                    f"STEP {step['step']}: {step['group_a']} with {step['group_b']}",
                    f"Reason: {step['reason']}",
                    "Deterministic result:",
                    step["result_csv"],
                ]
            )
        )

    return "\n\n".join(sections)


def build_decision_prompt(summary, trace, step_number):
    completed_pairs = [
        f"- {step['group_a']} with {step['group_b']}"
        for step in trace
        if step.get("action") == "joint_subgroup"
    ]

    completed_text = "\n".join(completed_pairs) if completed_pairs else "- none"

    if step_number == 1:
        action_rules = """
Choose exactly one approved joint subgroup analysis.

Return only JSON in this form:
{
  "action": "joint_subgroup",
  "group_a": "team_state",
  "group_b": "purchase_time_group",
  "reason": "One concise evidence-based reason."
}
""".strip()
    else:
        action_rules = """
Choose either one new approved joint subgroup analysis or finish.

For another joint subgroup analysis, return only JSON in this form:
{
  "action": "joint_subgroup",
  "group_a": "team_state",
  "group_b": "purchase_time_group",
  "reason": "One concise evidence-based reason."
}

To finish, return only JSON in this form:
{
  "action": "finish",
  "reason": "One concise evidence-based reason."
}
""".strip()

    return f"""
You are directing a bounded exploratory analysis after a frozen primary
matched study of Mejai's Soulstealer.

The primary analysis is complete. You may only choose a small number of
exploratory follow-up questions using descriptive joint subgroup breakdowns.

Allowed joint subgroup pairs:
- team_state with player_state
- team_state with purchase_time_group
- player_state with purchase_time_group

Already completed joint subgroup pairs:
{completed_text}

Rules:
- Use only the evidence supplied below.
- Do not repeat a completed joint subgroup analysis.
- Do not claim a joint subgroup result before Python calculates it.
- Explain why a question is worth investigating, not what you expect it to show.
- Do not use "effect", "impact", "influence", "modulates", "causes", or similar causal language.
- Do not describe exploratory findings as primary findings.
- Do not search for the most favourable result.
- Do not speculate about player behaviour, motivation, psychology, or decision-making.
- The available evidence contains matched outcome patterns only.
- Return JSON only.

{action_rules}

Primary results:
{primary_evidence_text(summary)}

Previous exploratory evidence:
{previous_results_text(trace)}
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


def validate_reason(reason):
    reason = str(reason).strip()

    if not reason:
        raise ValueError("AI decision reason is empty")

    if len(reason) > 600:
        raise ValueError("AI decision reason is unexpectedly long")

    return reason


def validate_decision(decision, trace, step_number):
    action = str(decision.get("action", "")).strip()

    if action == "finish":
        if step_number == 1:
            raise ValueError(
                "The AI must run at least one exploratory "
                "joint subgroup analysis before finishing"
            )

        return {
            "action": "finish",
            "reason": validate_reason(decision.get("reason", "")),
        }

    if action != "joint_subgroup":
        raise ValueError(
            "AI action must be either 'joint_subgroup' or 'finish'"
        )

    required_fields = {"group_a", "group_b", "reason"}
    missing = sorted(required_fields - set(decision))

    if missing:
        raise ValueError(
            "AI decision is missing fields: "
            f"{missing}"
        )

    group_a = str(decision["group_a"]).strip()
    group_b = str(decision["group_b"]).strip()
    pair = frozenset({group_a, group_b})

    if pair not in ALLOWED_PAIRS:
        raise ValueError(
            "AI selected an unsupported pair: "
            f"{group_a} and {group_b}"
        )

    completed_pairs = {
        frozenset({step["group_a"], step["group_b"]})
        for step in trace
        if step.get("action") == "joint_subgroup"
    }

    if pair in completed_pairs:
        raise ValueError(
            "AI selected a joint subgroup analysis that "
            "has already been completed"
        )

    return {
        "action": "joint_subgroup",
        "group_a": group_a,
        "group_b": group_b,
        "reason": validate_reason(decision["reason"]),
    }


def safe_output_name(step_number, group_a, group_b):
    return f"step_{step_number}_{group_a}_by_{group_b}.csv"


def save_trace(trace):
    AI_EXPLORATION_DIR.mkdir(parents=True, exist_ok=True)

    TRACE_FILE.write_text(
        json.dumps(trace, indent=2) + "\n",
        encoding="utf-8",
    )


def build_final_memo_prompt(summary, trace):
    primary = summary[
        ["group_type", "group", "matched_sets", "risk_difference"]
    ].to_csv(index=False)

    exploration = json.dumps(trace, indent=2)

    return f"""
You are writing a concise exploratory memo for the Mejai's Soulstealer analysis.

The primary matched analysis was completed before the AI-guided exploration.
The primary findings remain authoritative.

The AI then selected up to two bounded exploratory joint subgroup analyses.
Python performed every calculation deterministically.

Write a concise Markdown memo using exactly these headings:

# Primary research context

# AI-selected exploratory path

Explain which analyses the AI selected and why.

# Exploratory findings

Describe the deterministic joint subgroup results in percentage points.

# Relationship to the primary finding

Explain whether the exploratory evidence adds nuance to the original
win-more-versus-comeback question.

# Limitations

State clearly that:
- these analyses were selected after viewing the primary results;
- they are exploratory rather than primary findings;
- subgroup comparisons are observational;
- no causal interpretation is justified;
- groups below the minimum support threshold were excluded.

Do not use "effect", "impact", "influence", "caused", or similar causal wording.
Do not speculate about player psychology, motivation, behaviour, or decision-making.
Do not claim a formal statistical interaction, moderation, or subgroup difference test was performed.
The outputs are descriptive joint subgroup matched differences only.

Primary evidence:
{primary}

AI exploration trace:
{exploration}
""".strip()


def validate_final_memo(memo):
    memo = str(memo or "").strip()

    if not memo:
        raise ValueError("AI returned an empty exploratory memo")

    required_headings = [
        "# Primary research context",
        "# AI-selected exploratory path",
        "# Exploratory findings",
        "# Relationship to the primary finding",
        "# Limitations",
    ]

    missing = [
        heading
        for heading in required_headings
        if heading not in memo
    ]

    if missing:
        raise ValueError(
            "Exploratory memo is missing required headings: "
            f"{missing}"
        )

    review_phrases = [
        "significantly influence",
        "interaction effect",
        "modulates",
        "key driver",
        "caused",
        "causes",
    ]

    lowered = memo.lower()

    flagged = [
        phrase
        for phrase in review_phrases
        if phrase in lowered
    ]

    if flagged:
        print("[AI CHECK] Memo completed, but review wording manually: "+ ", ".join(flagged))

    return memo


def main():
    print("=" * 72)
    print("AI ANALYST — BOUNDED TWO-STEP EXPLORATION")
    print("=" * 72)

    effects = load_matched_set_effects()
    summary = load_primary_summary()

    AI_EXPLORATION_DIR.mkdir(parents=True, exist_ok=True)

    trace = []

    for step_number in range(1, MAX_ANALYSIS_STEPS + 1):
        print("")
        print(
            f"[AI] Choosing exploratory joint subgroup step "
            f"{step_number}/{MAX_ANALYSIS_STEPS}..."
        )

        prompt = build_decision_prompt(summary, trace, step_number)
        response = ask_model(prompt)

        decision = validate_decision(parse_model_json(response),trace,step_number,)

        if decision["action"] == "finish":
            print("[AI] Finished exploration.")
            print(f"[AI] Reason: {decision['reason']}")

            trace.append({"step": step_number,**decision,})
            save_trace(trace)
            break

        print("[AI] Selected: "f"{decision['group_a']} × {decision['group_b']}")
        print(f"[AI] Reason: {decision['reason']}")

        result = calculate_joint_subgroups(
            effects,
            decision["group_a"],
            decision["group_b"],
        )

        result_file = AI_EXPLORATION_DIR / safe_output_name(
            step_number,
            decision["group_a"],
            decision["group_b"],
        )

        result.to_csv(result_file, index=False)

        trace.append(
            {
                "step": step_number,
                **decision,
                "minimum_matched_sets": MIN_MATCHED_SETS,
                "result_file": str(result_file),
                "result_csv": result.to_csv(index=False),
            }
        )

        save_trace(trace)

        print("")
        print(result.to_string(index=False))
        print("")
        print(f"[SAVED] {result_file}")

    memo_prompt = build_final_memo_prompt(summary, trace)

    print("")
    print("[AI] Synthesising exploratory findings...")

    memo = validate_final_memo(ask_model(memo_prompt))

    EXPLORATORY_MEMO_FILE.write_text(memo + "\n",encoding="utf-8",)

    print("")
    print(memo)
    print("")
    print(f"[SAVED] {TRACE_FILE}")
    print(f"[SAVED] {EXPLORATORY_MEMO_FILE}")
    print("[PASSED] BOUNDED AI-GUIDED JOINT SUBGROUP EXPLORATION COMPLETED")


if __name__ == "__main__":
    main()