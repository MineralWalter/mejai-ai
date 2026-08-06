import argparse

import numpy as np
import pandas as pd

from src.research.AI.client import ask_model
from src.research.config import V2_MATCHING_DIR, V2_OUTCOME_DIR


PRIMARY_MATCHED_FILE = V2_MATCHING_DIR / "mejai_matched_primary.parquet"
NUMERIC_BALANCE_FILE = (
    V2_MATCHING_DIR
    / "balance"
    / "primary_variable_ratio_numeric_balance.csv"
)

OUTCOME_SUMMARY_FILE = V2_OUTCOME_DIR / "simple_outcome_summary.csv"
OUTCOME_GROUPS_FILE = V2_OUTCOME_DIR / "simple_outcome_by_group.csv"
MATCHED_SET_EFFECTS_FILE = V2_OUTCOME_DIR / "matched_set_effects.parquet"
DETERMINISTIC_REPORT_FILE = V2_OUTCOME_DIR / "simple_outcome_report.txt"
PROMPT_FILE = V2_OUTCOME_DIR / "ai_interpretation_prompt.txt"
AI_REPORT_FILE = V2_OUTCOME_DIR / "ai_interpretation.md"

TEAM_STATE_BINS = [-np.inf, -2_000, 2_000, np.inf]
PLAYER_STATE_BINS = [-np.inf, -1_000, 1_000, np.inf]
STATE_LABELS = ["behind", "close", "ahead"]


def normalise_outcome(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
                "win": 1,
                "loss": 0,
                "won": 1,
                "lost": 0,
            }
        )
    )


def load_primary_matched():
    if not PRIMARY_MATCHED_FILE.exists():
        raise FileNotFoundError(
            f"Primary matched dataset not found: {PRIMARY_MATCHED_FILE}\n"
            "Run:\n"
            "py -m src.research.build_matched_analysis_dataset"
        )

    matched = pd.read_parquet(PRIMARY_MATCHED_FILE, engine="pyarrow").copy()

    required = {
        "matched_set_id",
        "treatment",
        "matching_weight",
        "outcome_win",
        "observation_timestamp",
        "lifecycle_status",
        "team_total_gold_diff",
        "player_gold_diff_vs_role_opponent",
    }

    missing = sorted(required - set(matched.columns))
    if missing:
        raise ValueError(f"Primary matched dataset is missing columns: {missing}")

    matched["matched_set_id"] = matched["matched_set_id"].astype(str)
    matched["treatment"] = pd.to_numeric(matched["treatment"], errors="coerce")
    matched["matching_weight"] = pd.to_numeric(
        matched["matching_weight"],
        errors="coerce",
    )
    matched["outcome_numeric"] = normalise_outcome(matched["outcome_win"])

    numeric_columns = [
        "observation_timestamp",
        "team_total_gold_diff",
        "player_gold_diff_vs_role_opponent",
    ]

    for column in numeric_columns:
        matched[column] = pd.to_numeric(matched[column], errors="coerce")

    matched = matched.dropna(
        subset=[
            "matched_set_id",
            "treatment",
            "matching_weight",
            "outcome_numeric",
            "observation_timestamp",
            "team_total_gold_diff",
            "player_gold_diff_vs_role_opponent",
        ]
    ).copy()

    matched["treatment"] = matched["treatment"].astype(int)
    matched["outcome_numeric"] = matched["outcome_numeric"].astype(int)

    if not matched["treatment"].isin([0, 1]).all():
        raise ValueError("Treatment contains values outside 0 and 1")

    if not matched["outcome_numeric"].isin([0, 1]).all():
        raise ValueError("Outcome contains values outside 0 and 1")

    if (matched["matching_weight"] <= 0).any():
        raise ValueError("Matching weights must be positive")

    return matched.reset_index(drop=True)


def validate_matched_sets(matched):
    working = matched.copy()
    working["case_row"] = working["treatment"].eq(1).astype(int)
    working["control_row"] = working["treatment"].eq(0).astype(int)
    working["case_weight_component"] = (
        working["matching_weight"] * working["case_row"]
    )
    working["control_weight_component"] = (
        working["matching_weight"] * working["control_row"]
    )

    checks = (
        working.groupby("matched_set_id")
        .agg(
            case_rows=("case_row", "sum"),
            control_rows=("control_row", "sum"),
            case_weight=("case_weight_component", "sum"),
            control_weight=("control_weight_component", "sum"),
        )
    )

    if not checks["case_rows"].eq(1).all():
        raise ValueError("Each matched set must contain exactly one case")

    if not checks["control_rows"].between(1, 3, inclusive="both").all():
        raise ValueError("Each matched set must contain one to three controls")

    if not np.isclose(checks["case_weight"], 1.0).all():
        raise ValueError("Case weights do not sum to one per matched set")

    if not np.isclose(checks["control_weight"], 1.0).all():
        raise ValueError("Control weights do not sum to one per matched set")


def build_matched_set_effects(matched):
    cases = (
        matched[matched["treatment"] == 1]
        .set_index("matched_set_id")
        .copy()
    )

    if cases.index.duplicated().any():
        raise ValueError("More than one case row found in a matched set")

    controls = matched[matched["treatment"] == 0].copy()
    controls["weighted_outcome"] = (
        controls["outcome_numeric"] * controls["matching_weight"]
    )

    control_rates = (
        controls.groupby("matched_set_id")
        .agg(
            weighted_outcome_sum=("weighted_outcome", "sum"),
            control_weight_sum=("matching_weight", "sum"),
            selected_control_count=("matched_set_id", "size"),
        )
    )

    control_rates["control_win_rate"] = (
        control_rates["weighted_outcome_sum"]
        / control_rates["control_weight_sum"]
    )

    set_effects = cases.join(
        control_rates[
            [
                "control_win_rate",
                "control_weight_sum",
                "selected_control_count",
            ]
        ],
        how="inner",
    )

    if len(set_effects) != len(cases):
        raise ValueError("One or more case rows have no matched controls")

    set_effects["case_win"] = set_effects["outcome_numeric"].astype(int)
    set_effects["risk_difference"] = (
        set_effects["case_win"] - set_effects["control_win_rate"]
    )
    set_effects["observation_time_minutes"] = (
        set_effects["observation_timestamp"] / 60_000
    )

    return set_effects.reset_index()


def add_subgroups(set_effects):
    output = set_effects.copy()

    output["team_state"] = pd.cut(
        output["team_total_gold_diff"],
        bins=TEAM_STATE_BINS,
        labels=STATE_LABELS,
        right=False,
    )

    output["player_state"] = pd.cut(
        output["player_gold_diff_vs_role_opponent"],
        bins=PLAYER_STATE_BINS,
        labels=STATE_LABELS,
        right=False,
    )

    output["purchase_time_group"] = pd.cut(
        output["observation_time_minutes"],
        bins=[-np.inf, 15, 25, np.inf],
        labels=["before_15m", "15_to_25m", "after_25m"],
        right=False,
    )

    output["lifecycle_group"] = (
        output["lifecycle_status"].astype(str).str.strip().str.upper()
    )

    return output


def confidence_interval(values):
    values = pd.to_numeric(values, errors="coerce").dropna()

    if values.empty:
        return np.nan, np.nan, np.nan

    mean = float(values.mean())

    if len(values) < 2:
        return mean, np.nan, np.nan

    standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
    margin = 1.96 * standard_error

    return mean, mean - margin, mean + margin


def summarise_group(frame, group_type, group):
    risk_difference, ci_low, ci_high = confidence_interval(
        frame["risk_difference"]
    )

    return {
        "group_type": group_type,
        "group": group,
        "matched_sets": int(len(frame)),
        "case_win_rate": float(frame["case_win"].mean()),
        "control_win_rate": float(frame["control_win_rate"].mean()),
        "risk_difference": risk_difference,
        "risk_difference_ci_low": ci_low,
        "risk_difference_ci_high": ci_high,
    }


def build_outcome_summaries(set_effects):
    rows = [summarise_group(set_effects, "overall", "all")]

    for column in [
        "team_state",
        "player_state",
        "purchase_time_group",
        "lifecycle_group",
    ]:
        for value, group in set_effects.dropna(subset=[column]).groupby(
            column,
            observed=True,
            sort=False,
        ):
            rows.append(summarise_group(group, column, str(value)))

    summary = pd.DataFrame(rows)

    expected_counts = {
        "team_state": 3,
        "player_state": 3,
        "purchase_time_group": 3,
        "lifecycle_group": 2,
    }

    for group_type, expected in expected_counts.items():
        actual = int(summary["group_type"].eq(group_type).sum())
        if actual != expected:
            raise ValueError(
                f"Expected {expected} {group_type} rows but found {actual}"
            )

    return summary


def load_primary_balance():
    if not NUMERIC_BALANCE_FILE.exists():
        raise FileNotFoundError(
            f"Primary balance file not found: {NUMERIC_BALANCE_FILE}\n"
            "Run:\n"
            "py -m src.research.validate_matching_balance"
        )

    balance = pd.read_csv(NUMERIC_BALANCE_FILE)

    required = {
        "covariate",
        "case_mean",
        "control_mean",
        "standardised_mean_difference",
        "absolute_smd",
        "balance_status",
    }

    missing = sorted(required - set(balance.columns))
    if missing:
        raise ValueError(f"Primary balance file is missing columns: {missing}")

    return balance


def write_deterministic_report(summary):
    overall = summary[summary["group_type"] == "overall"].iloc[0]

    display = summary[
        [
            "group_type",
            "group",
            "matched_sets",
            "case_win_rate",
            "control_win_rate",
            "risk_difference",
            "risk_difference_ci_low",
            "risk_difference_ci_high",
        ]
    ]

    report = "\n".join(
        [
            "MEJAI MATCHED OUTCOME SUMMARY",
            "=" * 72,
            "",
            f"Matched sets: {int(overall['matched_sets']):,}",
            f"Case win rate: {overall['case_win_rate']:.2%}",
            f"Control win rate: {overall['control_win_rate']:.2%}",
            f"Matched difference: {overall['risk_difference']:+.2%}",
            (
                "Approximate 95% interval: "
                f"{overall['risk_difference_ci_low']:+.2%} to "
                f"{overall['risk_difference_ci_high']:+.2%}"
            ),
            "",
            (
                "These are observational matched associations. "
                "They are not estimates of a causal item effect."
            ),
            "",
            "Subgroup results:",
            display.to_string(index=False),
        ]
    )

    DETERMINISTIC_REPORT_FILE.write_text(
        report + "\n",
        encoding="utf-8",
    )


def save_deterministic_outputs(set_effects, summary):
    V2_OUTCOME_DIR.mkdir(parents=True, exist_ok=True)

    summary[summary["group_type"] == "overall"].to_csv(
        OUTCOME_SUMMARY_FILE,
        index=False,
    )
    summary.to_csv(OUTCOME_GROUPS_FILE, index=False)
    set_effects.to_parquet(
        MATCHED_SET_EFFECTS_FILE,
        index=False,
        engine="pyarrow",
    )
    write_deterministic_report(summary)

    for path in [
        OUTCOME_SUMMARY_FILE,
        OUTCOME_GROUPS_FILE,
        MATCHED_SET_EFFECTS_FILE,
        DETERMINISTIC_REPORT_FILE,
    ]:
        print(f"[SAVED] {path}")


def build_evidence(summary, balance):
    important_balance = (
        balance.sort_values("absolute_smd", ascending=False)
        .head(10)
        [
            [
                "covariate",
                "case_mean",
                "control_mean",
                "standardised_mean_difference",
                "absolute_smd",
                "balance_status",
            ]
        ]
    )

    overall = summary[summary["group_type"] == "overall"]
    subgroups = summary[summary["group_type"] != "overall"]

    return "\n\n".join(
        [
            "OVERALL MATCHED RESULT\n" + overall.to_csv(index=False),
            "MATCHED SUBGROUP RESULTS\n" + subgroups.to_csv(index=False),
            (
                "TEN LARGEST PRIMARY BALANCE DIFFERENCES\n"
                + important_balance.to_csv(index=False)
            ),
        ]
    )


def build_prompt(summary, balance):
    evidence = build_evidence(summary, balance)

    return f"""
You are the grounded AI interpretation layer inside a graduate data project
about Mejai's Soulstealer in League of Legends.

Python has already calculated and validated every number supplied below. Use
only that evidence. Do not recalculate values, exchange case and control
columns, invent mechanics, or invent explanations.

Main question:
Does the matched evidence suggest that Mejai is only associated with
already-ahead, win-more situations, or is the positive association also
present when the team or player is behind or close?

Important scope:
- These outputs compare matched outcomes conditional on pre-purchase state.
- They do not measure how frequently Mejai is purchased in each state.
- They do not prove that Mejai caused a win, comeback, snowball, benefit,
  improvement, loss, or any other outcome.

Mandatory wording rules:
- Use "association", "matched difference", or "observed pattern".
- Do not call a difference an "effect", "benefit", "impact", "effectiveness",
  or "improvement".
- Do not say one subgroup is statistically different from another subgroup.
  No pairwise subgroup test was performed.
- Report differences in percentage points.
- Treat team_state and player_state as pre-purchase descriptions.
- Treat RETAINED and SOLD as post-purchase lifecycle descriptions only.
- For lifecycle rows, copy case win rate, control win rate, and matched
  difference from their correct columns.
- Explicitly discuss remaining imbalance in recent kills, deaths, assists,
  and player current gold when shown by the balance evidence.
- Keep the deterministic Python outputs authoritative if interpretation is
  uncertain.
- Do not recommend more matching optimisation as the main conclusion.

Write only a concise finished Markdown report with exactly these headings:

# Executive answer

# Overall matched comparison

# Win-more versus comeback pattern

Discuss team state first, then player state.

# Purchase timing

# Retained and sold lifecycle description

Clearly label the lifecycle split as post-purchase and descriptive.

# Matching quality and limitations

# Final conclusion

# Questions the AI analyst could answer next

Give exactly these three grounded questions:

1. Which team-state subgroup has the largest matched win-rate difference?
2. How does the matched difference vary across the three purchase-time groups?
3. Which pre-purchase covariates have the largest remaining imbalance?

- Compare the numeric risk differences before identifying the largest or
  strongest subgroup. In the supplied team-state evidence, close is largest,
  behind is second, and ahead is smallest. Do not describe ahead as strongest.

- In the supplied player-state evidence, close is largest, ahead is second,
  and behind is smallest.

- Do not claim that purchase-timing groups are statistically different from
  one another. No pairwise timing comparison was performed. Only describe
  their numerical pattern.

- For lifecycle results, explicitly state that retaining Mejai did not
  necessarily cause wins and selling Mejai did not necessarily cause losses.
  Lifecycle status is post-purchase and may reflect subsequent game events.

- Avoid the phrases "statistically significant", "statistically distinct",
  "effective", "benefit", and "timing effect".

Structured evidence:
{evidence}

""".strip()



def validate_ai_report(report):
    report = str(report or "").strip()

    if not report:
        raise ValueError("The local model returned an empty report")

    required_headings = [
        "# Executive answer",
        "# Overall matched comparison",
        "# Win-more versus comeback pattern",
        "# Purchase timing",
        "# Retained and sold lifecycle description",
        "# Matching quality and limitations",
        "# Final conclusion",
        "# Questions the AI analyst could answer next",
    ]

    missing = [
        heading
        for heading in required_headings
        if heading not in report
    ]

    if missing:
        raise ValueError(
            "The AI report is missing required sections: "
            f"{missing}"
        )

    planning_phrases = [
        "we are given structured evidence",
        "let's write the report",
        "steps:",
        "draft:",
    ]

    lowered = report.lower()

    if any(
        phrase in lowered
        for phrase in planning_phrases
    ):
        raise ValueError(
            "The model returned planning text instead of "
            "only the finished report"
        )

    if (
        "observational" not in lowered
        and "association" not in lowered
    ):
        raise ValueError(
            "The AI report does not clearly describe the "
            "analysis as observational or associative"
        )

    return report

def calculate_deterministic_evidence():
    matched = load_primary_matched()
    validate_matched_sets(matched)

    set_effects = add_subgroups(build_matched_set_effects(matched))
    summary = build_outcome_summaries(set_effects)
    balance = load_primary_balance()

    save_deterministic_outputs(set_effects, summary)

    return summary, balance


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate deterministic matched outcomes and optionally generate "
            "a grounded local-AI interpretation."
        )
    )

    parser.add_argument(
        "--evidence-only",
        action="store_true",
        help="Calculate deterministic evidence without calling Ollama.",
    )
    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help="Calculate evidence and save the prompt without calling Ollama.",
    )

    args = parser.parse_args()

    print("=" * 72)
    print("CALCULATE AND INTERPRET MATCHED OUTCOMES")
    print("=" * 72)

    summary, balance = calculate_deterministic_evidence()

    print("")
    print(
        summary[
            [
                "group_type",
                "group",
                "matched_sets",
                "case_win_rate",
                "control_win_rate",
                "risk_difference",
            ]
        ].to_string(index=False)
    )

    if args.evidence_only:
        print("")
        print("[PASSED] DETERMINISTIC OUTCOME EVIDENCE COMPLETED")
        return

    prompt = build_prompt(summary, balance)
    PROMPT_FILE.write_text(prompt + "\n", encoding="utf-8")
    print("")
    print(f"[SAVED] {PROMPT_FILE}")

    if args.prompt_only:
        print(
            "[PASSED] GROUNDED AI PROMPT BUILT WITHOUT CALLING THE MODEL"
        )
        return

    try:
        response = ask_model(prompt)
    except Exception as error:
        raise RuntimeError(
            "The deterministic outcome analysis completed, but the local "
            "Ollama report could not run. Confirm that Ollama is running and "
            f"the configured model is installed. Original error: {error}"
        ) from error

    response = validate_ai_report(
        response
    )

    AI_REPORT_FILE.write_text(
        response + "\n",
        encoding="utf-8",
    )

    print("")
    print(response)
    print("")
    print(f"[SAVED] {AI_REPORT_FILE}")
    print("[PASSED] GROUNDED AI INTERPRETATION COMPLETED")


if __name__ == "__main__":
    main()