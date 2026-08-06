from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.research.AI.client import ask_model
from src.research.config import V2_MATCHING_DIR, V2_OUTCOME_DIR


OUTCOME_SUMMARY = V2_OUTCOME_DIR / "simple_outcome_summary.csv"
OUTCOME_GROUPS = V2_OUTCOME_DIR / "simple_outcome_by_group.csv"
NUMERIC_BALANCE = (
    V2_MATCHING_DIR
    / "balance"
    / "primary_variable_ratio_numeric_balance.csv"
)

PROMPT_FILE = V2_OUTCOME_DIR / "ai_interpretation_prompt.txt"
OUTPUT_FILE = V2_OUTCOME_DIR / "ai_interpretation.md"


def load_required_csv(filepath: Path) -> pd.DataFrame:
    if not filepath.exists():
        raise FileNotFoundError(
            f"Required analysis file not found: {filepath}"
        )

    try:
        return pd.read_csv(filepath)
    except Exception as error:
        raise RuntimeError(
            f"Could not read analysis file {filepath}: {error}"
        ) from error


def build_evidence() -> str:
    overall = load_required_csv(OUTCOME_SUMMARY)
    groups = load_required_csv(OUTCOME_GROUPS)
    balance = load_required_csv(NUMERIC_BALANCE)

    required_overall = {
        "group_type",
        "group",
        "matched_sets",
        "case_win_rate",
        "control_win_rate",
        "risk_difference",
        "risk_difference_ci_low",
        "risk_difference_ci_high",
    }

    required_groups = {
        "group_type",
        "group",
        "matched_sets",
        "case_win_rate",
        "control_win_rate",
        "risk_difference",
    }

    required_balance = {
        "covariate",
        "case_mean",
        "control_mean",
        "standardised_mean_difference",
        "absolute_smd",
        "balance_status",
    }

    checks = [
        ("overall outcome", overall, required_overall),
        ("group outcomes", groups, required_groups),
        ("numeric balance", balance, required_balance),
    ]

    for label, dataframe, required in checks:
        missing = sorted(required - set(dataframe.columns))

        if missing:
            raise ValueError(
                f"{label} file is missing columns: {missing}"
            )

    important_balance = (
        balance.sort_values(
            "absolute_smd",
            ascending=False,
        )
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

    return "\n\n".join(
        [
            "OVERALL MATCHED RESULT\n"
            + overall.to_csv(index=False),
            "MATCHED SUBGROUP RESULTS\n"
            + groups.to_csv(index=False),
            "TEN LARGEST PRIMARY BALANCE DIFFERENCES\n"
            + important_balance.to_csv(index=False),
        ]
    )


def build_prompt() -> str:
    evidence = build_evidence()

    return f"""
You are the grounded AI analyst inside a graduate data project about
Mejai's Soulstealer in League of Legends.

Use only the structured evidence supplied below. Do not invent numbers,
mechanics, sample details, or explanations that are not supported by the
evidence.

The analysis is a matched observational comparison. Never claim that buying
Mejai caused a win, loss, comeback, or snowball.

Main question:
Does Mejai appear mainly as a win-more purchase, or does it also appear in
viable comeback situations?

Interpretation rules:
- Report percentage-point differences accurately.
- Treat "team_state" and "player_state" as pre-purchase subgroup descriptions.
- Treat RETAINED versus SOLD as post-purchase lifecycle descriptions.
  Do not use that split to claim what caused the outcome.
- Explicitly state that recent kills, deaths, assists, and current gold retain
  some imbalance when the supplied balance table shows this.
- A positive matched difference is an association, not an item effect.
- Do not recommend further matching optimisation as the main conclusion.
  This is a moderate graduate project, not a publication-grade causal study.

Write a concise Markdown report with these sections:

# Executive answer

# Overall matched comparison

# Win-more versus comeback pattern
Discuss team state first, then player state.

# Purchase timing

# Retained and sold lifecycle description
Clearly label this as post-purchase and descriptive.

# Matching quality and limitations

# Final conclusion

# Questions the AI analyst could answer next
Give exactly three questions that can be answered from the existing structured
project outputs.

Structured evidence:
{evidence}
""".strip()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a grounded local-AI interpretation of the V2 results."
        )
    )

    parser.add_argument(
        "--prompt-only",
        action="store_true",
        help=(
            "Build and save the grounded prompt without calling Ollama."
        ),
    )

    args = parser.parse_args()

    V2_OUTCOME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prompt = build_prompt()
    PROMPT_FILE.write_text(
        prompt + "\n",
        encoding="utf-8",
    )

    print(f"[SAVED] {PROMPT_FILE}")

    if args.prompt_only:
        print(
            "[PASSED] GROUNDED AI PROMPT BUILT "
            "WITHOUT CALLING THE MODEL"
        )
        return

    try:
        response = ask_model(prompt)
    except Exception as error:
        raise RuntimeError(
            "The deterministic analysis is complete, but the local "
            "Ollama report could not run. Confirm that Ollama is running "
            "and that the model configured in "
            f"src/research/AI/client.py is installed. Original error: {error}"
        ) from error

    response = str(response).strip()

    if not response:
        raise ValueError(
            "The local model returned an empty response"
        )

    OUTPUT_FILE.write_text(
        response + "\n",
        encoding="utf-8",
    )

    print("")
    print(response)
    print("")
    print(f"[SAVED] {OUTPUT_FILE}")
    print("[PASSED] GROUNDED AI INTERPRETATION COMPLETED")


if __name__ == "__main__":
    main()