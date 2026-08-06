from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import norm



PRIMARY_FILE = Path(
    "data/analysis/mejai_matched_primary.parquet"
)

SENSITIVITY_FILE = Path(
    "data/analysis/mejai_matched_sensitivity.parquet"
)

OUTPUT_DIR = Path(
    "data/analysis/matched_outcome"
)

MODEL_SUMMARY_FILE = (
    OUTPUT_DIR / "matched_outcome_model_summary.csv"
)

STANDARDIZED_EFFECT_FILE = (
    OUTPUT_DIR / "matched_outcome_standardized_effects.csv"
)

RAW_EFFECT_FILE = (
    OUTPUT_DIR / "matched_outcome_raw_effects.csv"
)

TEXT_REPORT_FILE = (
    OUTPUT_DIR / "matched_outcome_report.txt"
)

JSON_REPORT_FILE = (
    OUTPUT_DIR / "matched_outcome_report.json"
)

CONTINUOUS_COVARIATES = [
    "player_total_gold",
    "player_level",
    "player_xp",
    "player_minions_killed",
    "player_jungle_minions_killed",
    "team_total_gold",
    "team_total_gold_diff",
    "team_xp",
    "team_xp_diff",
    "team_cs",
    "team_cs_diff",
]

CATEGORICAL_COVARIATES = [
    "team_position",
]

MIN_NON_MISSING_RATIO = 0.90

GRADIENT_STEP = 1e-5


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(message)


# ============================================================
# INPUT PREPARATION
# ============================================================

def normalise_boolean(series):
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(
            series,
            errors="coerce",
        )

        return numeric.map(
            {
                1: True,
                0: False,
                1.0: True,
                0.0: False,
            }
        ).astype("boolean")

    cleaned = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return cleaned.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "win": True,
            "loss": False,
            "won": True,
            "lost": False,
        }
    ).astype("boolean")


def load_matched_dataset(
    path,
    sample_name,
):
    if not path.exists():
        raise FileNotFoundError(
            f"{sample_name} dataset not found: "
            f"{path}"
        )

    df = pd.read_parquet(path)

    required = [
        "matched_set_id",
        "treatment",
        "outcome_win",
        "matching_weight",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{sample_name} dataset is missing "
            f"required columns: {missing}"
        )

    df = df.copy()

    df["matched_set_id"] = (
        df["matched_set_id"].astype(str)
    )

    df["treatment"] = pd.to_numeric(
        df["treatment"],
        errors="coerce",
    )

    df["matching_weight"] = pd.to_numeric(
        df["matching_weight"],
        errors="coerce",
    )

    df["outcome_win"] = normalise_boolean(
        df["outcome_win"]
    )

    df = df.dropna(
        subset=[
            "matched_set_id",
            "treatment",
            "matching_weight",
            "outcome_win",
        ]
    )

    df["treatment"] = (
        df["treatment"].astype(int)
    )

    df["outcome_numeric"] = (
        df["outcome_win"].astype(int)
    )

    if not set(
        df["treatment"].unique()
    ).issubset({0, 1}):
        raise ValueError(
            f"{sample_name}: treatment must "
            f"contain only 0 and 1"
        )

    if (
        df["matching_weight"] <= 0
    ).any():
        raise ValueError(
            f"{sample_name}: matching weights "
            f"must all be positive"
        )

    return df.reset_index(drop=True)


# ============================================================
# MATCHED-SAMPLE VALIDATION
# ============================================================

def validate_matched_sets(
    df,
    sample_name,
):
    summary = (
        df.groupby("matched_set_id")
        .agg(
            treatment_rows=(
                "treatment",
                "sum",
            ),
            total_rows=(
                "treatment",
                "size",
            ),
            total_weight=(
                "matching_weight",
                "sum",
            ),
        )
    )

    control_counts = (
        df[df["treatment"] == 0]
        .groupby("matched_set_id")
        .size()
    )

    missing_control_sets = (
        set(summary.index)
        - set(control_counts.index)
    )

    if (
        summary["treatment_rows"] != 1
    ).any():
        raise ValueError(
            f"{sample_name}: every matched set "
            f"must contain exactly one treatment row"
        )

    if missing_control_sets:
        raise ValueError(
            f"{sample_name}: one or more matched "
            f"sets contain no control rows"
        )

    if (
        ~np.isclose(
            summary["total_weight"],
            2.0,
        )
    ).any():
        raise ValueError(
            f"{sample_name}: matched-set weights "
            f"must sum to 2"
        )


# ============================================================
# RAW WEIGHTED EFFECT
# ============================================================

def weighted_mean(
    values,
    weights,
):
    return np.average(
        values.astype(float),
        weights=weights.astype(float),
    )


def calculate_raw_effect(
    df,
    sample_name,
):
    treated = df[
        df["treatment"] == 1
    ]

    controls = df[
        df["treatment"] == 0
    ]

    treated_rate = weighted_mean(
        treated["outcome_numeric"],
        treated["matching_weight"],
    )

    control_rate = weighted_mean(
        controls["outcome_numeric"],
        controls["matching_weight"],
    )

    risk_difference = (
        treated_rate
        - control_rate
    )

    risk_ratio = (
        treated_rate / control_rate
        if control_rate > 0
        else np.nan
    )

    return {
        "sample": sample_name,
        "matched_sets": (
            df["matched_set_id"].nunique()
        ),
        "treatment_rows": len(treated),
        "control_rows": len(controls),
        "weighted_treatment_win_rate": (
            treated_rate
        ),
        "weighted_control_win_rate": (
            control_rate
        ),
        "weighted_risk_difference": (
            risk_difference
        ),
        "weighted_risk_ratio": (
            risk_ratio
        ),
    }


# ============================================================
# MODEL FORMULAS
# ============================================================

def select_adjustment_covariates(df):
    continuous = []
    categorical = []

    for column in CONTINUOUS_COVARIATES:
        if column not in df.columns:
            continue

        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        non_missing_ratio = (
            numeric.notna().mean()
        )

        if (
            non_missing_ratio
            < MIN_NON_MISSING_RATIO
        ):
            continue

        if numeric.nunique(
            dropna=True
        ) <= 1:
            continue

        df[column] = numeric

        continuous.append(column)

    for column in CATEGORICAL_COVARIATES:
        if column not in df.columns:
            continue

        non_missing_ratio = (
            df[column].notna().mean()
        )

        if (
            non_missing_ratio
            < MIN_NON_MISSING_RATIO
        ):
            continue

        if df[column].nunique(
            dropna=True
        ) <= 1:
            continue

        df[column] = (
            df[column]
            .astype("category")
        )

        categorical.append(column)

    return continuous, categorical


def build_formula(
    continuous,
    categorical,
    adjusted,
):
    terms = ["treatment"]

    if adjusted:
        terms.extend(continuous)

        terms.extend(
            [
                f"C({column})"
                for column in categorical
            ]
        )

    return (
        "outcome_numeric ~ "
        + " + ".join(terms)
    )


# ============================================================
# MODEL FITTING
# ============================================================

def fit_weighted_logistic_model(
    df,
    formula,
):
    model_data = df.copy()

    formula_columns = [
        token
        for token in CONTINUOUS_COVARIATES
        if token in formula
        and token in model_data.columns
    ]

    categorical_columns = [
        column
        for column in CATEGORICAL_COVARIATES
        if f"C({column})" in formula
        and column in model_data.columns
    ]

    required_columns = [
        "outcome_numeric",
        "treatment",
        "matching_weight",
        "matched_set_id",
        *formula_columns,
        *categorical_columns,
    ]

    model_data = model_data.dropna(
        subset=required_columns
    ).copy()

    if model_data.empty:
        raise ValueError(
            "No complete rows remain for model fitting"
        )

    model = smf.glm(
        formula=formula,
        data=model_data,
        family=sm.families.Binomial(),
        freq_weights=(
            model_data["matching_weight"]
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        result = model.fit(
            cov_type="cluster",
            cov_kwds={
                "groups": (
                    model_data[
                        "matched_set_id"
                    ]
                ),
            },
        )

    return result, model_data


# ============================================================
# STANDARDIZED PROBABILITY EFFECT
# ============================================================

def weighted_standardized_difference(
    result,
    model_data,
    params=None,
):
    if params is None:
        params = result.params

    treated_data = model_data.copy()
    control_data = model_data.copy()

    treated_data["treatment"] = 1
    control_data["treatment"] = 0

    treated_design = result.model.data.design_info

    from patsy import build_design_matrices

    treated_matrix = build_design_matrices(
        [treated_design],
        treated_data,
        return_type="dataframe",
    )[0]

    control_matrix = build_design_matrices(
        [treated_design],
        control_data,
        return_type="dataframe",
    )[0]

    treated_linear = np.asarray(
        treated_matrix
    ) @ np.asarray(params)

    control_linear = np.asarray(
        control_matrix
    ) @ np.asarray(params)

    treated_probability = (
        1.0
        / (
            1.0
            + np.exp(
                -treated_linear
            )
        )
    )

    control_probability = (
        1.0
        / (
            1.0
            + np.exp(
                -control_linear
            )
        )
    )

    weights = model_data[
        "matching_weight"
    ].to_numpy(dtype=float)

    treated_mean = np.average(
        treated_probability,
        weights=weights,
    )

    control_mean = np.average(
        control_probability,
        weights=weights,
    )

    return (
        treated_mean,
        control_mean,
        treated_mean - control_mean,
    )


def numerical_gradient(
    result,
    model_data,
):
    base_params = result.params.to_numpy(
        dtype=float
    )

    gradient = np.zeros(
        len(base_params),
        dtype=float,
    )

    for index in range(
        len(base_params)
    ):
        step = (
            GRADIENT_STEP
            * max(
                1.0,
                abs(base_params[index]),
            )
        )

        upper = base_params.copy()
        lower = base_params.copy()

        upper[index] += step
        lower[index] -= step

        upper_effect = (
            weighted_standardized_difference(
                result,
                model_data,
                params=upper,
            )[2]
        )

        lower_effect = (
            weighted_standardized_difference(
                result,
                model_data,
                params=lower,
            )[2]
        )

        gradient[index] = (
            upper_effect
            - lower_effect
        ) / (
            2.0 * step
        )

    return gradient


def standardized_effect_with_ci(
    result,
    model_data,
):
    (
        treated_probability,
        control_probability,
        risk_difference,
    ) = weighted_standardized_difference(
        result,
        model_data,
    )

    gradient = numerical_gradient(
        result,
        model_data,
    )

    covariance = np.asarray(
        result.cov_params()
    )

    variance = float(
        gradient.T
        @ covariance
        @ gradient
    )

    standard_error = np.sqrt(
        max(variance, 0.0)
    )

    lower = (
        risk_difference
        - 1.96 * standard_error
    )

    upper = (
        risk_difference
        + 1.96 * standard_error
    )

    z_value = (
        risk_difference
        / standard_error
        if standard_error > 0
        else np.nan
    )

    if np.isnan(z_value):
        p_value = np.nan
    else:
        p_value = float(
            2.0
            * (
                1.0
                - norm.cdf(
                    abs(z_value)
                )
            )
        )

    return {
        "standardized_treatment_probability": (
            treated_probability
        ),
        "standardized_control_probability": (
            control_probability
        ),
        "standardized_risk_difference": (
            risk_difference
        ),
        "risk_difference_standard_error": (
            standard_error
        ),
        "risk_difference_ci_lower": lower,
        "risk_difference_ci_upper": upper,
        "risk_difference_p_value": p_value,
    }


# ============================================================
# MODEL RESULT EXTRACTION
# ============================================================

def extract_treatment_result(
    result,
    sample_name,
    model_name,
    formula,
    rows_used,
    matched_sets_used,
):
    coefficient = float(
        result.params["treatment"]
    )

    standard_error = float(
        result.bse["treatment"]
    )

    p_value = float(
        result.pvalues["treatment"]
    )

    confidence_interval = (
        result.conf_int()
        .loc["treatment"]
    )

    lower_log_odds = float(
        confidence_interval.iloc[0]
    )

    upper_log_odds = float(
        confidence_interval.iloc[1]
    )

    return {
        "sample": sample_name,
        "model": model_name,
        "formula": formula,
        "rows_used": rows_used,
        "matched_sets_used": (
            matched_sets_used
        ),
        "treatment_log_odds": (
            coefficient
        ),
        "cluster_robust_standard_error": (
            standard_error
        ),
        "treatment_p_value": (
            p_value
        ),
        "treatment_odds_ratio": (
            np.exp(coefficient)
        ),
        "odds_ratio_ci_lower": (
            np.exp(lower_log_odds)
        ),
        "odds_ratio_ci_upper": (
            np.exp(upper_log_odds)
        ),
        "aic": float(result.aic),
        "deviance": float(
            result.deviance
        ),
    }


# ============================================================
# SAMPLE ANALYSIS
# ============================================================

def analyse_sample(
    df,
    sample_name,
):
    validate_matched_sets(
        df,
        sample_name,
    )

    raw_result = calculate_raw_effect(
        df,
        sample_name,
    )

    continuous, categorical = (
        select_adjustment_covariates(
            df
        )
    )

    log("")
    log(
        f"{sample_name}: adjustment "
        f"covariates"
    )

    log(
        f"Continuous: "
        f"{continuous}"
    )

    log(
        f"Categorical: "
        f"{categorical}"
    )

    model_rows = []
    effect_rows = []
    fitted_models = {}

    specifications = [
        (
            "treatment_only",
            False,
        ),
        (
            "adjusted",
            True,
        ),
    ]

    for model_name, adjusted in (
        specifications
    ):
        formula = build_formula(
            continuous,
            categorical,
            adjusted,
        )

        result, model_data = (
            fit_weighted_logistic_model(
                df,
                formula,
            )
        )

        fitted_models[model_name] = (
            result
        )

        model_rows.append(
            extract_treatment_result(
                result=result,
                sample_name=sample_name,
                model_name=model_name,
                formula=formula,
                rows_used=len(model_data),
                matched_sets_used=(
                    model_data[
                        "matched_set_id"
                    ].nunique()
                ),
            )
        )

        standardized = (
            standardized_effect_with_ci(
                result,
                model_data,
            )
        )

        standardized.update(
            {
                "sample": sample_name,
                "model": model_name,
                "formula": formula,
                "rows_used": len(
                    model_data
                ),
                "matched_sets_used": (
                    model_data[
                        "matched_set_id"
                    ].nunique()
                ),
            }
        )

        effect_rows.append(
            standardized
        )

    return (
        raw_result,
        model_rows,
        effect_rows,
        fitted_models,
    )


# ============================================================
# REPORTING
# ============================================================

def percentage(value):
    if pd.isna(value):
        return "NA"

    return f"{value:.2%}"


def format_report(
    raw_df,
    model_df,
    effect_df,
):
    lines = []

    lines.append(
        "=" * 76
    )

    lines.append(
        "MEJAI MATCHED OUTCOME ANALYSIS"
    )

    lines.append(
        "=" * 76
    )

    for sample_name in (
        raw_df["sample"].unique()
    ):
        raw = raw_df[
            raw_df["sample"]
            == sample_name
        ].iloc[0]

        lines.append("")
        lines.append(
            sample_name.upper()
        )

        lines.append(
            "-" * 76
        )

        lines.append(
            f"Matched sets: "
            f"{int(raw['matched_sets']):,}"
        )

        lines.append(
            f"Weighted treatment win rate: "
            f"{percentage(raw['weighted_treatment_win_rate'])}"
        )

        lines.append(
            f"Weighted control win rate: "
            f"{percentage(raw['weighted_control_win_rate'])}"
        )

        lines.append(
            f"Raw weighted risk difference: "
            f"{percentage(raw['weighted_risk_difference'])}"
        )

        sample_models = model_df[
            model_df["sample"]
            == sample_name
        ]

        sample_effects = effect_df[
            effect_df["sample"]
            == sample_name
        ]

        for _, model_row in (
            sample_models.iterrows()
        ):
            model_name = (
                model_row["model"]
            )

            effect_row = sample_effects[
                sample_effects["model"]
                == model_name
            ].iloc[0]

            lines.append("")
            lines.append(
                f"Model: {model_name}"
            )

            lines.append(
                f"Formula: "
                f"{model_row['formula']}"
            )

            lines.append(
                f"Treatment odds ratio: "
                f"{model_row['treatment_odds_ratio']:.4f}"
            )

            lines.append(
                f"95% OR CI: "
                f"[{model_row['odds_ratio_ci_lower']:.4f}, "
                f"{model_row['odds_ratio_ci_upper']:.4f}]"
            )

            lines.append(
                f"Treatment p-value: "
                f"{model_row['treatment_p_value']:.6f}"
            )

            lines.append(
                f"Standardized treatment probability: "
                f"{percentage(effect_row['standardized_treatment_probability'])}"
            )

            lines.append(
                f"Standardized control probability: "
                f"{percentage(effect_row['standardized_control_probability'])}"
            )

            lines.append(
                f"Standardized risk difference: "
                f"{percentage(effect_row['standardized_risk_difference'])}"
            )

            lines.append(
                f"95% RD CI: "
                f"[{percentage(effect_row['risk_difference_ci_lower'])}, "
                f"{percentage(effect_row['risk_difference_ci_upper'])}]"
            )

            lines.append(
                f"Risk-difference p-value: "
                f"{effect_row['risk_difference_p_value']:.6f}"
            )

    return "\n".join(lines)


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    raw_df,
    model_df,
    effect_df,
    report,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_df.to_csv(
        RAW_EFFECT_FILE,
        index=False,
    )

    model_df.to_csv(
        MODEL_SUMMARY_FILE,
        index=False,
    )

    effect_df.to_csv(
        STANDARDIZED_EFFECT_FILE,
        index=False,
    )

    TEXT_REPORT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    json_payload = {
        "raw_effects": (
            raw_df.replace(
                {
                    np.nan: None,
                }
            )
            .to_dict(
                orient="records"
            )
        ),
        "models": (
            model_df.replace(
                {
                    np.nan: None,
                }
            )
            .to_dict(
                orient="records"
            )
        ),
        "standardized_effects": (
            effect_df.replace(
                {
                    np.nan: None,
                }
            )
            .to_dict(
                orient="records"
            )
        ),
    }

    JSON_REPORT_FILE.write_text(
        json.dumps(
            json_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    log("")
    log(
        f"[SAVED] {RAW_EFFECT_FILE}"
    )

    log(
        f"[SAVED] {MODEL_SUMMARY_FILE}"
    )

    log(
        f"[SAVED] {STANDARDIZED_EFFECT_FILE}"
    )

    log(
        f"[SAVED] {TEXT_REPORT_FILE}"
    )

    log(
        f"[SAVED] {JSON_REPORT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    log("=" * 76)
    log("MEJAI MATCHED OUTCOME MODEL")
    log("=" * 76)

    samples = [
        (
            "primary",
            PRIMARY_FILE,
        ),
        (
            "sensitivity",
            SENSITIVITY_FILE,
        ),
    ]

    raw_rows = []
    model_rows = []
    effect_rows = []

    for sample_name, path in samples:
        log("")
        log(
            f"Loading {sample_name} dataset..."
        )

        df = load_matched_dataset(
            path,
            sample_name,
        )

        log(
            f"Rows loaded: {len(df):,}"
        )

        log(
            f"Matched sets: "
            f"{df['matched_set_id'].nunique():,}"
        )

        (
            raw_result,
            sample_model_rows,
            sample_effect_rows,
            _,
        ) = analyse_sample(
            df,
            sample_name,
        )

        raw_rows.append(
            raw_result
        )

        model_rows.extend(
            sample_model_rows
        )

        effect_rows.extend(
            sample_effect_rows
        )

    raw_df = pd.DataFrame(
        raw_rows
    )

    model_df = pd.DataFrame(
        model_rows
    )

    effect_df = pd.DataFrame(
        effect_rows
    )

    report = format_report(
        raw_df,
        model_df,
        effect_df,
    )

    log("")
    log(report)

    save_outputs(
        raw_df,
        model_df,
        effect_df,
        report,
    )

    log("")
    log(
        "[PASSED] MATCHED OUTCOME "
        "MODELS COMPLETE"
    )


if __name__ == "__main__":
    main()