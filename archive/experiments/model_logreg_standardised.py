from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/analysis/mejai_research_dataset.parquet")

TARGET = "outcome_win"
GROUP_COLUMN = "match_id"

NUMERIC_FEATURES = [
    "purchase_time_seconds",
    "player_current_gold",
    "player_total_gold",
    "player_level",
    "player_xp",
    "player_minions_killed",
    "player_jungle_minions_killed",
    "team_current_gold_diff",
    "team_total_gold_diff",
    "team_xp_diff",
    "team_cs_diff",
]

CATEGORICAL_FEATURES = [
    "champion_name",
    "team_position",
    "game_version",
]


# ============================================================
# LOGGING
# ============================================================

def log(message=""):
    print(message)


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():
    log("=" * 75)
    log("MEJAI STANDARDISED LOGISTIC REGRESSION")
    log("=" * 75)

    log("")
    log("[1] Loading research dataset...")

    if not DATA_PATH.exists():
        log(f"[ERROR] Dataset not found: {DATA_PATH}")
        return pd.DataFrame()

    df = pd.read_parquet(DATA_PATH)

    log(f"Rows:    {len(df):,}")
    log(f"Columns: {len(df.columns):,}")

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):
    log("")
    log("[2] Preparing completed Mejai decisions...")

    # UNDONE purchases are excluded because they do not represent
    # a completed Mejai decision.
    df = df[
        df["lifecycle_status"].isin(["RETAINED", "SOLD"])
    ].copy()

    log(f"Completed decisions: {len(df):,}")

    log("")
    log("Lifecycle:")
    log(
        df["lifecycle_status"]
        .value_counts()
        .to_string()
    )

    log("")
    log(
        f"Overall win rate: "
        f"{df[TARGET].mean():.2%}"
    )

    return df


# ============================================================
# FEATURE CHECKS
# ============================================================

def check_features(df):
    log("")
    log("[3] Checking candidate features...")

    all_features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    missing_columns = [
        feature
        for feature in all_features + [TARGET, GROUP_COLUMN]
        if feature not in df.columns
    ]

    if missing_columns:
        log(
            "[ERROR] Missing columns:"
            f" {missing_columns}"
        )
        return False

    log("")
    log("Numeric features:")

    for feature in NUMERIC_FEATURES:
        missing = df[feature].isna().sum()

        if missing:
            log(
                f"WARNING {feature}: "
                f"{missing:,} missing"
            )
        else:
            log(f"OK   {feature}")

    log("")
    log("Categorical features:")

    for feature in CATEGORICAL_FEATURES:
        missing = df[feature].isna().sum()

        if missing:
            log(
                f"WARNING {feature}: "
                f"{missing:,} missing"
            )
        else:
            log(f"OK   {feature}")

    return True


# ============================================================
# LEAKAGE CHECK
# ============================================================

def check_leakage(df):
    log("")
    log("[4] Checking for obvious outcome leakage...")

    suspicious = [
        column
        for column in (
            NUMERIC_FEATURES
            + CATEGORICAL_FEATURES
        )
        if column.startswith("outcome_")
    ]

    if suspicious:
        log(
            "[FAIL] Outcome fields found:"
            f" {suspicious}"
        )
        return False

    log(
        "[PASS] Candidate features contain "
        "no obvious post-purchase outcome fields."
    )

    return True


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def create_grouped_split(df):
    log("")
    log("[5] Creating grouped train/test split...")

    X = df[
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    ]

    y = df[TARGET].astype(int)

    groups = df[GROUP_COLUMN]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_index, test_index = next(
        splitter.split(
            X,
            y,
            groups=groups,
        )
    )

    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    groups_train = groups.iloc[train_index]
    groups_test = groups.iloc[test_index]

    log(
        f"Training cases: {len(X_train):,}"
    )

    log(
        f"Testing cases:  {len(X_test):,}"
    )

    log(
        f"Training matches: "
        f"{groups_train.nunique():,}"
    )

    log(
        f"Testing matches:  "
        f"{groups_test.nunique():,}"
    )

    log("")
    log(
        f"Training win rate: "
        f"{y_train.mean():.2%}"
    )

    log(
        f"Testing win rate:  "
        f"{y_test.mean():.2%}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================
# MODEL
# ============================================================

def build_model():
    log("")
    log("[6] Building standardised logistic regression...")

    # IMPORTANT:
    #
    # StandardScaler is applied ONLY to numeric predictors.
    #
    # Categorical variables are one-hot encoded but NOT
    # standardised.
    #
    # The scaler is fitted only on training data because it is
    # inside the Pipeline / ColumnTransformer.

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "logistic_regression",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )

    return model


# ============================================================
# TRAIN
# ============================================================

def train_model(
    model,
    X_train,
    y_train,
):
    log("")
    log("[7] Training standardised logistic regression...")

    model.fit(
        X_train,
        y_train,
    )

    log("[PASS] Model trained.")

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
):
    log("")
    log("[8] Evaluating model...")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    log("")
    log(
        f"Accuracy: {accuracy:.4f}"
    )

    log(
        f"ROC-AUC:  {roc_auc:.4f}"
    )

    log("")
    log("Confusion matrix:")

    log(
        str(
            confusion_matrix(
                y_test,
                predictions,
            )
        )
    )

    log("")
    log("Classification report:")

    log(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Loss",
                "Win",
            ],
        )
    )

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
    }


# ============================================================
# COEFFICIENTS
# ============================================================

def print_coefficients(model):

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "logistic_regression"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    coefficients = classifier.coef_[0]

    coefficient_table = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
    })

    coefficient_table[
        "absolute_coefficient"
    ] = coefficient_table[
        "coefficient"
    ].abs()

    coefficient_table = (
        coefficient_table
        .sort_values(
            "absolute_coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    log("")
    log("Strongest coefficients:")

    log(
        coefficient_table[
            [
                "feature",
                "coefficient",
            ]
        ]
        .head(25)
        .to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_dataset()

    if df.empty:
        return

    df = prepare_data(df)

    if df.empty:
        log(
            "[ERROR] No completed Mejai decisions found."
        )
        return

    if not check_features(df):
        return

    if not check_leakage(df):
        return

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = create_grouped_split(df)

    model = build_model()

    model = train_model(
        model,
        X_train,
        y_train,
    )

    evaluate_model(
        model,
        X_test,
        y_test,
    )

    print_coefficients(model)

    log("")
    log("=" * 75)
    log("STANDARDISED LOGISTIC REGRESSION COMPLETE")
    log("=" * 75)


if __name__ == "__main__":
    main()