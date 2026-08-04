from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIG
# ============================================================

DATA_FILE = Path(
    "data/analysis/mejai_research_dataset.parquet"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(message)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    log("=" * 70)
    log("MEJAI LOGISTIC REGRESSION BASELINE")
    log("=" * 70)

    log("")
    log("[1] Loading research dataset...")

    if not DATA_FILE.exists():
        log(f"[ERROR] Dataset not found: {DATA_FILE}")
        return pd.DataFrame()

    df = pd.read_parquet(DATA_FILE)

    log(f"Rows: {len(df):,}")
    log(f"Columns: {len(df.columns):,}")

    return df


# ============================================================
# PREPARE ANALYSIS DATA
# ============================================================

def prepare_data(df):
    log("")
    log("[2] Preparing completed Mejai decisions...")

    # UNDONE is not a completed decision.
    df = df[
        df["lifecycle_status"].isin(
            ["RETAINED", "SOLD"]
        )
    ].copy()

    # RETAINED = 1
    # SOLD = 0
    df["outcome"] = (
        df["outcome_win"]
        .astype(int)
    )

    log(
        f"Completed decisions: "
        f"{len(df):,}"
    )

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
        f"{df['outcome'].mean() * 100:.2f}%"
    )

    return df


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

# These are all known to describe the state at or before
# the Mejai purchase.
#
# Deliberately excluded:
# - outcome_final_*
# - outcome_game_duration
# - outcome_game_result
# - lifecycle_status
# - outcome_win
#
# Those contain information that occurs after the purchase
# or directly defines the target.

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
# FEATURE CHECK
# ============================================================

def check_features(df):
    log("")
    log("[3] Checking candidate features...")

    available_numeric = [
        column
        for column in NUMERIC_FEATURES
        if column in df.columns
    ]

    available_categorical = [
        column
        for column in CATEGORICAL_FEATURES
        if column in df.columns
    ]

    missing_numeric = [
        column
        for column in NUMERIC_FEATURES
        if column not in df.columns
    ]

    missing_categorical = [
        column
        for column in CATEGORICAL_FEATURES
        if column not in df.columns
    ]

    log("")
    log("Numeric features:")
    for column in available_numeric:
        log(f"  OK   {column}")

    log("")
    log("Categorical features:")
    for column in available_categorical:
        log(f"  OK   {column}")

    if missing_numeric:
        log("")
        log("Missing numeric features:")
        for column in missing_numeric:
            log(f"  MISS {column}")

    if missing_categorical:
        log("")
        log("Missing categorical features:")
        for column in missing_categorical:
            log(f"  MISS {column}")

    features = (
        available_numeric
        + available_categorical
    )

    if not features:
        log("[ERROR] No usable features found.")
        return []

    return features


# ============================================================
# LEAKAGE CHECK
# ============================================================

def check_for_leakage(df, features):
    log("")
    log("[4] Checking for obvious outcome leakage...")

    forbidden_terms = [
        "outcome_final_",
        "outcome_game_",
        "outcome_win",
        "lifecycle_status",
    ]

    leakage_columns = []

    for column in features:
        if any(
            term in column
            for term in forbidden_terms
        ):
            leakage_columns.append(column)

    if leakage_columns:
        log("[ERROR] Possible leakage detected:")

        for column in leakage_columns:
            log(f"  {column}")

        return False

    log(
        "[PASS] Candidate features contain "
        "no obvious post-purchase outcome fields."
    )

    return True


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

def split_data(df, features):
    log("")
    log("[5] Creating grouped train/test split...")

    X = df[features].copy()
    y = df["outcome"].copy()

    # Group by match so cases from the same game cannot
    # appear in both training and testing.
    groups = df["match_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
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

    log(
        f"Training cases: {len(X_train):,}"
    )

    log(
        f"Testing cases:  {len(X_test):,}"
    )

    log(
        f"Training matches: "
        f"{groups.iloc[train_index].nunique():,}"
    )

    log(
        f"Testing matches:  "
        f"{groups.iloc[test_index].nunique():,}"
    )

    log("")
    log(
        f"Training win rate: "
        f"{y_train.mean() * 100:.2f}%"
    )

    log(
        f"Testing win rate:  "
        f"{y_test.mean() * 100:.2f}%"
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

def build_model(features):
    numeric_features = [
        column
        for column in NUMERIC_FEATURES
        if column in features
    ]

    categorical_features = [
        column
        for column in CATEGORICAL_FEATURES
        if column in features
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):
    log("")
    log("[7] Evaluating model...")
    log("")

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

    log(
        f"Accuracy: {accuracy:.4f}"
    )

    log(
        f"ROC-AUC:  {roc_auc:.4f}"
    )

    log("")
    log("Confusion matrix:")

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    print(matrix)

    log("")
    log("Classification report:")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Loss",
                "Win",
            ],
        )
    )

    return accuracy, roc_auc


# ============================================================
# COEFFICIENTS
# ============================================================

def show_coefficients(model):
    log("")
    log("[8] Logistic regression coefficients...")
    log("")
    log(
        "Positive coefficient = associated "
        "with higher predicted win probability."
    )
    log(
        "Negative coefficient = associated "
        "with lower predicted win probability."
    )

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    coefficients = classifier.coef_[0]

    coefficient_table = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
        }
    )

    coefficient_table[
        "abs_coefficient"
    ] = coefficient_table[
        "coefficient"
    ].abs()

    coefficient_table = (
        coefficient_table
        .sort_values(
            "abs_coefficient",
            ascending=False,
        )
    )

    log("")
    log("Strongest coefficients:")

    print(
        coefficient_table[
            [
                "feature",
                "coefficient",
            ]
        ].head(25).to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_data()

    if df.empty:
        return

    df = prepare_data(df)

    if df.empty:
        log("[ERROR] No completed decisions available.")
        return

    features = check_features(df)

    if not features:
        return

    if not check_for_leakage(
        df,
        features,
    ):
        return

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(
        df,
        features,
    )

    log("")
    log("[6] Training logistic regression...")

    model = build_model(features)

    model.fit(
        X_train,
        y_train,
    )

    log("[PASS] Model trained.")

    evaluate_model(
        model,
        X_test,
        y_test,
    )

    show_coefficients(model)

    log("")
    log("=" * 70)
    log("BASELINE MODEL COMPLETE")
    log("=" * 70)


if __name__ == "__main__":
    main()