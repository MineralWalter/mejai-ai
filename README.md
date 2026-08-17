# Analysing High-Risk, High-Reward Game Decisions Using Matched Game States

This project investigates whether **Mejai's Soulstealer** is mainly a "win-more" item, or whether positive associations also appear when players buy it from close or losing game states.

I built a multi-region Riot API pipeline to collect match and timeline data, reconstruct Mejai purchase events, engineer purchase-time features, and compare purchases with similar non-purchase situations using matched observational analysis. The project also includes balance and robustness checks and a local LLM that selects from predefined exploratory analyses, while Python performs the calculations.

> **Important:** this is an observational study. Reported differences are associations within matched comparisons and should not be interpreted as causal effects of purchasing Mejai's Soulstealer.
---

## Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Research Design](#research-design)
- [Primary Matching Strategy](#primary-matching-strategy)
- [Primary Results](#primary-results)
- [Balance and Robustness](#balance-and-robustness)
- [AI-Native Exploratory Analysis](#ai-native-exploratory-analysis)
- [Exploratory Results](#exploratory-results)
- [Interpretation](#interpretation)
- [Limitations](#limitations)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data Collection and Processing](#data-collection-and-processing)
- [Reproducing the Analysis](#reproducing-the-analysis)
- [Tracked Outputs](#tracked-outputs)
- [Tech Stack](#technical-stack)
- [Takeaway](#final-takeaway)

---
## Project Overview

### End-to-end pipeline

```text
Riot API
   |
   v
Match and timeline collection
   |
   v
Parquet datasets
   |
   v
Mejai purchase reconstruction
   |
   v
Purchase-time feature engineering
   |
   v
Matched analysis
   |
   +--> balance checks
   +--> sensitivity and robustness checks
   |
   v
Outcome analysis
   |
   v
Local LLM exploratory analysis
```

### Design principles

The project has three main parts:

1. **Data collection and preparation**: collect match and timeline data from the Riot API and reconstruct Mejai purchase events.
2. **Matched analysis**: compare Mejai purchases with similar non-purchase situations using purchase-time game state and player features.
3. **LLM-guided exploration**: after the main analysis, a local model selects from a small set of predefined subgroup analyses while Python performs the calculations.
---

## Dataset

### Match-level scale

| Stage | Count |
| --- | ---: |
| Total collected matches | 80,824 |
| Eligible valid matches | 80,821 |
| Reconstructed Mejai purchase events | 34,251 |
| Reconstructed purchase lifecycles | 26,350 |
| Primary eligible Mejai purchases | 25,416 |

The dataset focuses on **high-MMR Ranked Solo/Duo** games from Vietnam, EU West, North America, and Korea. Match collection began from seed accounts in each region and expanded through players found in collected matches.

### Purchase lifecycle reconstruction

Mejai purchases were reconstructed from timeline events rather than inferred only from final inventories.

| Lifecycle status | Count |
| --- | ---: |
| `RETAINED` | 21,196 |
| `SOLD` | 4,220 |
| `UNDONE` | 934 |
| **Total** | **26,350** |

The primary analysis includes `RETAINED` and `SOLD` purchases and excludes `UNDONE` purchases, leaving **25,416 eligible purchases**.

`UNDONE` purchases are excluded because they represent immediately reversed transactions rather than completed purchase decisions.

---
## Research Design

### Unit of analysis

The analysis focuses on the **Mejai purchase event**, rather than treating the whole match as a single observation.

For each eligible purchase, the pipeline reconstructs the game state around the purchase time. It then looks for similar situations where a comparable player **did not purchase Mejai**.

### Matching features

Matching uses purchase-time information including:

- game time;
- player level, gold, XP, and CS;
- gold and XP difference versus the same-role opponent;
- team gold and XP state;
- recent kills, deaths, and assists;
- prior Dark Seal ownership;
- region; and
- team position.

Champion identity is also checked as part of the balance diagnostics.

Recent combat features use a five-minute lookback window.

### Why matching?

Players are more likely to buy Mejai when the game is already going well, and those situations are also more likely to end in a win.

A simple win-rate comparison would therefore mix the effect of the purchase with the game state in which the purchase was made.

Matching reduces this problem by comparing Mejai purchases with similar non-purchase situations based on observed game state. It improves comparability, but does not remove all possible confounding.

---

## Primary Matching Strategy

The primary analysis uses **variable-ratio matching with up to three controls per Mejai purchase**.

### Matching restrictions

Controls must have:

- the same region;
- the same team position;
- the same prior Dark Seal status; and
- no more than **750 gold** difference in current gold.

Other game-state and recent-combat features are used to rank eligible controls. Each case can have up to **3 controls**, with control weights summing to 1 within each matched set.

### Matched sample

| Metric | Primary result |
| --- | ---: |
| Eligible Mejai cases | 25,416 |
| Matched cases | 20,991 |
| Matching coverage | 82.59% |
| Selected control rows | 34,128 |
| Unique control observations | 31,647 |
| Controls reused | 2,191 |
| Maximum control reuse | 5 |
| Mean current-gold gap | 318.4 |
| Maximum current-gold gap | 750 |

All **34,128** selected controls matched the case on prior Dark Seal status.

### Outcome calculation

For each matched set:

```text
matched win-rate difference
    = case win
    - weighted control win rate
```

The overall result is the average of these matched-set differences, so cases with more controls do not receive more weight.

---

## Primary Results

### Overall matched result

The final primary analysis contains **20,991 matched Mejai purchase cases**.

| Group | Win rate |
| --- | ---: |
| Mejai purchase cases | 77.28% |
| Weighted matched controls | 73.20% |
| **Matched win-rate difference** | **+4.08 percentage points** |

Approximate 95% confidence interval:

```text
+3.43 pp to +4.72 pp
```

The result is positive in the final matched sample, but remains an **observational association**.

### Team state

| Team state | Matched sets | Matched win-rate difference |
| --- | ---: | ---: |
| Close | 4,999 | **+10.42 pp** |
| Ahead | 13,983 | **+1.21 pp** |
| Behind | 2,009 | **+8.28 pp** |

The largest matched differences appear in close and behind team states rather than in the already-ahead group. This argues against a simple "Mejai only looks good because it is bought while already far ahead" explanation, but it does not show that buying Mejai causes a comeback.

### Player state

| Player state | Matched sets | Matched win-rate difference |
| --- | ---: | ---: |
| Close | 5,830 | **+5.03 pp** |
| Ahead | 13,610 | **+3.82 pp** |
| Behind | 1,551 | **+2.75 pp** |

The matched difference remains positive across all three player-state groups.

### Purchase timing

| Purchase timing | Matched sets | Matched win-rate difference |
| --- | ---: | ---: |
| Before 15 minutes | 4,364 | **+4.61 pp** |
| 15-25 minutes | 11,198 | **+4.31 pp** |
| After 25 minutes | 5,429 | **+3.17 pp** |

The positive matched association is not isolated to a single purchase-time window.

---

## Balance and Robustness

Matching quality was evaluated separately from the outcome analysis.

### Primary balance

| Diagnostic | Result |
| --- | ---: |
| Matched sets | 20,991 |
| Selected control rows | 34,128 |
| Control effective sample size | 28,050.5 |
| Median absolute SMD | 0.0185 |
| Maximum absolute SMD | 0.3615 |
| Covariates with \|SMD\| < 0.10 | 14 |
| Covariates with 0.10 <= \|SMD\| < 0.20 | 0 |
| Covariates with \|SMD\| >= 0.20 | 4 |

Exact within-set mismatch counts for **region**, **team position**, and **prior Dark Seal status** were all zero.

### Residual imbalance

Four numeric covariates retained larger standardized mean differences:

| Covariate | SMD |
| --- | ---: |
| Deaths in previous 5 minutes | -0.361 |
| Kills in previous 5 minutes | +0.295 |
| Player current gold | +0.260 |
| Assists in previous 5 minutes | +0.207 |

These residual imbalances are important limitations. The matched groups are not perfectly balanced on every measured feature.

### Sensitivity and robustness checks

| Specification | Matched cases | Coverage | Matched difference |
| --- | ---: | ---: | ---: |
| Primary variable ratio | 20,991 | 82.59% | **+4.08 pp** |
| Relaxed variable ratio | 24,939 | 98.12% | **+3.35 pp** |
| Primary 1:1 | 20,991 | 82.59% | **+4.16 pp** |

The relaxed specification increases coverage but permits Dark Seal fallback matching, so it is treated as a sensitivity analysis rather than the primary design.

The 1:1 robustness result is close to the primary variable-ratio estimate.

---
## LLM-Guided Exploratory Analysis

The main matched analysis was completed before the LLM exploration.

The project uses a local `qwen3:4b-instruct` model through Ollama. Rather than generating statistics itself, the model selects from a small set of predefined subgroup analyses and Python performs the calculations.

### Available analyses

The model can choose from:

```text
team_state × player_state
team_state × purchase_time_group
player_state × purchase_time_group
```

The model can select up to two exploratory analyses.

For each step:

1. the model reviews the main analysis results;
2. it selects one available subgroup analysis;
3. Python calculates the result; and
4. the result is returned to the model before it decides whether to continue.

The model cannot run arbitrary code or change the matching design. This keeps the exploratory choices separate from the statistical calculations.

---

## Exploratory Results

These are **descriptive subgroup analyses**, not formal tests of interaction between variables.

Only cells with at least **200 matched sets** are included.

### Step 1 — Team State × Purchase Timing

| Team state | Purchase timing | Matched sets | Matched difference |
| --- | --- | ---: | ---: |
| Close | 15-25 min | 2,330 | **+11.69 pp** |
| Close | Before 15 min | 1,417 | **+11.37 pp** |
| Behind | 15-25 min | 927 | **+9.89 pp** |
| Behind | After 25 min | 923 | **+8.63 pp** |
| Close | After 25 min | 1,252 | **+6.96 pp** |
| Ahead | Before 15 min | 2,788 | **+1.61 pp** |
| Ahead | 15-25 min | 7,941 | **+1.50 pp** |
| Ahead | After 25 min | 3,254 | **+0.15 pp** |

The behind-before-15 group did not meet the minimum sample threshold.

### Step 2 — Player State × Purchase Timing

| Player state | Purchase timing | Matched sets | Matched difference |
| --- | --- | ---: | ---: |
| Close | Before 15 min | 1,215 | **+5.54 pp** |
| Close | 15-25 min | 3,002 | **+5.21 pp** |
| Ahead | Before 15 min | 3,101 | **+4.32 pp** |
| Close | After 25 min | 1,613 | **+4.30 pp** |
| Ahead | 15-25 min | 7,507 | **+4.10 pp** |
| Behind | After 25 min | 814 | **+2.97 pp** |
| Behind | 15-25 min | 689 | **+2.71 pp** |
| Ahead | After 25 min | 3,002 | **+2.61 pp** |

The behind-before-15 group did not meet the minimum sample threshold.

These results describe patterns within subgroups. They do not show that the differences between subgroups are statistically significant.

---

## Interpretation

The primary matched analysis finds a positive association between Mejai purchases and winning:

```text
Overall: +4.08 pp
```

By team state:

```text
Close:  +10.42 pp
Behind:  +8.28 pp
Ahead:   +1.21 pp
```

The largest matched differences appear in close and behind team states rather than in the already-ahead group. This suggests that the observed Mejai association is not explained only by players buying it while already far ahead.

The exploratory subgroup results show a similar pattern across several purchase-time windows.

However, this remains an observational analysis. The results do not show that buying Mejai causes a higher chance of winning, and unmeasured factors such as player skill, champion choice, team composition, and decision context may still affect both the purchase and the outcome.

### Lifecycle outcomes

| Lifecycle | Matched sets | Matched difference |
| --- | ---: | ---: |
| Retained | 17,507 | +8.34 pp |
| Sold | 3,484 | -17.36 pp |

These results are descriptive only. Whether Mejai is retained or sold happens after the original purchase and may depend on how the match develops.

---

## Limitations

### Observational design

Matching improves comparability on observed variables but cannot eliminate unmeasured confounding.

### Residual imbalance

Several recent-combat and gold variables remain meaningfully imbalanced after matching.

### Purchase selection

Mejai is not purchased randomly. The purchase may depend on player skill, champion identity, team composition, perceived risk, recall timing, or other unmeasured information.

### Matching coverage

The strict primary design matches **82.59%** of eligible cases. Unmatched cases may differ systematically from matched cases.

### High-MMR collection strategy

The dataset focuses on high-MMR Ranked Solo/Duo games. Results may not generalize directly to lower ranks, other queues, coordinated teams, or professional competition.

### Exploratory subgroup selection

The LLM-guided subgroup analyses were selected after the primary results were available, so they are treated as exploratory.

### No formal interaction test

The subgroup tables are descriptive. The project does not claim statistically significant differences between subgroup cells.

### Lifecycle is post-purchase

`RETAINED` and `SOLD` describe what happened after purchase and cannot be interpreted as if lifecycle status were assigned at purchase time.

---

## Repository Structure

```text
mejai-ai/
|
|-- main.py
|   Match-processing entry point for collected match IDs.
|
|-- requirements.txt
|   Python dependencies.
|
|-- src/
|   |
|   |-- api/
|   |   |-- crawler.py
|   |   `-- riot_client.py
|   |       Riot API discovery, routing, requests, and checkpoints.
|   |
|   |-- extractors/
|   |   |-- extract_match.py
|   |   |-- extract_snapshots.py
|   |   `-- extract_events.py
|   |       Structured extraction from match and timeline responses.
|   |
|   |-- process_match.py
|   |-- parquet_writer.py
|   |
|   |-- prepare/
|   |   `-- build_valid_match_manifest.py
|   |
|   |-- analysis/
|   |   |-- mejai_events.py
|   |   `-- reconstruct_mejai_events.py
|   |
|   |-- validation/
|   |   |-- validate_data.py
|   |   `-- validate_relationships.py
|   |
|   `-- research/
|       |-- config.py
|       |-- utils.py
|       |-- build_research_dataset.py
|       |-- build_purchase_comparison.py
|       |-- build_purchase_decision_features.py
|       |-- build_matched_analysis_dataset.py
|       |-- validate_dataset.py
|       |-- validate_semantics.py
|       |-- validate_matching_balance.py
|       |
|       `-- AI/
|           |-- client.py
|           |-- generate_report.py
|           `-- analyst.py
|
|-- reports/
    |
    |-- primary/
    |   |-- matching_balance_report.txt
    |   |-- matching_summary.csv
    |   |-- outcome_by_group.csv
    |   `-- outcome_summary.csv
    |
    `-- exploratory/
        |-- step_1_team_state_by_purchase_time_group.csv
        `-- step_2_player_state_by_purchase_time_group.csv

```

---

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/MineralWalter/mejai-ai.git
cd mejai-ai
```

### 2. Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

Current Python dependencies:

```text
numpy
pandas
pyarrow
requests
python-dotenv
ollama
```

### 4. Configure Riot API access

Create a local `.env` file:

```text
RIOT_API_KEY=your_riot_api_key_here
```

`.env` is ignored by Git and should not be committed.

### 5. Configure the local AI analyst

The configured Ollama model is:

```text
qwen3:4b-instruct
```

After installing Ollama:

```powershell
ollama pull qwen3:4b-instruct
```

The Python client expects Ollama at:

```text
http://localhost:11434
```

---

## Data Collection and Processing

The full raw and generated dataset is intentionally **not tracked in Git**.

Generated data lives under:

```text
data/
```

Crawler state lives under:

```text
checkpoints/
```

Both are ignored by version control.

### Collection layer

The collection pipeline works across Riot routing regions such as:

```text
sea
asia
europe
americas
```

Checkpointing allows large collection runs to resume without starting over. Collected match IDs are then processed by `main.py` to retrieve detailed match and timeline data.

Because the full collection is large and Riot API access is rate-limited, reproducing the complete acquisition stage is not necessary simply to review the research implementation.

### Parquet layer

Processed data is separated into:

```text
matches
participants
snapshots
events
```

This keeps match-level metadata, participant information, game-state snapshots, and timeline events logically separated for downstream analysis.

---

## Reproducing the Analysis

The analysis pipeline is modular. Generated intermediate datasets are expected under `data/`.

A representative active run order is:

### 1. Valid-match manifest

```powershell
py -m src.prepare.build_valid_match_manifest
```

### 2. Mejai event catalogue

```powershell
py -m src.analysis.mejai_events
```

### 3. Purchase lifecycle reconstruction

```powershell
py -m src.analysis.reconstruct_mejai_events
```

### 4. Purchase-case research dataset

```powershell
py -m src.research.build_research_dataset
```

### 5. Generalized non-purchase candidate pool

```powershell
py -m src.research.build_purchase_comparison
```

### 6. Purchase-decision event features

```powershell
py -m src.research.build_purchase_decision_features
```

### 7. Matched analysis datasets

```powershell
py -m src.research.build_matched_analysis_dataset
```

### 8. Research validation

```powershell
py -m src.research.validate_dataset
py -m src.research.validate_semantics
py -m src.research.validate_matching_balance
```

### 9. Outcome summaries

```powershell
py -m src.research.AI.generate_report --evidence-only
```

### LLM-guided exploration

Requires Ollama and `qwen3:4b-instruct`:

```powershell
py -m src.research.AI.analyst
```

The LLM exploration is run after the main matched analysis.

---

## Tracked Outputs

Large generated datasets remain local, while the main analysis outputs are kept under `reports/`.

### `reports/primary/`

#### `matching_summary.csv`

Summary of the primary, sensitivity, and 1:1 matching specifications.

#### `matching_balance_report.txt`

Matching balance diagnostics.

#### `outcome_summary.csv`

Overall matched outcome result.

#### `outcome_by_group.csv`

Results by team state, player state, purchase timing, and lifecycle status.

### `reports/exploratory/`

#### `step_1_team_state_by_purchase_time_group.csv`

Results from the first LLM-selected subgroup analysis.

#### `step_2_player_state_by_purchase_time_group.csv`

Results from the second LLM-selected subgroup analysis.

---

## Technical Stack

### Data collection

- Python
- Riot Games API
- `requests`
- `python-dotenv`
- checkpointed multi-region crawling

### Data engineering

- Pandas
- NumPy
- PyArrow
- Parquet
- structured match, participant, snapshot, and event tables

### Statistical analysis

- observational matched comparisons
- variable-ratio matching
- weighted within-set controls
- standardized mean difference diagnostics
- sensitivity and robustness specifications
- deterministic subgroup analysis

### AI integration

- Ollama
- Qwen3 4B Instruct
- structured JSON actions
- predefined subgroup analyses
- Python-based statistical execution

---

## Final Takeaway

The matched analysis found a **+4.08 percentage-point** win-rate difference between Mejai purchases and similar non-purchase situations.

The largest differences appeared when the buyer's team was close or behind rather than already ahead, which argues against a simple "Mejai only looks good because it is bought while already winning" explanation.

The LLM-guided exploration found similar descriptive patterns across several purchase-time subgroups.

These results remain observational. Matching improves comparability, but residual imbalance and unmeasured factors mean the analysis does not show that purchasing Mejai itself causes a higher chance of winning.

The project's local-LLM analyst is a separate, intentional component of the research workflow and is described in the AI-Native Exploratory Analysis section above.
