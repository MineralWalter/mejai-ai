# Analysing High-Risk, High-Reward Game Decisions Using Matched Game States

An observational League of Legends data project examining whether **Mejai's Soulstealer** is mainly associated with already-winning game states,or whether positive matched win-rate differences also appear when the buyer's team is losing.

This repository covers an AI-guided research workflow, combining a constrained LLM  with  Python analysis. It also includes data collection via Riot API, timeline extraction, purchase-lifecycle reconstruction, feature engineering, matched observational analysis, and balance and robustness validation. The LLM selects bounded exploratory follow-up questions from an approved statistical toolset, while Python validates actions and performs all numerical calculations.

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
Match discovery and collection
   |
   v
Match + timeline extraction
   |
   v
Parquet research tables
   |
   v
Valid-match manifest
   |
   v
Mejai event catalogue
   |
   v
Purchase lifecycle reconstruction
   |
   v
Case dataset + non-purchase candidate pool
   |
   v
Purchase-time feature engineering
   |
   v
Matched analysis
   |
   +--> balance diagnostics
   +--> sensitivity / robustness analyses
   |
   v
Deterministic outcome summaries
   |
   v
Bounded local-LLM exploratory analyst
   |
   v
Auditable exploratory CSVs + trace
```

### Design principles

The final project has 4 phases:

1. **Data acquisition** — Riot API match and timeline collection.
2. **Deterministic research pipeline** — event reconstruction, feature engineering, matching, validation, and outcome calculation.
3. **AI-guided exploration** — a bounded local model chooses from a restricted set of follow-up analyses after the primary analysis is frozen.
4. **Tracked evidence** — compact final CSV and text outputs under `reports/`
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

The collection strategy targeted **high-MMR Ranked Solo/Duo** games across multiple Riot servers: Vietnam, EU West, North America, Korea. Match discovery began from regional seed accounts and expanded through participant relationships with checkpointed collection state.

### Purchase lifecycle reconstruction

Mejai purchases were reconstructed from timeline events rather than inferred only from final inventories.

| Lifecycle status | Count |
| --- | ---: |
| `RETAINED` | 21,196 |
| `SOLD` | 4,220 |
| `UNDONE` | 934 |
| **Total** | **26,350** |

The primary analysis uses:

```text
RETAINED + SOLD
```

and excludes:

```text
UNDONE
```

This produces **25,416 primary eligible purchases**.

`UNDONE` purchases are excluded because an immediately reversed transaction does not represent the same purchase decision as a retained or later-sold item.

---

## Research Design

### Unit of analysis

The analysis is built around the **Mejai purchase event**, not simply the player or full match.

For each eligible purchase, the pipeline reconstructs game state around the purchase timestamp. Candidate controls are observations where a comparable player **did not purchase Mejai** at a similar point under similar conditions.

### Purchase-time information

The matching and diagnostic pipeline uses information such as:

- game time;
- player level;
- player total gold;
- player current gold;
- player XP;
- minion and jungle CS;
- player gold difference versus the same-role opponent;
- player XP difference versus the same-role opponent;
- rest-of-team gold difference;
- rest-of-team XP difference;
- whole-team gold and XP state;
- recent kills, deaths, and assists;
- prior Dark Seal ownership;
- region;
- team position; and
- champion identity for balance diagnostics.

Recent combat features use a five-minute lookback window.

### Why matching?

The central confounding problem is:

```text
Strong game state
      |
      +------> higher probability of buying Mejai
      |
      +------> higher probability of winning
```

A raw win-rate comparison would therefore mix the purchase with the conditions that made the purchase more likely.

Matching attempts to construct a more comparable non-purchase reference group on observed pre-purchase state. This reduces, but does not eliminate, confounding.

---

## Primary Matching Strategy

The frozen primary design uses **variable-ratio matching with up to three controls per Mejai case**.

### Primary restrictions

The final primary matcher includes:

- same region;
- same team position;
- exact prior Dark Seal status;
- a hard maximum current-gold gap of **750 gold**;
- state and recent-event variables used for candidate ranking;
- up to **3 controls per case**; and
- within-set control weights that sum to 1.

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

All **34,128** selected primary control rows satisfied the exact prior-Dark-Seal requirement. No Dark Seal fallback controls were used in the primary analysis.

### Outcome calculation

For each matched set:

```text
matched risk difference
    = case win indicator
    - weighted control win rate
```

The overall reported difference is the mean matched-set difference. This preserves the variable-ratio weighting design instead of allowing cases with more controls to dominate the estimate.

---

## Primary Results

### Overall matched result

The final primary analysis contains **20,991 matched Mejai purchase cases**.

| Group | Win rate |
| --- | ---: |
| Mejai purchase cases | 77.28% |
| Weighted matched controls | 73.20% |
| **Matched win-rate difference** | **+4.08 percentage points** |

Approximate 95% confidence interval for the matched-set difference:

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

The largest descriptive matched differences appear in **close** and **behind** team states rather than in the already-ahead group. What this means is that it argues against interpreting the entire observed Mejai association as a simple "already winning by a lot" pattern; it does **not** show that buying Mejai causes a comeback.

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

## AI-Native Exploratory Analysis

The primary statistical analysis was completed and frozen **before** the AI-guided exploration.

The AI component is deliberately constrained so that the model can direct a small exploratory workflow without becoming the statistical engine.

### Local model

The project uses:

```text
Ollama
qwen3:4b-instruct
```

### Allowed analyses

The model may choose only from:

```text
team_state × player_state
team_state × purchase_time_group
player_state × purchase_time_group
```

The maximum number of exploratory steps is:

```text
2
```

The model can:

1. inspect deterministic primary evidence;
2. choose one approved joint subgroup analysis;
3. receive the deterministic Python result; and
4. choose one additional approved analysis or stop.

### Guardrails

The model cannot:

- execute arbitrary Python or SQL;
- modify the matcher;
- change the primary analysis;
- alter matching calipers;
- create unrestricted statistical procedures;
- search arbitrary subgroup combinations;
- change the minimum support threshold;
- calculate final statistics itself; or
- promote exploratory results into primary findings.

Python validates the requested action and performs all calculations.

### Architecture

```text
Frozen primary evidence
        |
        v
Local LLM chooses approved question
        |
        v
Python validates action
        |
        v
Python computes deterministic result
        |
        v
Result returned to LLM
        |
        v
LLM chooses another approved question or stops
        |
        v
Full audit trace saved
```

This makes the project AI-native without allowing model-generated prose or calculations to replace the statistical pipeline.

---

## Exploratory Results

These results are **descriptive joint subgroup analyses**, not formal interaction or moderation tests.

Cells require at least **200 matched sets**.

### Step 1 — Team State x Purchase Timing

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

The behind-before-15 cell did not meet the minimum support threshold.

### Step 2 — Player State x Purchase Timing

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

The behind-before-15 cell did not meet the minimum support threshold.

Cell-level confidence intervals do **not** establish statistically significant differences between cells.

---

## Interpretation

The primary matched analysis does not support a purely simplistic interpretation of Mejai's Soulstealer as an item whose observed success is explained only by players purchasing it while already far ahead.

Overall:

```text
Matched difference: +4.08 pp
```

By team state:

```text
Close:  +10.42 pp
Behind:  +8.28 pp
Ahead:   +1.21 pp
```

The exploratory joint subgroup analysis adds nuance by showing that several larger descriptive differences also occur in close or behind team states across different purchase windows.

The appropriate conclusion is:

> **Mejai purchases are positively associated with winning within the final matched sample, including in close and behind game-state strata.**

The project does **not** establish:

> Buying Mejai causes a higher probability of winning.

Unobserved player quality, champion-specific context, team composition, recall timing, decision context, and other factors may still influence both the purchase and final outcome.

### Lifecycle outcomes

Lifecycle was also retained for descriptive post-purchase analysis:

| Lifecycle | Matched sets | Matched difference |
| --- | ---: | ---: |
| Retained | 17,507 | +8.34 pp |
| Sold | 3,484 | -17.36 pp |

These results are **descriptive only**. Whether Mejai is eventually retained or sold is determined after the original purchase and may depend on how the match develops, so lifecycle status is not treated as a purchase-time causal grouping.

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

The collection strategy targeted high-MMR Ranked Solo/Duo games, as players are likely to have more accurate itemization and decision-making. Results may not generalize directly to lower ranks, other queues, coordinated teams, or professional competition.

### Exploratory subgroup selection

The AI-guided subgroup analyses were selected after the primary results were available. They are exploratory rather than independently pre-specified confirmatory analyses.

### No formal interaction test

The joint subgroup tables are descriptive cross-stratified analyses. The project does not claim formal statistical interaction, moderation, or significant differences between subgroup cells.

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
|   |
|   |-- primary/
|   |   |-- matching_balance_report.txt
|   |   |-- matching_summary.csv
|   |   |-- outcome_by_group.csv
|   |   `-- outcome_summary.csv
|   |
|   `-- exploratory/
|       |-- analyst_trace.json
|       |-- step_1_team_state_by_purchase_time_group.csv
|       `-- step_2_player_state_by_purchase_time_group.csv
|
`-- archive/
    |-- legacy_preprocessing/
    |-- diagnostics/
    `-- experiments/
```

`archive/` preserves historical development work but is not part of the final active primary analysis.

---

## Installation

### 1. Clone the repository

```powershell
git clone <repository-url>
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

Checkpointed state supports large API collection jobs. The historical collection stage produced the match IDs used by `main.py` for detailed match and timeline processing.

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

### 9. Deterministic outcome evidence

```powershell
py -m src.research.AI.generate_report --evidence-only
```

### 10. Bounded AI-guided exploration

Requires Ollama and `qwen3:4b-instruct`:

```powershell
py -m src.research.AI.analyst
```

The AI stage is **post-primary exploratory analysis**.

---

## Tracked Outputs

Large generated datasets remain local. The repository tracks a compact evidence package instead.

### `reports/primary/`

#### `matching_summary.csv`

Summary of primary variable-ratio, relaxed sensitivity, and 1:1 robustness matching.

#### `matching_balance_report.txt`

Detailed final matching-balance diagnostics.

#### `outcome_summary.csv`

Overall deterministic matched outcome result.

#### `outcome_by_group.csv`

Deterministic subgroup summaries by team state, player state, purchase timing, and descriptive lifecycle status.

### `reports/exploratory/`

#### `analyst_trace.json`

Audit trail of the bounded AI-guided exploratory path.

#### `step_1_team_state_by_purchase_time_group.csv`

Deterministic output for the first AI-selected joint subgroup analysis.

#### `step_2_player_state_by_purchase_time_group.csv`

Deterministic output for the second AI-selected joint subgroup analysis.

The deterministic CSV outputs are the authoritative exploratory evidence. Free-form model prose is not treated as statistical evidence.

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
- restricted JSON action protocol
- deterministic Python execution
- bounded two-step exploratory loop
- auditable trace logging

---

## Final Takeaway

The final matched study finds a positive association between Mejai's Soulstealer purchases and winning:

```text
Overall matched difference: +4.08 percentage points
```

The association is not concentrated only in already-ahead team states:

```text
Close team state:  +10.42 pp
Behind team state:  +8.28 pp
Ahead team state:   +1.21 pp
```

These results suggest that Mejai's observed success is not explained solely by players purchasing it from already-dominant positions. In the matched sample, the largest descriptive win-rate differences instead appeared in close and behind team states, supporting a more nuanced interpretation than "Mejai only looks good because it is bought while already winning."

The bounded AI-guided exploratory analysis added further context by selecting cross-stratified follow-up analyses across game state and purchase timing. Those deterministic results showed similar patterns across several close and behind-state subgroups, while keeping the primary matched analysis unchanged.

The conclusion remains observational: matching improves comparability between purchase and non-purchase situations, but residual imbalance and unmeasured factors remain. The project therefore supports an association between Mejai purchases and higher win rates across multiple game states, rather than a causal claim that purchasing Mejai itself increases the probability of winning.
