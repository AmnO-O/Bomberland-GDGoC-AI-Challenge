# Bomberland Strategy Trend Analysis: Final Report
**Date generated:** June 17, 2026
**Data span:** May 21, 2026 – June 16, 2026

## 1. Overview and Participation
The data extracted from the evaluation pipeline provides a comprehensive look into the competition's scale and participant activity:
- **Active Teams:** 267
- **Total Student Submissions:** 1,257
- **Total Matches Evaluated:** 129,425 matches
- **Behavioral Dataset:** 79,632 match-player action sequences successfully processed.

## 2. Meta-Game and Code Classification
We successfully classified 1,112 agent submissions by parsing their abstract syntax trees (AST) and regex patterns. The results highlight the dominant coding paradigms:

| Code Type | Submission Count | Description |
| :--- | :--- | :--- |
| **Rule-Based** | 593 (53%) | Hardcoded logic, state machines, and BFS pathfinding. |
| **Hybrid** | 477 (43%) | Machine learning models (PyTorch/ONNX) layered with hardcoded safety rules (e.g., danger avoidance). |
| **Copy Baseline** | 39 (3.5%) | Direct copies of starter kits with minimal to no changes. |
| **Pure ML (PyTorch)** | 3 (0.3%) | Neural networks operating entirely without explicit safety rules. |

*Observation:* The competition is heavily split between pure rule-based developers and those attempting to augment Reinforcement Learning templates with manual safety overrides (Hybrids).

## 3. Performance by Strategy Type
By merging the code classification with the competition TrueSkill database (`submission_performance.csv`), a clear hierarchy in effectiveness emerges:

- **Rule-Based:** `TrueSkill (Mu) = 109.90` | **Win Rate: 33.4%**
- **Hybrid:** `TrueSkill (Mu) = 105.15` | **Win Rate: 15.9%**
- **Copy Baseline:** `TrueSkill (Mu) = 102.90` | **Win Rate: 16.7%**
- **Pure ML:** `TrueSkill (Mu) = 101.92` | **Win Rate: 0.7%**

*Conclusion:* **Rule-based agents are absolutely dominating the meta.** The Hybrid approach (ML + Rules) is vastly underperforming compared to pure heuristics. Pure ML agents are almost entirely non-viable, boasting a sub-1% win rate, likely due to the strict penalty of self-destruction which neural networks struggle to learn perfectly.

## 4. Behavioral Metrics
By stream-processing JSON match logs, we extracted the following global averages across all agents:

- **Survival & Navigation:**
  - **Average Steps Alive:** 335.9 steps per match.
  - **Map Coverage:** 48.1% (Agents tend to explore about half the map before the match ends).
  - **Idle Ratio:** 36.4% (Agents spend over a third of the match standing still—likely waiting for bombs to detonate safely, or getting stuck in decision-loops).
  - **Movement Entropy:** 1.791 (Indicates moderately unpredictable movement patterns).

- **Offense & Resource Management:**
  - **Total Bombs Placed:** ~24.8 bombs per match.
  - **Bombs near Enemy (Aggro):** ~13.1 bombs (Over half of all bombs are used aggressively against players).
  - **Bombs near Boxes (Farming):** ~3.5 bombs (Box farming is a surprisingly small fraction of bomb usage, suggesting an overly aggressive, player-hunting meta).

## 5. Visualizations Overview (`analysis/plots/`)
The generated visual reports provide further diagnostic clarity:
1. **`code_type_distribution.png`**: Highlights the massive dichotomy between Rule-Based and Hybrid strategies.
2. **`complexity_vs_rating.png`**: Confirms that complex ML strategies are negatively correlated with leaderboard rating.
3. **`strategy_clusters_pca.png`**: Unsupervised PCA clustering reveals distinct clusters of playstyles—separating the "Aggressive Hunters" (high movement, high bombs_near_enemy) from the "Turtling Survivors" (high idle ratio, high steps_alive).
4. **`behavioral_heatmap.png`**: Shows strong correlations between specific traits (e.g., negative correlation between idle ratio and win probability).
5. **`submission_timeline.png` & `matches_per_day.png`**: Show a healthy, steady daily engagement rate, with a smooth upward trajectory as we approach the final week.

## Summary
The Bomberland AI Challenge is currently defined by an **aggressive, heuristic-driven meta**. Participants who focus on hardcoded pathfinding, explicit danger avoidance, and player-hunting are significantly outperforming those attempting to train Reinforcement Learning models. 
