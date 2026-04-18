# Mean Consensus Deviation (MCD) & Mean Consensus Bias (MCB) — Leave-One-Out

**MCD** = mean |judge − LOO consensus| → magnitude of disagreement (always ≥ 0).
**MCB** = mean  (judge − LOO consensus) → signed direction: **+** = lenient, **−** = harsh.
LOO consensus = average of the *other six* judges on each item (the evaluated judge excluded).

## Condition A — Generic overall

| Judge | Mean Score | MCD (pts) | MCB (pts) |
|-------|------------|-----------|-----------|
| Claude Haiku 4.5 | 75.7 | 15.0 | -6.5 |
| Claude Opus 4 | 71.6 | 16.2 | -11.4 |
| Claude Sonnet 4 | 70.8 | 16.7 | -12.2 |
| Claude Sonnet 4.6 | 70.9 | 14.9 | -12.1 |
| GPT-4 | 98.5 | 22.0 | +20.1 |
| GPT-4o | 91.3 | 12.6 | +11.7 |
| GPT-4o mini | 90.2 | 12.5 | +10.4 |
| **Panel mean** | | **15.7** | **-0.0** |

## Condition B — Metric rubric (mean of accuracy/relevance/completeness)

| Judge | Mean Score | MCD (pts) | MCB (pts) |
|-------|------------|-----------|-----------|
| Claude Haiku 4.5 | 77.1 | 15.7 | -10.0 |
| Claude Opus 4 | 81.8 | 10.8 | -4.4 |
| Claude Sonnet 4 | 83.7 | 10.9 | -2.2 |
| Claude Sonnet 4.6 | 76.1 | 13.7 | -11.0 |
| GPT-4 | 97.6 | 16.2 | +14.0 |
| GPT-4o | 95.7 | 11.9 | +11.8 |
| GPT-4o mini | 87.1 | 11.2 | +1.8 |
| **Panel mean** | | **12.9** | **+0.0** |

## Condition C — Per-item custom

| Judge | Mean Score | MCD (pts) | MCB (pts) |
|-------|------------|-----------|-----------|
| Claude Haiku 4.5 | 84.9 | 6.4 | -2.4 |
| Claude Opus 4 | 79.0 | 11.7 | -9.3 |
| Claude Sonnet 4 | 81.6 | 8.2 | -6.3 |
| Claude Sonnet 4.6 | 83.2 | 6.4 | -4.5 |
| GPT-4 | 95.5 | 9.9 | +9.9 |
| GPT-4o | 95.8 | 10.3 | +10.3 |
| GPT-4o mini | 89.0 | 6.5 | +2.4 |
| **Panel mean** | | **8.5** | **+0.0** |

## Condition B — per-metric detail

| Judge | Acc MCD | Acc MCB | Rel MCD | Rel MCB | Comp MCD | Comp MCB |
|-------|---------|---------|---------|---------|----------|----------|
| Claude Haiku 4.5 | 17.3 | -8.8 | 14.8 | -11.6 | 14.8 | -9.4 |
| Claude Opus 4 | 19.3 | -14.8 | 5.4 | +1.7 | 7.7 | -0.1 |
| Claude Sonnet 4 | 17.4 | -9.3 | 8.4 | +3.2 | 7.0 | -0.5 |
| Claude Sonnet 4.6 | 14.8 | -9.9 | 13.2 | -11.1 | 13.1 | -12.0 |
| GPT-4 | 22.9 | +16.3 | 14.1 | +14.1 | 11.5 | +11.5 |
| GPT-4o | 16.6 | +16.5 | 10.3 | +10.2 | 8.7 | +8.7 |
| GPT-4o mini | 12.6 | +10.1 | 14.2 | -6.6 | 7.0 | +1.7 |

## Cross-condition summary (RS% / MCD / MCB / Mean Score)

| Judge | A: RS% | A: MCD | A: MCB | A: Score | B: RS% | B: MCD | B: MCB | B: Score | C: RS% | C: MCD | C: MCB | C: Score | **Overall RS%** | **Overall MCD** | **Overall MCB** | **Overall Score** |
|-------|--------|--------|--------|----------|--------|--------|--------|----------|--------|--------|--------|----------|----------------|----------------|----------------|-------------------|
| Claude Haiku 4.5 | 96.7% | 15.0 | -6.5 | 75.7 | 94.4% | 15.7 | -10.0 | 77.1 | 93.3% | 6.4 | -2.4 | 84.9 | **94.8%** | **12.4** | **-6.3** | **79.2** |
| Claude Opus 4 | 93.3% | 16.2 | -11.4 | 71.6 | 95.6% | 10.8 | -4.4 | 81.8 | 70.0% | 11.7 | -9.3 | 79.0 | **86.3%** | **12.9** | **-8.4** | **77.5** |
| Claude Sonnet 4 | 100.0% | 16.7 | -12.2 | 70.8 | 100.0% | 10.9 | -2.2 | 83.7 | 100.0% | 8.2 | -6.3 | 81.6 | **100.0%** | **12.0** | **-6.9** | **78.7** |
| Claude Sonnet 4.6 | 73.3% | 14.9 | -12.1 | 70.9 | 73.3% | 13.7 | -11.0 | 76.1 | 80.0% | 6.4 | -4.5 | 83.2 | **75.6%** | **11.7** | **-9.2** | **76.7** |
| GPT-4 | 90.0% | 22.0 | +20.1 | 98.5 | 94.4% | 16.2 | +14.0 | 97.6 | 83.3% | 9.9 | +9.9 | 95.5 | **89.3%** | **16.0** | **+14.6** | **97.2** |
| GPT-4o | 100.0% | 12.6 | +11.7 | 91.3 | 72.2% | 11.9 | +11.8 | 95.7 | 90.0% | 10.3 | +10.3 | 95.8 | **87.4%** | **11.6** | **+11.3** | **94.3** |
| GPT-4o mini | 73.3% | 12.5 | +10.4 | 90.2 | 71.1% | 11.2 | +1.8 | 87.1 | 96.7% | 6.5 | +2.4 | 89.0 | **80.4%** | **10.1** | **+4.9** | **88.8** |
| **Mean** | 89.5% | 15.7 | -0.0 | 81.3 | 85.9% | 12.9 | +0.0 | 85.6 | 87.6% | 8.5 | +0.0 | 87.0 | **87.7%** | **12.4** | **+0.0** | **84.6** |

## Key observations

- **Correlation(RS%, MCD)** across 7 judges (overall): **r = 0.277**
  - Weakly positive: higher repeat stability does *not* predict closer consensus alignment.

- **Closest to consensus (lowest MCD):** GPT-4o mini — MCD = 10.1 pts
- **Farthest from consensus (highest MCD):** GPT-4 — MCD = 16.0 pts

- **Most lenient (highest MCB):** GPT-4 — MCB = +14.6 pts
- **Most harsh (lowest MCB):** Claude Sonnet 4.6 — MCB = -9.2 pts

- **GPT-4:** RS% = 89.3%, MCD = 16.0, MCB = +14.6, Mean Score = 97.2
  - High stability, highest deviation, and most lenient — systematic high-scorer.

- **Claude Sonnet 4:** RS% = 100.0%, MCD = 12.0, MCB = -6.9, Mean Score = 78.7
  - Perfect stability but mid-pack consensus alignment; sign shows scoring direction.

- **Panel mean MCD by condition:** A = 15.7, B = 12.9, C = 8.5
  - Per-item custom instructions (C) nearly halve cross-judge disagreement vs generic (A).
