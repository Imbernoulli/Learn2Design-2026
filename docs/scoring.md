# Scoring

## Per-run score

For a single topology, your run's score is the best loss reached by your
algorithm within the time budget:

$$
s_{\text{run}} = \min_{t \in [0,\, T_{\text{budget}}]} \mathcal{L}(\theta_t)
$$

where $\mathcal{L}$ is the sensitivity-derived loss returned by
`Objective.value` and $\theta_t$ are the parameters evaluated at time $t$.
Lower is better.

The value is read from `objective.best_loss` after the run terminates. If your
algorithm crashes, exceeds memory, or fails to call `objective.value` at all,
the run's score is evlauated as the initial parameter's loss.

`NaN` losses are coerced to `+inf` before aggregation.

---

## Per-month score

Each public-leaderboard month is scored on 10 topologies. The monthly score is
the arithmetic mean of the 10 per-run scores:

$$
S_{\text{month}} = \frac{1}{10} \sum_{i=1}^{10} s_{\text{run}}^{(i)}
$$

Loss magnitudes are comparable across different topologies.

---

## Final score

The final leaderboard is computed identically, but on the 10 **private**
topologies, which are never published. Your final score is

$$
S_{\text{final}} = \frac{1}{10} \sum_{i=1}^{10} s_{\text{run}}^{(i, \text{private})}
$$

The submission used for the final evaluation is the last submission you
made to the public leaderboard before the deadline. You may keep iterating
all month. Only your final commit counts.

---

## Tie-breaking

If two submissions are within machine precision on the final score, the
tie-breaker is, in order:

1. Lower mean wall-clock time to reach the best loss (faster algorithm wins).
2. Lower mean number of `Objective.value` calls (fewer evaluations wins).
3. Earlier submission timestamp.

---

## Anti-cheating policy

See [submission.md](submission.md#prize-eligibility-and-source-disclosure)
for the source-disclosure requirement for top-10 finishers, and the
[Disqualification criteria](submission.md#disqualification-criteria) section.
