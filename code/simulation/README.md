# Zorowitz weighted-extrema simulation

This folder holds the simulation reported in the manuscript. It implements the
successor-state backup of Zorowitz, Momennejad, and Daw (2020):

\[
U_w(s')=w\max_{a'}Q(s',a')+(1-w)\min_{a'}Q(s',a'),\qquad 0\leq w\leq1.
\]

The environment is a bidirectional 10-state chain with learned left and right
actions, a nonrewarded endpoint, and a stochastic rewarded endpoint (reward
probability .85). Learning is epsilon-greedy (`epsilon = .20`, `alpha = .1`,
`gamma = .8`, 100 episodes, 20 runs per efficacy weight
`w = 0, .25, .50, .75, 1`). The state value reported after learning is the
greedy value `V(s) = max_a Q(s,a)`. The simulation uses no participant data.

## Reproduce

From this directory:

```bash
python run_simulation.py
python verify_simulation.py
```

`run_all.sh` at the top of the release runs both and moves the outputs to
`results/simulation/`. The run is deterministic at seed 1 and writes:

- `simulation_results.npz`: learned Q values, state values, curve fits, and
  simulation diagnostics;
- `simulation_run_level.csv`: one row per Monte Carlo run;
- `simulation_summary.csv`: one row per efficacy weight;
- `simulation_manifest.json`: model, environment, mapping, fit, seed, and
  verification metadata;
- `figures/`: standalone and three-panel PNG/PDF figures.

`verify_simulation.py` checks the backup operator on a toy case and confirms
that the run reproduces the archived condition means
(`mu_sim` = 15.38, 18.43, 23.30, 32.49, 46.82 for `w` = 0 to 1).

## Seed robustness

A 20-seed audit (seeds 1 to 20; otherwise identical settings) reproduced the
direction in every run: all 20 seeds yielded strictly increasing condition
means across `w = [0, .25, .50, .75, 1]`. All 2,000 fitted curves were valid
and no episode reached the safety limit. These Monte Carlo replications
establish numerical robustness, not additional inferential sample size, so the
manuscript reports the five condition means and their monotonic ordering
without a linear trend statistic. Pearson fields remain in the outputs only as
deterministic provenance checks.

## Reporting coordinate and mapping

The simulated physical coordinate places the rewarded endpoint at 90. Figures
use distance from reward (0 rewarded, 90 nonrewarded) in simulation units and
report `mu_sim = 90 - physical midpoint`; larger `mu_sim` means broader
simulated reward generalization. `mu_sim` is a simulation-coordinate summary,
not the participant-level behavioral `mu`; the two 0-90 scales are not
numerically commensurate.

Values are mapped to choices with a binary softmax against zero (`beta = 3`)
followed by within-run rescaling to 0-1, so the y-axis is labeled
*endpoint-rescaled reward-choice propensity*, not a calibrated reward
probability. In the choice-propensity panel of the simulation figure, points
are exact condition-mean propensities and lines are the means of the 20
run-specific four-parameter logistic fits.
