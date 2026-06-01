# Experiment Index

What each experiment is, where its frozen results live, and what it shows.
Theory first (Exp 1-2), then the networked simulation (Exp 3-6). All networked
runs use N=240 agents on a 6-community stochastic block model, 50 seeds per
condition.

| # | Experiment | Results dir | Varies | What it shows |
|---|---|---|---|---|
| 1 | ODE validation | (run `run_ode_validation.py`; figures) | Delta/Delta_c in [0,4] | Hopf bifurcation at Delta_c exactly; supercritical (bounded cycles) |
| 2 | Discrete mean-field | (run `run_discrete_meanfield.py`) | step size eta, Delta | discretization shrinks the stability margin monotonically |
| 3 | Delay sweep | `results/delay_sweep_aggressive/` | Delta in {0..30}, k=20 | runaway 10% -> 72% (+62 pp): delay alone destabilizes |
| 4 | Sharpness sweep | `results/k_sweep_aggressive/` | k in {3..40}, Delta=15 | runaway 16% -> 64%; sharp knee at k=7-10 (crossing Delta_c) |
| 5 | **Crossed delay x architecture** (central) | `results/delay_x_learning_3way/` | Delta x {fixed, reactive, Q-learning}, k=10 | reactive 96% > Q-learning 66% > fixed 0% -- reactivity, not learning, destabilizes |
| 6 | RL regulator (exploratory) | `results/learning_ablation/` | regulator type, Delta=6 | RL regulator converts runaway -> bounded oscillation, not full stability |

Robustness checks (support the claims above):

| Check | Results dir | What it shows |
|---|---|---|
| Reactive fine sweep | `results/reactive_delay_fine/` | reactive threshold sits sharply between Delta=2 (90% stable) and Delta=3 (62% runaway) |
| Denser graph | `results/network_robustness/` | architecture ordering holds on a 4-community 0.15/0.02 graph |
| RL hyperparameters | `results/alpha_gamma_robust/` | ordering and delay->runaway trend hold across the alpha x gamma grid |
| 2-architecture precursor | `results/delay_x_learning/` | fixed vs Q-learning only (Exp 5 adds the reactive arm) |

## Reproduce Exp 5 (central crossing)

```python
import pandas as pd
df = pd.read_csv("results/delay_x_learning_3way/summary.csv")
rate = (df.assign(runaway=df.regime.eq("runaway"))
          .groupby(["learning_mode", "delay_steps"]).runaway.mean()
          .unstack().round(2))
print(rate)   # fixed 0 everywhere; reactive 0/.80/.96/.96/.96; q_learning .10/.18/.32/.50/.66
```

## Reporting convention

50 seeds per configuration. Every seeded run stores `history.csv`,
`summary.json`, and `config_resolved.json`; each sweep also writes the
aggregated `summary.csv` committed here.
