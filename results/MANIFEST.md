# Paper 1 v2 Results — MANIFEST

Content-neutral runs (ctype='N' when enable_content=false). Adaptive FFT regime classifier.

## Experiments

| ID | Experiment | Config | Key params | Runs |
|---|---|---|---|---|
| P1-E0 | ODE validation | (script: run_ode_validation.py) | a=2, C=7, k=10, x_c=0.72 | N/A (ODE) |
| P1-E1 | Discrete mean-field | (script: run_discrete_meanfield.py) | η sweep, delay sweep | N/A (scalar) |
| P1-E2 | Aggressive delay sweep | paper1_delay_sweep_aggressive.json | k=20, 500 steps, delays 0-30 | 450 |
| P1-E3 | Sharpness sweep | paper1_k_sweep_aggressive.json | delay=15, 500 steps, k=3-40 | 400 |
| P1-E4 | Crossed delay × learning | paper1_delay_x_learning.json | k=10, 500 steps, delays 0-20 × learning | 500 |
| P1-E5 | RL regulator ablation | paper1_learning_ablation_v2.json | 3 conditions, 500 steps | 150 |

## Per-run artifacts

- `run_NNNN_seed_S/history.csv` — full time series
- `run_NNNN_seed_S/summary.json` — tail statistics + regime labels
- `run_NNNN_seed_S/config_resolved.json` — exact parameters
- `summary.csv` — sweep-level aggregate (one row per run)

## Common parameters

- N=240 agents, 6 communities, p_in=0.08, p_out=0.004
- Bridge fraction 12% (betweenness centrality)
- Q-learning: alpha=0.10, epsilon=0.08, gamma=0.95
- enable_content=false (neutral payoffs, no content bonuses)
- 50 seeds (1-50)
