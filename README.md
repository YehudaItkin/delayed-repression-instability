# Delayed Repression and Emergent Instability in Adaptive Multi-Agent Systems

Code, data, and paper source for:

> **Delayed Repression and Emergent Instability in Adaptive Multi-Agent Systems**
> Igor Itkin, 2026
> arXiv:2605.30392 [cs.MA]

## Key finding

Institutional delay alone can destabilize an otherwise stable multi-agent system. The destabilizing ingredient is not learning but *reactivity* to delayed signals. Reactive agents are perfectly stable without delay (100% stable at delay=0) yet collapse catastrophically once delay is introduced (96% runaway by delay>=8). Q-learning agents achieve partial resilience (66% runaway at delay=20) by encoding punishment history into value functions, and fixed-policy agents are immune (0% at all delays).

## Setup

```bash
pip install -r requirements.txt  # numpy, pandas, networkx, matplotlib
```

## Reproducing results

```bash
# Smoke test (single run, ~5 seconds)
python experiments/scripts/run_single.py --config experiments/configs/default.json --out results/smoke

# Theory validation
python experiments/scripts/run_ode_validation.py        # Exp 1: Hopf bifurcation at Delta_c
python experiments/scripts/run_discrete_meanfield.py    # Exp 2: discretization effect

# Networked simulation (each takes 5-30 minutes)
python experiments/scripts/run_paper1_delay_sweep.py --config experiments/configs/paper1_delay_sweep_aggressive.json --out results/delay_sweep_aggressive
python experiments/scripts/run_paper1_k_sweep.py --config experiments/configs/paper1_k_sweep_aggressive.json --out results/k_sweep_aggressive
python experiments/scripts/run_paper1_phase_diagram.py --config experiments/configs/paper1_delay_x_learning_3way.json --out results/delay_x_learning_3way
python experiments/scripts/run_paper1_learning_ablation.py --config experiments/configs/paper1_learning_ablation_v2.json --out results/learning_ablation

# Generate figures from results
python experiments/scripts/make_paper1_figures_v2.py
```

## What is what

`EXPERIMENT_INDEX.md` lists each experiment, the results directory that holds it,
and what it shows. The central experiment (delay x architecture) is in
`results/delay_x_learning_3way/`; reproduce its table by grouping `summary.csv`
on the `learning_mode` column (`fixed` / `reactive` / `q_learning`) and taking
the fraction of `regime == runaway` per delay.

## Pre-computed results

`results/` holds the frozen `summary.csv` for every experiment (one row per run:
tail-period means, regime classification, resolved parameters). The six main
experiments plus four robustness checks (`network_robustness`, `alpha_gamma_robust`,
`reactive_delay_fine`, and the threshold-sensitivity sweep) are all included.
Full per-step trajectories regenerate by re-running the commands above.

## Structure

```
paper/latex/              # Paper source (main.tex, sections, figures, .bbl)
experiments/
  src/autonomy_lab/       # Core simulation library
    simulation.py         # Network builder, Q-learning agents, step logic
    theory.py             # Analytical: x_star, delta_c (critical delay)
    metrics.py            # summarize_history(), classify_run()
  scripts/                # CLI entry points
    run_paper1_*.py       # Sweep runners
    run_ode_validation.py # ODE integration (Exp 1)
    run_discrete_meanfield.py  # Discrete mean-field bridge (Exp 2)
    make_paper1_figures_v2.py  # Figure generation
  configs/                # JSON experiment configs
results/                  # Frozen summary statistics + MANIFEST.md
figures/                  # Paper figures (PDF)
EXPERIMENT_INDEX.md       # Which experiment is what, and where its results live
```

## License

MIT
