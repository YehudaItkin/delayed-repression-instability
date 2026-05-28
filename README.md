# Delayed Repression and Emergent Instability in Adaptive Multi-Agent Systems

Code and data for the paper:

> **Delayed Repression and Emergent Instability in Adaptive Multi-Agent Systems**
> Igor Itkin, 2026

## Setup

```bash
pip install -r requirements.txt  # numpy, pandas, networkx, matplotlib
```

## Running experiments

```bash
# Smoke test
python experiments/scripts/run_single.py --config experiments/configs/default.json --out results/smoke

# Paper sweeps
python experiments/scripts/run_paper1_delay_sweep.py --config experiments/configs/paper1_delay_sweep.json --out results/delay_sweep
python experiments/scripts/run_paper1_k_sweep.py --config experiments/configs/paper1_k_sweep.json --out results/k_sweep
python experiments/scripts/run_paper1_phase_diagram.py --config experiments/configs/paper1_phase_diagram.json --out results/phase_diagram
python experiments/scripts/run_paper1_learning_ablation.py --config experiments/configs/paper1_learning_ablation.json --out results/learning_ablation
```

## Structure

```
experiments/
  src/autonomy_lab/   # Core simulation library
    simulation.py     # Network builder, Q-learning agents, step logic
    theory.py         # Analytical: x_star, delta_c (critical delay)
    metrics.py        # summarize_history(), classify_run()
  scripts/            # CLI entry points for sweeps
  configs/            # JSON experiment configs
```

## License

MIT
