#!/usr/bin/env python3
"""Generate Paper 1 figures from v2 (content-neutral) data."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
RESULTS = REPO_ROOT / 'results'
FIGDIR = REPO_ROOT / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)


def fig_delay_sweep():
    base = RESULTS / 'delay_sweep_aggressive'
    summary = pd.read_csv(base / 'summary.csv')
    delays = sorted(summary['delay_steps'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for delay in delays:
        runs = summary[summary['delay_steps'] == delay]
        seed = int(runs.iloc[0]['seed'])
        idx = delays.index(delay) * 50
        run_dir = base / f'run_{idx:04d}_seed_{seed}'
        if not run_dir.exists():
            continue
        hist = pd.read_csv(run_dir / 'history.csv')
        label = f'Δ={delay}'
        axes[0, 0].plot(hist['R'], alpha=0.7, label=label)
        axes[0, 1].plot(hist['alarm_true'], alpha=0.7, label=label)
        axes[1, 0].plot(hist['avg_repression'], alpha=0.7, label=label)
        axes[1, 1].plot(hist['punished_fraction'], alpha=0.7, label=label)

    axes[0, 0].set_ylabel('Radical fraction $x_R$')
    axes[0, 1].set_ylabel('Alarm')
    axes[1, 0].set_ylabel('Avg repression prob')
    axes[1, 1].set_ylabel('Punished fraction')
    for ax in axes[1, :]:
        ax.set_xlabel('Step')
    axes[0, 0].legend(fontsize=7)
    fig.suptitle('Delay Sweep — Representative Trajectories (k=20, 500 steps, content-neutral)')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_delay_sweep.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'fig_delay_sweep.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved fig_delay_sweep')


def fig_phase_diagram():
    summary = pd.read_csv(RESULTS / 'k_sweep_aggressive' / 'summary.csv')
    delay_data = pd.read_csv(RESULTS / 'delay_sweep_aggressive' / 'summary.csv')
    combined = pd.concat([summary, delay_data], ignore_index=True)

    regime_map = {'stable': 0, 'oscillatory': 1, 'runaway': 2}
    combined['regime_code'] = combined['regime'].map(regime_map).fillna(0)

    pivot = combined.groupby(['delay_steps', 'k'])['regime_code'].mean()
    if hasattr(pivot, 'unstack'):
        pivot = pivot.unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, aspect='auto', origin='lower', cmap='RdYlGn_r',
                   vmin=0, vmax=2)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f'{x:.0f}' for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f'{y}' for y in pivot.index])
    ax.set_xlabel('Sharpness $k$')
    ax.set_ylabel('Delay steps')
    fig.colorbar(im, ax=ax, label='Mean regime (0=stable, 1=osc, 2=runaway)')
    fig.suptitle('Phase Diagram (delay × sharpness, content-neutral, 50 seeds)')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_phase_diagram.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'fig_phase_diagram.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved fig_phase_diagram')


def fig_crossed_delay_learning():
    summary = pd.read_csv(RESULTS / 'delay_x_learning_3way' / 'summary.csv')
    delays = sorted(summary['delay_steps'].unique())

    archs = [
        ('fixed', 's-', 'Non-reactive (fixed policy)', '#2ecc71'),
        ('reactive', '^-', 'Reactive (threshold heuristic)', '#e74c3c'),
        ('q_learning', 'o-', 'Q-learning (adaptive)', '#3498db'),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    for mode, style, label, color in archs:
        sub = summary[summary['learning_mode'] == mode]
        rates = []
        cis_lo, cis_hi = [], []
        for d in delays:
            s = sub[sub['delay_steps'] == d]
            n = len(s)
            r = (s['regime'] == 'runaway').sum()
            p = r / n
            z = 1.96
            denom = 1 + z ** 2 / n
            center = (p + z ** 2 / (2 * n)) / denom
            margin = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
            rates.append(p)
            cis_lo.append(max(0, center - margin))
            cis_hi.append(min(1, center + margin))
        ax.plot(delays, rates, style, label=label, color=color, linewidth=2, markersize=8)
        ax.fill_between(delays, cis_lo, cis_hi, alpha=0.15, color=color)

    ax.set_xlabel('Repression delay (steps)')
    ax.set_ylabel('Runaway rate')
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.0)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_crossed_delay_learning.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'fig_crossed_delay_learning.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved fig_crossed_delay_learning')


def fig_rl_regulator():
    summary = pd.read_csv(RESULTS / 'learning_ablation' / 'summary.csv')
    conditions = ['fixed_agents_static_regulator', 'rl_agents_static_regulator', 'rl_agents_rl_regulator']
    labels = ['Fixed agents\n(static reg.)', 'Q-learning\n(static reg.)', 'Q-learning\n(RL reg.)']
    regimes = ['stable', 'oscillatory', 'runaway']
    colors = ['#2ecc71', '#f39c12', '#e74c3c']

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(conditions))
    width = 0.25
    for i, (regime, color) in enumerate(zip(regimes, colors)):
        vals = []
        for cond in conditions:
            sub = summary[summary['condition'] == cond]
            vals.append((sub['regime'] == regime).mean())
        ax.bar(x + i * width, vals, width, label=regime.capitalize(), color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Fraction of runs')
    ax.set_title('RL Regulator Effect (500 steps, 50 seeds)')
    ax.legend()
    ax.set_ylim(0, 1.0)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_rl_regulator.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'fig_rl_regulator.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved fig_rl_regulator')


if __name__ == '__main__':
    print('Generating Paper 1 v2 figures...')
    fig_delay_sweep()
    fig_crossed_delay_learning()
    fig_rl_regulator()
    print('Done.')
