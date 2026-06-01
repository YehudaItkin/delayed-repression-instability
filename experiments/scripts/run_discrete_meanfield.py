#!/usr/bin/env python3
"""Discrete mean-field bridge: x_{t+1} = x_t + η·x_t(1-x_t)[a - C·p(x_{t-d})]

Bridges the gap between continuous ODE (where Hopf bifurcation is exact)
and the full agent simulation. By sweeping η (step size), we isolate how
much instability discretization alone consumes."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT.parent / 'paper1' / 'latex' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)
OUTDIR = ROOT / 'results' / 'paper1' / 'discrete_meanfield'
OUTDIR.mkdir(parents=True, exist_ok=True)


def sigmoid(x, k, x_c):
    return 1.0 / (1.0 + np.exp(-k * (x - x_c)))


def run_discrete_mf(a, C, k, x_c, delay_steps, eta, T, x0=None):
    if x0 is None:
        ratio = a / C
        if 0 < ratio < 1:
            x0 = x_c + (1.0 / k) * np.log(ratio / (1.0 - ratio)) + 0.05
            x0 = np.clip(x0, 0.05, 0.95)
        else:
            x0 = 0.3
    x_hist = np.zeros(T + 1)
    x_hist[:delay_steps + 1] = x0
    for t in range(delay_steps, T):
        x = x_hist[t]
        x_del = x_hist[max(0, t - delay_steps)]
        p_del = sigmoid(x_del, k, x_c)
        x_next = x + eta * x * (1.0 - x) * (a - C * p_del)
        x_hist[t + 1] = np.clip(x_next, 1e-8, 1.0 - 1e-8)
    return x_hist


def compute_x_star(a, C, k, x_c):
    ratio = a / C
    if ratio <= 0 or ratio >= 1:
        return None
    return x_c + (1.0 / k) * np.log(ratio / (1.0 - ratio))


def compute_delta_c(a, C, k, x_star):
    if x_star is None or not (0 < x_star < 1):
        return float('inf')
    p_prime = k * (a / C) * (1.0 - a / C)
    denom = 2.0 * C * x_star * (1.0 - x_star) * p_prime
    if denom <= 0:
        return float('inf')
    return np.pi / denom


def sweep_eta_delay():
    a, C, k, x_c = 2.0, 7.0, 10.0, 0.72
    x_star = compute_x_star(a, C, k, x_c)
    dc_cont = compute_delta_c(a, C, k, x_star)
    T = 2000

    etas = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    delays = list(range(0, 61, 2))

    rows = []
    for eta in etas:
        for d in delays:
            x_hist = run_discrete_mf(a, C, k, x_c, d, eta, T)
            tail = x_hist[T // 2:]
            amp = float(tail.max() - tail.min())
            std = float(tail.std())
            mx = float(x_hist.max())
            regime = 'runaway' if mx > 0.95 else ('oscillatory' if amp > 0.05 else 'stable')
            rows.append({
                'eta': eta, 'delay_steps': d, 'amp_tail': amp,
                'std_tail': std, 'max_x': mx, 'regime': regime,
                'delta_c_continuous': dc_cont,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / 'eta_delay_sweep.csv', index=False)
    print(f'Saved {len(df)} runs to {OUTDIR / "eta_delay_sweep.csv"}')
    return df


def sweep_k_delay():
    a, C, x_c = 2.0, 7.0, 0.72
    eta = 0.1
    T = 2000

    k_values = [5, 10, 15, 20, 30, 40, 50]
    delays = list(range(0, 61, 2))

    rows = []
    for k in k_values:
        x_star = compute_x_star(a, C, k, x_c)
        dc = compute_delta_c(a, C, k, x_star)
        for d in delays:
            x_hist = run_discrete_mf(a, C, k, x_c, d, eta, T)
            tail = x_hist[T // 2:]
            amp = float(tail.max() - tail.min())
            regime = 'runaway' if x_hist.max() > 0.95 else ('oscillatory' if amp > 0.05 else 'stable')
            rows.append({
                'k': k, 'delay_steps': d, 'amp_tail': amp,
                'max_x': float(x_hist.max()), 'regime': regime,
                'x_star': x_star, 'delta_c': dc,
                'delay_over_dc': d * eta / dc if dc < float('inf') else 0,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / 'k_delay_sweep.csv', index=False)
    print(f'Saved {len(df)} runs to {OUTDIR / "k_delay_sweep.csv"}')
    return df


def fig_eta_phase_diagram(df):
    regime_map = {'stable': 0, 'oscillatory': 1, 'runaway': 2}
    df['regime_code'] = df['regime'].map(regime_map)

    etas = sorted(df['eta'].unique(), reverse=True)
    delays = sorted(df['delay_steps'].unique())

    fig, axes = plt.subplots(1, len(etas), figsize=(3 * len(etas), 4), sharey=True)
    for idx, eta in enumerate(etas):
        ax = axes[idx]
        sub = df[df['eta'] == eta].sort_values('delay_steps')
        colors = ['green' if r == 'stable' else 'orange' if r == 'oscillatory' else 'red'
                  for r in sub['regime']]
        ax.bar(range(len(sub)), sub['amp_tail'], color=colors, width=0.8)
        ax.set_title(f'η={eta}', fontsize=9)
        ax.set_xlabel('delay steps')
        ax.set_xticks(range(0, len(sub), 5))
        ax.set_xticklabels([str(delays[i]) for i in range(0, len(delays), 5)], fontsize=7)
        if idx == 0:
            ax.set_ylabel('Tail amplitude')

    fig.suptitle('Discrete Mean-Field: η controls effective stability margin\n'
                 'Green=stable, Orange=oscillatory, Red=runaway', fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_discrete_mf_eta.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'fig_discrete_mf_eta.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGDIR / "fig_discrete_mf_eta.pdf"}')


def fig_k_phase_diagram(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    k_values = sorted(df['k'].unique())
    delays = sorted(df['delay_steps'].unique())

    regime_map = {'stable': 0, 'oscillatory': 1, 'runaway': 2}
    pivot = df.pivot_table(index='k', columns='delay_steps',
                           values='regime', aggfunc=lambda x: regime_map.get(x.iloc[0], 0))
    im = ax.imshow(pivot.values, aspect='auto', origin='lower', cmap='RdYlGn_r',
                   vmin=0, vmax=2)
    ax.set_xticks(range(0, len(delays), 5))
    ax.set_xticklabels([str(delays[i]) for i in range(0, len(delays), 5)])
    ax.set_yticks(range(len(k_values)))
    ax.set_yticklabels([str(k) for k in k_values])
    ax.set_xlabel('Delay steps')
    ax.set_ylabel('Sharpness k')
    fig.colorbar(im, ax=ax, label='0=stable, 1=osc, 2=runaway')
    ax.set_title('Discrete Mean-Field Phase Diagram (η=0.1)')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_discrete_mf_phase.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'fig_discrete_mf_phase.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGDIR / "fig_discrete_mf_phase.pdf"}')


def fig_trajectories():
    a, C, k, x_c = 2.0, 7.0, 10.0, 0.72
    x_star = compute_x_star(a, C, k, x_c)
    T = 500
    eta = 0.1

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    configs = [(0, 'delay=0'), (5, 'delay=5'), (10, 'delay=10'),
               (15, 'delay=15'), (25, 'delay=25'), (40, 'delay=40')]

    for idx, (d, label) in enumerate(configs):
        ax = axes[idx // 3, idx % 3]
        x_hist = run_discrete_mf(a, C, k, x_c, d, eta, T)
        ax.plot(x_hist, 'b-', linewidth=0.8)
        if x_star and 0 < x_star < 1:
            ax.axhline(x_star, color='r', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, 1)
        if idx // 3 == 1:
            ax.set_xlabel('Step')
        if idx % 3 == 0:
            ax.set_ylabel('x(t)')

    fig.suptitle(f'Discrete Mean-Field Trajectories (η={eta}, k={k})', fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_discrete_mf_trajectories.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'fig_discrete_mf_trajectories.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGDIR / "fig_discrete_mf_trajectories.pdf"}')


if __name__ == '__main__':
    print('=== Discrete Mean-Field Calibration ===')
    fig_trajectories()
    df_eta = sweep_eta_delay()
    fig_eta_phase_diagram(df_eta)
    df_k = sweep_k_delay()
    fig_k_phase_diagram(df_k)
    print('Done.')
