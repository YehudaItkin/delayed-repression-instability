#!/usr/bin/env python3
"""Numerically integrate the delayed replicator ODE and compare bifurcation
behavior to the agent-based simulation. This validates the theory in its own
terms, independent of the multi-agent simulation."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from autonomy_lab.theory import x_star_sigmoid, delta_c_sigmoid

FIGDIR = ROOT.parent / 'paper1' / 'latex' / 'figures'
FIGDIR.mkdir(parents=True, exist_ok=True)


def sigmoid(x, k, x_c):
    return 1.0 / (1.0 + np.exp(-k * (x - x_c)))


def delayed_replicator_step(x_hist, t_idx, dt, a, C, k, x_c, delay_steps):
    x_now = x_hist[t_idx]
    x_delayed = x_hist[max(0, t_idx - delay_steps)]
    p_delayed = sigmoid(x_delayed, k, x_c)
    dxdt = x_now * (1.0 - x_now) * (a - C * p_delayed)
    x_next = x_now + dt * dxdt
    return np.clip(x_next, 1e-8, 1.0 - 1e-8)


def run_ode(a, C, k, x_c, delay, dt=0.01, T=50.0, x0=None):
    steps = int(T / dt)
    delay_steps = int(delay / dt)
    xs = x_star_sigmoid(a, C, k, x_c)
    if x0 is None:
        x0 = (xs + 0.05) if xs and 0 < xs < 1 else 0.3
    x_hist = np.zeros(steps + 1)
    x_hist[0] = x0
    for t in range(steps):
        x_hist[t + 1] = delayed_replicator_step(x_hist, t, dt, a, C, k, x_c, delay_steps)
    time = np.arange(steps + 1) * dt
    return time, x_hist, xs


def fig_ode_delay_sweep():
    a, C, k, x_c = 2.0, 7.0, 10.0, 0.72
    xs = x_star_sigmoid(a, C, k, x_c)
    dc = delta_c_sigmoid(a, C, k, xs) if xs else float('inf')
    print(f'Theory: x* = {xs:.4f}, Delta_c = {dc:.4f}')

    delay_fracs = [0.0, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0]
    delays = [f * dc for f in delay_fracs]

    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=True, sharey=True)
    for idx, (frac, delay) in enumerate(zip(delay_fracs, delays)):
        ax = axes[idx // 4, idx % 4]
        time, x_hist, _ = run_ode(a, C, k, x_c, delay, T=80.0)
        ax.plot(time, x_hist, 'b-', linewidth=0.8)
        if xs:
            ax.axhline(xs, color='r', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_title(f'Δ/Δc = {frac}', fontsize=9)
        ax.set_ylim(0, 1)
        if idx // 4 == 1:
            ax.set_xlabel('Time')
        if idx % 4 == 0:
            ax.set_ylabel('x(t)')

    fig.suptitle(f'Delayed Replicator ODE: a={a}, C={C}, k={k}\n'
                 f'x*={xs:.3f}, Δc={dc:.3f}', fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_ode_delay_sweep.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'fig_ode_delay_sweep.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGDIR / "fig_ode_delay_sweep.pdf"}')


def fig_ode_k_sweep():
    a, C, x_c = 2.0, 7.0, 0.72
    k_values = [5, 10, 15, 20, 30, 50]

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
    for idx, k in enumerate(k_values):
        ax = axes[idx // 3, idx % 3]
        xs = x_star_sigmoid(a, C, k, x_c)
        dc = delta_c_sigmoid(a, C, k, xs) if xs else float('inf')
        delay = 1.5 * dc if dc < float('inf') else 1.0
        time, x_hist, _ = run_ode(a, C, k, x_c, delay, T=80.0)
        ax.plot(time, x_hist, 'b-', linewidth=0.8)
        if xs:
            ax.axhline(xs, color='r', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.set_title(f'k={k}, Δ/Δc=1.5\nΔc={dc:.3f}', fontsize=9)
        ax.set_ylim(0, 1)
        if idx // 3 == 1:
            ax.set_xlabel('Time')
        if idx % 3 == 0:
            ax.set_ylabel('x(t)')

    fig.suptitle(f'Sharpness sweep at Δ/Δc=1.5: a={a}, C={C}', fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_ode_k_sweep.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'fig_ode_k_sweep.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGDIR / "fig_ode_k_sweep.pdf"}')


def fig_ode_bifurcation():
    a, C, k, x_c = 2.0, 7.0, 10.0, 0.72
    xs = x_star_sigmoid(a, C, k, x_c)
    dc = delta_c_sigmoid(a, C, k, xs) if xs else float('inf')

    fracs = np.linspace(0.0, 4.0, 80)
    max_x = []
    min_x = []
    for frac in fracs:
        delay = frac * dc
        _, x_hist, _ = run_ode(a, C, k, x_c, delay, T=120.0)
        tail = x_hist[len(x_hist) // 2:]
        max_x.append(float(tail.max()))
        min_x.append(float(tail.min()))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.fill_between(fracs, min_x, max_x, alpha=0.3, color='steelblue')
    ax.plot(fracs, max_x, 'b-', linewidth=1, label='max x(t) tail')
    ax.plot(fracs, min_x, 'b--', linewidth=1, label='min x(t) tail')
    ax.axvline(1.0, color='red', linestyle=':', label='Δ/Δc = 1')
    if xs:
        ax.axhline(xs, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Δ / Δc')
    ax.set_ylabel('x(t) range in tail')
    ax.set_title('Bifurcation Diagram: Delayed Replicator ODE')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_ode_bifurcation.pdf', bbox_inches='tight')
    fig.savefig(FIGDIR / 'fig_ode_bifurcation.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {FIGDIR / "fig_ode_bifurcation.pdf"}')


if __name__ == '__main__':
    print('=== ODE Validation ===')
    fig_ode_delay_sweep()
    fig_ode_k_sweep()
    fig_ode_bifurcation()
    print('Done.')
