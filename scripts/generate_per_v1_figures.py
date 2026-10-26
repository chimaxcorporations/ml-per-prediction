#!/usr/bin/env python3
"""Recreate the five PER Benchmark V1 manuscript figures in the project folder.

Run from repository root: python scripts/generate_per_v1_figures.py
Uses verified overall diagnostic metrics, PER bins and scenarios; the five
baseline and within-domain RMSEs below are the published V1 benchmark values.
"""
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASELINE = {
    'persistence': (0.198680, 0.203748),
    'ridge':       (0.183564, 0.220791),
    'histgb':      (0.195276, 0.201516),
    'mlp':         (0.188441, 0.195058),
    'cnn':         (0.206729, 0.287960),
}
WITHIN = {
    'persistence': (0.203748, 0.198680),
    'ridge':       (0.176019, 0.187399),
    'histgb':      (0.152728, 0.199469),
    'mlp':         (0.154547, 0.190530),
    'cnn':         (0.153402, 0.191086),
}
TEMPORAL = ('gru', 'lstm', 'tcn', 'transformer')
DIRECTIONS = ('juelich_to_berlin', 'berlin_to_juelich')
DIRECTION_LABELS = ('Jülich → Berlin', 'Berlin → Jülich')


def save(fig, directory, stem, dpi):
    fig.savefig(directory / (stem + '.pdf'), bbox_inches='tight')
    fig.savefig(directory / (stem + '.png'), dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print('Saved', directory / (stem + '.pdf'))


def grouped_bars(ax, names, first, second, labels, ylabel='RMSE'):
    x = np.arange(len(names))
    w = 0.37
    ax.bar(x - w/2, first, w, label=labels[0])
    ax.bar(x + w/2, second, w, label=labels[1])
    ax.set_xticks(x, [n.upper() if n != 'transformer' else 'Transformer' for n in names], rotation=25, ha='right')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=.22)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=Path('.'))
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Default: <project>/results/manuscript_v1/figures')
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = (args.output_dir if args.output_dir else root / 'results/manuscript_v1/figures').resolve()
    out.mkdir(parents=True, exist_ok=True)

    bins_path = root / 'results/temporal_checkpoint_diagnostics_v1/error_by_bin.csv'
    scenario_path = root / 'results/temporal_checkpoint_diagnostics_v1/error_summary.csv'
    counts_path = root / 'results/temporal_transfer_v1/parameter_counts.csv'
    for p in (bins_path, scenario_path, counts_path):
        if not p.exists():
            raise FileNotFoundError(f'Missing input: {p}')

    bins = pd.read_csv(bins_path)
    scenarios = pd.read_csv(scenario_path)
    counts = pd.read_csv(counts_path)
    for frame, cols in ((bins, ('direction','model','variable','bin','bias')),
                        (scenarios, ('direction','level','scenario','model','samples','rmse')),
                        (counts, ('direction','model','parameters'))):
        missing = set(cols) - set(frame.columns)
        if missing:
            raise ValueError(f'Missing CSV columns: {sorted(missing)}')

    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42,
                         'savefig.facecolor': 'white'})

    # Figure 1: nine models, bidirectional zero-shot transfer.
    names = list(BASELINE) + list(TEMPORAL)
    transfer = dict(BASELINE)
    for model in TEMPORAL:
        values = []
        for direction in DIRECTIONS:
            rows = scenarios[(scenarios.direction == direction) &
                             (scenarios.level == 'overall') &
                             (scenarios.model == model)]
            if len(rows) != 1:
                raise ValueError(f'Expected exactly one verified overall row for {direction}/{model}; found {len(rows)}')
            values.append(float(rows.iloc[0].rmse))
        transfer[model] = tuple(values)
    fig, ax = plt.subplots(figsize=(9.0, 3.5))
    grouped_bars(ax, names, [transfer[n][0] for n in names],
                 [transfer[n][1] for n in names], DIRECTION_LABELS)
    ax.set_title('Zero-shot cross-dataset PER prediction')
    fig.tight_layout()
    save(fig, out, 'fig1_transfer_rmse', args.dpi)

    # Figure 2: five baseline models, within-domain (different CV protocols).
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    grouped_bars(ax, list(WITHIN), [WITHIN[n][0] for n in WITHIN],
                 [WITHIN[n][1] for n in WITHIN], ('Jülich within-domain', 'Berlin within-domain'))
    ax.set_title('Within-domain benchmark (different validation protocols)')
    fig.tight_layout()
    save(fig, out, 'fig2_within_domain', args.dpi)

    # Figure 3: temporal parameter count versus transfer RMSE.
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    for direction, label, marker in zip(DIRECTIONS, DIRECTION_LABELS, ('o', 's')):
        x, y = [], []
        for name in TEMPORAL:
            cr = counts[(counts.direction == direction) & (counts.model == name)]
            if len(cr) != 1:
                raise ValueError(f'Parameter count missing/duplicated: {direction}/{name}')
            x.append(int(cr.iloc[0].parameters))
            y.append(transfer[name][DIRECTIONS.index(direction)])
        ax.scatter(x, y, marker=marker, s=65, label=label)
        for name, xx, yy in zip(TEMPORAL, x, y):
            ax.annotate(name.upper() if name != 'transformer' else 'Transformer',
                        (xx, yy), xytext=(5, 4 if marker == 'o' else -11),
                        textcoords='offset points', fontsize=8)
    ax.set_xlabel('Trainable parameters')
    ax.set_ylabel('Zero-shot RMSE')
    ax.set_title('Temporal-model complexity and generalization')
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out, 'fig3_complexity', args.dpi)

    # Figure 4: bias conditioned on *observed* PER; do not interpolate missing bins.
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7), sharey=True)
    for ax, direction, label in zip(axes, DIRECTIONS, DIRECTION_LABELS):
        subset = bins[(bins.direction == direction) & (bins.variable == 'true_per')]
        for model in TEMPORAL:
            part = subset[subset.model == model].copy()
            part['left'] = part['bin'].map(lambda s: float(re.match(r'\[([0-9.]+),', str(s)).group(1)))
            part = part.sort_values('left')
            ax.plot(part.left.to_numpy(), part.bias.to_numpy(), marker='o',
                    markersize=3.5, linewidth=1.3, label=model.upper() if model != 'transformer' else 'Transformer')
        ax.axhline(0, color='black', linewidth=.8, linestyle='--')
        ax.set_xlim(-.025, 1.0)
        ax.set_xlabel('Observed PER bin lower bound')
        ax.set_title(label)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean prediction bias')
    axes[1].legend(frameon=False, fontsize=8, loc='lower left')
    fig.suptitle('Conditional error by observed packet-loss regime')
    fig.tight_layout()
    save(fig, out, 'fig4_per_bias', args.dpi)

    # Figure 5: aggregate squared scenario RMSE, weighted by samples.
    subset = scenarios[(scenarios.direction == 'berlin_to_juelich') & (scenarios.level == 'scenario')].copy()
    def propagation(name):
        if '_log_distance_nakagami_' in name:
            return 'Nakagami'
        if '_log_distance_' in name:
            return 'Log-distance'
        if '_urban_' in name:
            return 'Urban'
        raise ValueError(f'Unrecognized scenario: {name}')
    subset['propagation'] = subset.scenario.map(propagation)
    props = ('Log-distance', 'Nakagami', 'Urban')
    matrix = np.zeros((len(props), len(TEMPORAL)))
    for i, prop in enumerate(props):
        for j, model in enumerate(TEMPORAL):
            g = subset[(subset.propagation == prop) & (subset.model == model)]
            if len(g) != 9:
                raise ValueError(f'Expected nine scenarios for {prop}/{model}; found {len(g)}')
            matrix[i, j] = np.sqrt(np.average(g.rmse.to_numpy()**2, weights=g.samples.to_numpy()))
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    x = np.arange(len(props))
    w = .19
    for j, model in enumerate(TEMPORAL):
        ax.bar(x + (j-1.5)*w, matrix[:, j], w,
               label=model.upper() if model != 'transformer' else 'Transformer')
    ax.set_xticks(x, props)
    ax.set_ylabel('Pooled RMSE')
    ax.set_title('Berlin → Jülich: propagation-dependent transfer')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    save(fig, out, 'fig5_propagation', args.dpi)

    print(f'All five figures saved as PDF and PNG in {out}')
    print('Note: within-domain and baseline transfer scores are fixed V1 values in this script.')


if __name__ == '__main__':
    main()
