"""Assemble Chapter 5 figures from local exports; zero training or inference."""
from pathlib import Path
import argparse
import hashlib
import json
import os

import numpy as np

from .chapter5_io import load_snapshot, write_json


METHODS = ['SyncFull', 'SyncGreedy', 'AsyncTraj']
COLORS = ['#E24A33', '#348ABD', '#8EBA42', '#988ED5']
EVALUATION = '#E53935'
STYLE = {'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.labelsize': 8,
         'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
         'legend.fontsize': 7, 'axes.linewidth': .6, 'pdf.fonttype': 42,
         'ps.fonttype': 42, 'path.simplify': False}


def further_smooth(x, y, width=5000):
    """Centered integral average of the MA20 polyline, in iteration units.

    Symmetric windows shorten at the endpoints; endpoint values are retained.
    This is a display transformation after MA20, not a raw 5000-sample average.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return x, y
    if not np.all(np.diff(x) > 0):
        raise ValueError('Moving-average indices must increase strictly')
    slope = np.diff(y)/np.diff(x)
    area = np.r_[0, np.cumsum(np.diff(x)*(y[:-1]+y[1:])/2)]
    def integral(t):
        i = np.clip(np.searchsorted(x, t, side='right')-1, 0, len(x)-2)
        d = t-x[i]
        return area[i]+y[i]*d+.5*slope[i]*d*d
    grid = np.linspace(x[0], x[-1], min(2001, len(x)))
    radius = np.minimum(width/2, np.minimum(grid-x[0], x[-1]-grid))
    answer = np.interp(grid, x, y)
    inside = radius > 1e-8
    answer[inside] = (integral(grid[inside]+radius[inside])-
                      integral(grid[inside]-radius[inside]))/(2*radius[inside])
    return grid, answer


class Figures:
    def __init__(self, output, allow_partial=False):
        # Imports happen only after the command selects local cache directories.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        self.plt = plt
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.partial = allow_partial
        self.records, self.sources = {}, []
        plt.rcdefaults()
        plt.rcParams.update(STYLE)

    def save(self, fig, name, **metadata):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for text in fig.findobj(match=self.plt.Text):
            if text.get_visible() and text.get_text().strip():
                box = text.get_window_extent(renderer)
                if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
                    raise ValueError(f'Text outside figure canvas: {name}: {text.get_text()}')
        path = self.output/(name+'.pdf')
        fig.savefig(path, metadata={'Title': name, 'Creator': 'Matplotlib; local numerical exports; plot only'})
        fig.savefig(path.with_suffix('.png'), dpi=200)
        self.records[name] = dict(pdf=path.name, png=path.with_suffix('.png').name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            inches=list(fig.get_size_inches()), **metadata)
        self.plt.close(fig)
        print('Exported '+path.name, flush=True)

    def read(self, folder, kind):
        values, descriptor = load_snapshot(folder, kind)
        self.sources.append(dict(folder=str(Path(folder).resolve()), kind=kind,
                                 index=descriptor['index'], sha256=descriptor['sha256']))
        return values, descriptor

    def validate_run(self, folder, kind, config_name):
        from .utils import load_config
        info_path = Path(folder)/'chapter5_run.json'
        if not info_path.is_file():
            raise ValueError(f'Missing {info_path}; run the updated training entrypoint first')
        info = json.loads(info_path.read_text())
        if info['kind'] != kind:
            raise ValueError(f'{folder} is {info["kind"]}, expected {kind}')
        reference = load_config(Path(__file__).resolve().parents[2]/'configs'/config_name)
        config = info['config']
        # The lightweight config parser recognizes unquoted "none" as null.
        # Both spellings identify the deterministic planning environment.
        for settings in [reference, config]:
            noise = settings.get('common_noise', {})
            if 'mode' in noise and noise['mode'] is None:
                noise['mode'] = 'none'
        keys = ['seed', 'device', 'model', 'training', 'network', 'common_noise', 'solvers']
        differing = [key for key in keys if config.get(key) != reference.get(key)]
        if differing and not self.partial:
            raise ValueError(f'{folder}: settings differ from {config_name}: {differing}; '
                             'use --allow-partial only for smoke/custom figures')
        return config

    @staticmethod
    def axis(ax):
        ax.tick_params(length=2.5, width=.6, pad=2)
        ax.set_axisbelow(True)
        ax.grid(True, ls='--', lw=.45, color='#b0b0b0', alpha=.6)

    def trajectories(self, records, labels, name):
        from matplotlib.lines import Line2D
        height = 1.2*len(records)+.68
        fig, axes = self.plt.subplots(len(records), 4, figsize=(6, height), squeeze=False)
        fig.subplots_adjust(left=.13, right=.995, bottom=.60/height,
                            top=1-.08/height, hspace=.10, wspace=.12)
        for i, (record, label) in enumerate(zip(records, labels)):
            data, _ = record
            time = data['time']
            if data['trajectories'].shape != (4, len(time), 4):
                raise ValueError('Cybersecurity figures require four starts and four state coordinates')
            for j, ax in enumerate(axes[i]):
                if 'reference' in data:
                    for k, color in enumerate(COLORS):
                        ax.plot(data['reference_time'], data['reference'][j, :, k],
                                color=color, lw=2.1, ls=(0, (4, 2.3)), alpha=.65, zorder=3)
                for k, color in enumerate(COLORS):
                    ax.plot(time, data['trajectories'][j, :, k], color=color, lw=1.05, zorder=5)
                padding = .05*(time[-1]-time[0])
                ax.set_xlim(time[0]-padding, time[-1]+padding); ax.set_ylim(0, 1.05)
                ax.set_xticks(np.linspace(time[0], time[-1], 3)); ax.set_yticks([0, .5, 1])
                ax.tick_params(labelleft=j==0, labelbottom=i==len(records)-1)
                if j==0: ax.set_ylabel('Proportion', labelpad=3)
                if i==len(records)-1: ax.set_xlabel('Time', labelpad=2)
                self.axis(ax)
                ax.set_xticks(time[0]+(time[-1]-time[0])*np.array([.25, .75]), minor=True)
                ax.set_yticks([.25, .75], minor=True)
                ax.tick_params(which='minor', length=0, labelbottom=False, labelleft=False)
                ax.grid(True, which='minor', ls='--', lw=.45, color='#b0b0b0', alpha=.6)
            pos = axes[i, 0].get_position()
            fig.text(.018, (pos.y0+pos.y1)/2, label, rotation=90, va='center',
                     ha='center', fontsize=8, fontweight='bold')
        handles = [Line2D([], [], color=c, lw=1.2, label=l)
                   for c, l in zip(COLORS, ['DI', 'DS', 'UI', 'US'])]
        handles += [Line2D([], [], color='#555555', lw=1.05, label='Learned'),
                    Line2D([], [], color='#555555', lw=2.1, ls=(0, (4, 2.3)), alpha=.65, label='Reference')]
        fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False,
                   fontsize=7, handlelength=3.2, columnspacing=1.2, borderaxespad=.2)
        self.save(fig, name, reference='Forward-backward ODE, constant attack rate')

    def training(self, records, labels, name, planning=False):
        from matplotlib.ticker import MaxNLocator, ScalarFormatter
        fig, axes = self.plt.subplots(1, len(records), figsize=(6, 2.5 if planning else 2.3), squeeze=False)
        fig.subplots_adjust(left=.085 if planning else .12, right=.98 if planning else .96,
                            bottom=.20, top=.87, wspace=.32 if planning else (.45 if len(records)==3 else .30))
        smoothing = []
        for ax, (data, descriptor), label in zip(axes[0], records, labels):
            costs, indices = -data['rewards'], data['indices']
            window = descriptor['metadata']['moving_average_window']
            extra = label=='AsyncTraj'
            if len(costs):
                ax.plot(indices, costs, color='#348ABD', lw=.55, alpha=.28, label='Training cost', zorder=2)
                if len(costs) >= window:
                    mean = np.convolve(costs, np.ones(window)/window, mode='valid')
                    x = indices[window-1:]
                    if extra: x, mean = further_smooth(x, mean)
                    ax.plot(x, mean, color='#1f5f8b', lw=1.15,
                            label='Moving average' if extra else f'Moving average ({window})', zorder=4)
            ax.plot(data['eval_indices'], -data['eval_rewards'], color=EVALUATION,
                    lw=1.2, ls=(0, (3.5, 2)), marker=None if len(costs) else 'o',
                    markersize=3, label='Evaluation\n(mean over starts)' if len(costs) else 'Evaluation', zorder=5)
            ax.set_title(label, pad=7, fontsize=9 if planning else 8, fontweight='bold')
            maximum = max(descriptor['index'], 1)
            ticks = np.linspace(0, maximum, 4) if planning else [0, maximum/2, maximum]
            if planning and maximum < 3: ticks = np.arange(maximum+1)
            ax.set_xticks(ticks)
            formatter = ScalarFormatter(useOffset=False); formatter.set_scientific(False)
            ax.xaxis.set_major_formatter(formatter)
            lo, hi = ax.get_ylim()
            ax.set_yticks([tick for tick in MaxNLocator(nbins=4).tick_values(lo, hi) if lo<=tick<=hi])
            ax.set_xlabel('Episodes' if planning else 'Iterations', labelpad=2)
            ax.set_ylabel('Total cost' if len(costs) else 'Average cost', labelpad=2)
            self.axis(ax)
            if planning:
                ax.set_ylim(0, max(110, np.max(costs)*1.02 if len(costs) else 110))
                ax.set_yticks([0, 25, 50, 75, 100] if ax.get_ylim()[1]==110 else ax.get_yticks())
                ax.tick_params(labelsize=8); ax.xaxis.label.set_size(9); ax.yaxis.label.set_size(9)
            legend = ax.legend(loc='upper right', frameon=True, framealpha=.9,
                              fontsize=7.4 if planning else 6.6, handlelength=1.4,
                              borderpad=.3, labelspacing=.35)
            if len(costs):
                legend.legend_handles[0].set_alpha(.7)
                legend.legend_handles[0].set_linewidth(.9)
            smoothing.append(dict(label=label, MA_window=window,
                                  additional_polyline_integral_window=5000 if extra else 0))
        self.save(fig, name, smoothing=smoothing, evaluation='Mean over starts and their evaluation rollouts',
                  cost_convention='Negative undiscounted episode total reward')

    def simplex(self, records, labels, name, ddpg=False):
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        rows = len(records)*(2 if ddpg else 1)
        height = .865*rows+.14*(rows-1)+.61
        fig = self.plt.figure(figsize=(6, height))
        grid = fig.add_gridspec(rows, 6, left=.095, right=.90, bottom=.32/height,
            top=1-.29/height, width_ratios=[1, 1, 1, 1, 1, .085], wspace=.12, hspace=.14/.865)
        r, scales = 0, []
        middle = .070 if ddpg else .100
        show_middle = all(description['metadata']['vmin']+.15*(description['metadata']['vmax']-description['metadata']['vmin']) < middle
                          < description['metadata']['vmax']-.15*(description['metadata']['vmax']-description['metadata']['vmin'])
                          for _, description in records)
        for (data, descriptor), label in zip(records, labels):
            slices = data['slices']
            if len(slices) != 5:
                raise ValueError('Book simplex layouts require five slices')
            low, high = descriptor['metadata']['vmin'], descriptor['metadata']['vmax']
            for field in (['rl', 'benchmark'] if ddpg else ['rl']):
                for j, di in enumerate(slices):
                    ax = fig.add_subplot(grid[r, j])
                    ax.pcolormesh(data['X'], data['Y'], data[field][j], cmap='magma',
                                  shading='auto', vmin=low, vmax=high)
                    # Preserve complete boundary cells, including their half-grid
                    # extension beyond the probability-coordinate tick centers.
                    ax.set_aspect('equal')
                    ax.set_xticks([0, .5, 1], labels=['0', '0.5', '1'])
                    ax.set_yticks([0, .5, 1], labels=['0', '0.5', '1'])
                    ax.tick_params(length=2, width=.6, pad=1.5, labelleft=j==0, labelbottom=r==rows-1)
                    if r==rows-1: ax.set_xlabel(r'$\mu_{DS}$', labelpad=1)
                    if j==0:
                        ax.set_ylabel(r'$\mu_{UI}$', labelpad=1.5)
                        pos = ax.get_position()
                        row_label = label+(' - '+('RL' if field=='rl' else 'Benchm.') if ddpg else '')
                        fig.text(.018, (pos.y0+pos.y1)/2, row_label, rotation=90,
                                 va='center', ha='center', fontsize=7.4, fontweight='bold')
                    if r==0: ax.set_title(rf'$\mu_{{DI}}={di:g}$', fontsize=8, pad=5)
                cax = fig.add_subplot(grid[r, 5])
                colorbar = fig.colorbar(ScalarMappable(norm=Normalize(low, high), cmap='magma'), cax=cax)
                ticks = [low, middle, high] if show_middle else [low, high]
                colorbar.set_ticks(ticks, labels=[f'{0.0 if round(value, 3)==0 else value:.3f}' for value in ticks])
                colorbar.ax.tick_params(labelsize=6.8, length=2, pad=2)
                if r==0: cax.set_title('Cost', fontsize=7.4, pad=6)
                scales.append(dict(vmin=low, vmax=high, ticks=ticks))
                r += 1
        self.save(fig, name, color_scales=scales, normalization='Unrounded exported limits',
                  benchmark='Forward-backward ODE reference, not an exact learned-policy value')

    def planning(self, record, name):
        from matplotlib.patches import Patch
        from matplotlib.colors import to_rgba
        data, _ = record
        if data['trajectories'].shape[0] != 4 or len(data['target']) != 11:
            raise ValueError('Planning book figures require four starts and eleven states')
        height = 5.22
        fig = self.plt.figure(figsize=(6, height))
        grid = fig.add_gridspec(4, 9, left=.115, right=.98, bottom=.38/height,
            top=1-.72/height, wspace=0, hspace=.12/.94,
            width_ratios=[1.15, .12, 1.15, .60, 1, .12, 1, .60, .9])
        axes = np.asarray([[fig.add_subplot(grid[i, 2*j]) for j in range(5)] for i in range(4)])
        titles = ['Initial\ndistribution', 'Final\ndistribution', 'Initial\nactions', 'Terminal\nactions', 'Cumulative\nCN']
        for j, title in enumerate(titles):
            pos = axes[0, j].get_position()
            fig.text((pos.x0+pos.x1)/2, 1-.035/height, title, ha='center', va='top',
                     fontsize=8.2, fontweight='bold', linespacing=1.05)
        for j, label, color in [(0, 'Initial', '#358635'), (1, 'Final', '#E24A33')]:
            pos = axes[0, j].get_position()
            handles = [Patch(facecolor=to_rgba('#348ABD', .22), edgecolor='#348ABD', lw=.4, label='Target'),
                       Patch(facecolor='none', edgecolor=color, lw=.95, label=label)]
            fig.legend(handles=handles, bbox_to_anchor=((pos.x0+pos.x1)/2, 1-.30/height),
                       loc='upper center', frameon=False, fontsize=6.8, handlelength=.9,
                       handletextpad=.45, borderaxespad=0, labelspacing=.22)
        action_colors = ['#009E73', '#E6B422', '#555555']
        start, end = axes[0, 2].get_position(), axes[0, 3].get_position()
        fig.legend(handles=[Patch(facecolor=to_rgba(c, .85), label=l) for c, l in
                    zip(action_colors, ['Left', 'Stay', 'Right'])],
            bbox_to_anchor=((start.x0+end.x1)/2, 1-.30/height), loc='upper center',
            frameon=False, ncol=3, fontsize=6.8, handlelength=.8,
            handletextpad=.35, columnspacing=.65, borderaxespad=0)
        X, ticks = np.arange(1, 12), [1, 3, 5, 7, 9, 11]
        noise = np.column_stack([np.zeros(4), np.cumsum(data['noise'], axis=1)])
        low, high = min(-2.2, noise.min()-.2), max(1.2, noise.max()+.2)
        zero = not np.any(noise)
        for i in range(4):
            pos = axes[i, 0].get_position()
            fig.text(.018, (pos.y0+pos.y1)/2, f'Case {i+1}', rotation=90,
                     fontweight='bold', fontsize=7.7, ha='center', va='center')
            for j, color in [(0, '#358635'), (1, '#E24A33')]:
                ax = axes[i, j]
                values = data['trajectories'][i, 0 if j==0 else -1]
                ax.bar(X, data['target'], width=.9, facecolor=to_rgba('#348ABD', .22),
                       edgecolor='#348ABD', lw=.4, align='center')
                ax.bar(X, values, width=.9, facecolor='none', edgecolor=color, lw=.95, align='center')
                ax.set_ylim(0, max(.45, values.max()*1.02, data['target'].max()*1.02))
                ax.set_yticks([0, .2, .4]); ax.set_xticks(ticks); ax.set_xlim(.5, 11.5)
                ax.set_axisbelow(True); ax.grid(axis='y', ls='--', lw=.4, color='#b0b0b0', alpha=.55)
                if j==0: ax.set_ylabel('Probability', fontsize=7.5, labelpad=2)
                else: ax.tick_params(labelleft=False)
            for j, key in [(2, 'actions_initial'), (3, 'actions_terminal')]:
                ax, values, bottom = axes[i, j], data[key][i], np.zeros(11)
                for k, color in enumerate(action_colors):
                    ax.bar(X, values[:, k], bottom=bottom, width=.8, align='center',
                           color=color, alpha=.85, lw=0)
                    bottom += values[:, k]
                ax.set_xlim(.5, 11.5); ax.set_ylim(0, 1.05)
                ax.set_xticks(ticks); ax.set_yticks([0, .5, 1])
                if j==3: ax.tick_params(labelleft=False)
            ax = axes[i, 4]
            ax.step(data['time'], noise[i], where='post', color='black', lw=1.05)
            ax.set_xlim(0, data['time'][-1]); ax.set_xticks([0, data['time'][-1]/2, data['time'][-1]])
            ax.set_ylim((-.055, .055) if zero else (low, high))
            ax.set_yticks([0] if zero else np.arange(np.ceil(low), high+1e-8))
            ax.set_axisbelow(True); ax.grid(ls='--', lw=.4, color='#b0b0b0', alpha=.55)
            for j, ax in enumerate(axes[i]):
                ax.tick_params(length=2, width=.6, pad=1.5, labelbottom=i==3)
                if i==3: ax.set_xlabel('State' if j<4 else 'Time', fontsize=7.5, labelpad=2)
        self.save(fig, name, average_displayed=False, actions_policy='online actor',
                  noise_paths='Paired with displayed first evaluation rollouts')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments = ['cyber-q-nocn', 'cyber-q-cn', 'cyber-ddpg-nocn', 'cyber-ddpg-cn', 'planning-nocn', 'planning-cn']
    for name in arguments: parser.add_argument('--'+name, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--allow-partial', action='store_true', help='Explicit smoke/custom settings; output is not the book reproduction')
    args = parser.parse_args()
    if not any(getattr(args, name.replace('-', '_')) for name in arguments):
        parser.error('Provide at least one local run directory')
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(args.output/'.matplotlib-cache'))
    figures = Figures(args.output, args.allow_partial)
    write_json(args.output/'chapter5_figure_manifest.json', dict(status='in_progress'))
    for condition, suffix, attr in [('nocn', 'without_cn', 'cyber_q_nocn'), ('withcn', 'with_cn', 'cyber_q_cn')]:
        folder = getattr(args, attr)
        if folder is None: continue
        config = figures.validate_run(folder, 'cyber_qlearning', f'cyber_qlearning_{suffix}.yaml')
        evaluations, trainings, surfaces = [], [], []
        for method in METHODS:
            evaluations.append(figures.read(folder/method, 'evaluation'))
            trainings.append(figures.read(folder/method, 'training'))
            surfaces.append(figures.read(folder/method, 'simplex'))
            target = next(item['total_iters'] for item in config['solvers'] if item['name']==method)
            if not args.allow_partial and any(record[1]['index'] != target for record in [evaluations[-1], trainings[-1], surfaces[-1]]):
                raise ValueError(f'{method}: final exports are incomplete; finish the configured run')
        figures.trajectories(evaluations, METHODS, 'cyber_q_eval_'+condition)
        figures.training(trainings, METHODS, 'cyber_q_train_'+condition+'_extra_display_smoothing')
        if condition=='withcn': figures.simplex(surfaces, METHODS, 'cyber_q_simplex_withcn')
    for prefix, kind, config_prefix in [('cyber_ddpg', 'cyber_ddpg', 'cyber_ddpg'), ('planning', 'distribution_planning', 'distribution_planning')]:
        evaluations, trainings, surfaces, labels = [], [], [], []
        for suffix, attr, label in [('without_cn', prefix+'_nocn', 'No CN'), ('with_cn', prefix+'_cn', 'CN')]:
            folder = getattr(args, attr)
            if folder is None: continue
            config = figures.validate_run(folder, kind, config_prefix+'_'+suffix+'.yaml')
            evaluation = figures.read(folder, 'evaluation' if kind=='cyber_ddpg' else 'planning')
            training = figures.read(folder, 'training')
            target = config['training']['episodes']
            records = [evaluation, training]
            if kind=='cyber_ddpg':
                surface = figures.read(folder, 'simplex'); surfaces.append(surface); records.append(surface)
            if not args.allow_partial and any(record[1]['index'] != target for record in records):
                raise ValueError(f'{folder}: final exports are incomplete; finish the configured run')
            evaluations.append(evaluation); trainings.append(training); labels.append(label)
            if kind=='distribution_planning':
                figures.planning(evaluation, 'distribution_eval_'+('nocn' if label=='No CN' else 'withcn'))
        if not labels: continue
        if kind=='cyber_ddpg':
            figures.trajectories(evaluations, labels, 'cyber_ddpg_eval')
            figures.simplex(surfaces, labels, 'cyber_ddpg_simplex', ddpg=True)
        figures.training(trainings, labels, 'cyber_ddpg_train' if kind=='cyber_ddpg' else 'distribution_ddpg_train',
                         planning=kind=='distribution_planning')
    unexpected = [name for name in ['torch', 'gymnasium'] if name in __import__('sys').modules]
    if unexpected: raise RuntimeError('Plotting unexpectedly imported '+str(unexpected))
    write_json(args.output/'chapter5_figure_manifest.json', dict(
        status='smoke_or_custom_not_book' if args.allow_partial else 'complete_requested_figures',
        figures=figures.records, sources=figures.sources, training_calls=0,
        policy_inference_calls=0, simulator_calls=0, requires_local_data=True))


if __name__ == '__main__':
    main()
