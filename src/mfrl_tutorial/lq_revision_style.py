"""Accepted figure styling and display transform; no model or trainer imports."""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import numpy as np
import matplotlib.pyplot as plt

OUT=None
digest=lambda p:sha256(p.read_bytes()).hexdigest()
COLORS = {'Exact': '#222222', 'MKV': '#1f77b4', 'Pop_N2': '#9467bd',
          'Pop_N5': '#8c564b', 'Pop_N10': '#2ca02c', 'Pop_N20': '#17becf',
          'Pop_N50': '#e377c2', 'Pop_N100': '#ff7f0e', 'Optimum': '#666666'}

STYLES = {'Exact': '-', 'MKV': (0, (1.0, 1.5)), 'Pop_N2': '-',
          'Pop_N10': '-.', 'Pop_N100': '-'}

LABELS = {'Exact': 'Exact gradient', 'MKV': 'MKV', 'Optimum': 'Optimum',
          **{m: '$N=' + m[5:] + '$' for m in COLORS if m.startswith('Pop_N')}}

METHODS1 = ['Exact', 'MKV', 'Pop_N2', 'Pop_N10', 'Pop_N100']

X = np.arange(801)

def style(method):
    return dict(color=COLORS[method], ls=STYLES.get(method, '-'),
                lw=1.5 if method == 'MKV' else 1.15,
                zorder=12 if method == 'MKV' else 7 if method == 'Exact' else 5,
                dash_capstyle='round')

def decorate(ax, ylabel, log=False):
    ax.set(xlabel='Training updates', ylabel=ylabel, xlim=(0, 800),
           xticks=[0, 200, 400, 600, 800])
    if log:
        ax.set_yscale('log')
    ax.grid(alpha=.18, lw=.5)
    ax.tick_params(width=.6, length=2.5)

def visible_text(fig):
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += [ax.title, ax.xaxis.label, ax.yaxis.label,
                  ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()]
        for axis, lim in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if min(lim) <= tick.get_loc() <= max(lim):
                    texts += [tick.label1, tick.label2]
    for lg in fig.legends:
        texts += lg.get_texts()
    return [t for t in texts if t.get_visible() and t.get_text()]

def export(fig, name, caption, records):
    fig.canvas.draw()
    renderer, canvas = fig.canvas.get_renderer(), fig.bbox
    texts = visible_text(fig)
    outside = []
    for t in texts:
        b = t.get_window_extent(renderer)
        if b.width and b.height and (b.x0 < canvas.x0-.5 or b.y0 < canvas.y0-.5
                                    or b.x1 > canvas.x1+.5 or b.y1 > canvas.y1+.5):
            outside.append(t.get_text())
    assert not outside, (name, outside)
    public = (' '.join(t.get_text() for t in texts) + ' ' + caption).lower()
    assert not any(s in public for s in ['additional', 'previous', 'cohort', 'checkpoint', '202612'])
    record = dict(width_inches=fig.get_figwidth(), height_inches=fig.get_figheight(),
                  outside_text=outside, caption=caption, public_labels=[t.get_text() for t in texts])
    for ext in ['pdf', 'png']:
        p = OUT / (name + '.' + ext)
        kwargs = dict(metadata=dict(CreationDate=datetime(2026, 10, 26, tzinfo=timezone.utc),
                                    ModDate=datetime(2026, 10, 26, tzinfo=timezone.utc))) if ext == 'pdf' else {}
        fig.savefig(p, dpi=220, **kwargs)
        record[ext + '_sha256'] = digest(p)
    records[name] = record
    plt.close(fig)

def moving_average(a, window):
    """Centred arithmetic mean; shrink symmetrically near both endpoints.

    This preserves k=0 and k=800 exactly and never pads or extrapolates.
    The full window is used away from the first/last (window-1)/2 updates.
    """
    a = np.asarray(a)
    if window == 1:
        return a.copy()
    assert a.ndim == 2 and window % 2 == 1
    k = np.arange(a.shape[1])
    radius = np.minimum(np.minimum(k, a.shape[1]-1-k), window//2)
    first, last = k-radius, k+radius+1
    sums = np.pad(np.cumsum(a, axis=1), ((0,0),(1,0)))
    result = (sums[:,last]-sums[:,first])/(last-first)
    # Explicit endpoint assignments avoid subtraction roundoff at the last value.
    result[:,0], result[:,-1] = a[:,0], a[:,-1]
    for index in sorted({0,1,10,25,200,400,799,800} & set(range(a.shape[1]))):
        np.testing.assert_allclose(result[:,index], a[:,first[index]:last[index]].mean(axis=1),
                                   rtol=1e-10, atol=2e-13)
    return result

def thin_style(method):
    result=style(method)
    result["lw"]=1.05 if method in ["MKV","Exact"] else .85
    return result
