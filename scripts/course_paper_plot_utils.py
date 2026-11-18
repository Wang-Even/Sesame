"""Presentation utilities for recorded scalar results (no model computation)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/course_draft_20261115'
OUT = SOURCE / 'paper_style'
HALF_LIFE_STEPS = 131072
BLUE, ORANGE, GRAY = '#1054ad', '#c65520', '#757575'
COLORS = {'ppo': ORANGE, 'ppo_extended': ORANGE, 'supermimic': BLUE,
          'supermimic_deepseek': '#00866b', 'supermimic_codex': '#a3468c',
          'dqn': '#8a7367', 'sac': BLUE,
          'baseline': ORANGE, 'smooth': BLUE, 'fast': BLUE}


def set_report_source(source):
    """Select the report whose archived scalars and derived figures are used."""
    global SOURCE, OUT
    SOURCE = Path(source).resolve()
    OUT = SOURCE / 'paper_style'
    if not (SOURCE / 'report.md').is_file():
        raise FileNotFoundError(SOURCE / 'report.md')
    return SOURCE, OUT


def configure():
    plt.rcParams.update({
        'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 13,
        'axes.labelsize': 14, 'axes.titlesize': 17,
        'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11.5,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': .7, 'axes.edgecolor': '#707070',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'path',
        'savefig.dpi': 320, 'figure.facecolor': 'white',
        'axes.facecolor': 'white', 'legend.frameon': False,
    })


def panel(ax, letter, title, xlabel='', ylabel=''):
    ax.set_title(r'$\mathbf{(' + letter + r')}$ ' + title, loc='left', pad=12)
    ax.set_xlabel(xlabel, labelpad=7)
    ax.set_ylabel(ylabel, labelpad=7)
    ax.tick_params(which='major', length=5, width=.7, color='#777777')
    ax.tick_params(which='minor', length=2.8, width=.6, color='#999999')
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(False)
    ax.margins(x=.025)


def ema(steps, values):
    """Causal EMA with a fixed half-life in REAL transitions, not log rows."""
    steps, values = np.asarray(steps, float), np.asarray(values, float)
    assert len(steps) == len(values) and len(values) > 0
    assert np.all(np.isfinite(values)) and np.all(np.diff(steps) > 0)
    result = values.copy()
    for i in range(1, len(values)):
        alpha = -np.expm1(-np.log(2) * (steps[i] - steps[i-1]) / HALF_LIFE_STEPS)
        result[i] = (1 - alpha) * result[i-1] + alpha * values[i]
    return result


def draw_curve(ax, steps, x, y, color, label=None, *,
               gap_detection=False, linewidth=1.8, alpha=1., start_x=None):
    """Connect EMA values with one continuous solid line, without raw overlays.

    The original sample coordinates and smoothing rule are unchanged. Long gaps
    remain recorded in provenance, but no longer change the visible line style.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    smooth = ema(steps, y)
    assert np.all(np.diff(x) > 0)
    keep = np.ones(len(x), dtype=bool) if start_x is None else x >= start_x
    indices = np.flatnonzero(keep)
    assert len(indices) >= 2
    shown_x = x[keep]
    gaps = []
    if gap_detection and len(shown_x) > 2:
        gaps = indices[np.flatnonzero(np.diff(shown_x) > 3 * np.median(np.diff(shown_x))) + 1].tolist()
    ax.plot(shown_x, smooth[keep], color=color, alpha=alpha, lw=linewidth, ls='-',
            marker=None, label=label, zorder=3,
            solid_capstyle='round', solid_joinstyle='round')
    return smooth, gaps


def add_detail_view(ax, *, bounds, xlim, ylim, labels=None,
                    title='Late-stage detail (min)'):
    """Magnify existing line coordinates without shifting or resampling them."""
    selected = [line for line in ax.lines
                if labels is None or line.get_label() in labels]
    detail = ax.inset_axes(bounds)
    for line in selected:
        detail.plot(line.get_xdata(), line.get_ydata(),
                    color=line.get_color(), lw=1.35, ls='-', marker=None,
                    label=line.get_label(), solid_capstyle='round',
                    solid_joinstyle='round')
    detail.set(xlim=xlim, ylim=ylim)
    detail.set_title(title, loc='left', fontsize=10.5, pad=4)
    detail.tick_params(labelsize=10, length=3, width=.6, pad=2)
    detail.xaxis.set_major_locator(MaxNLocator(3))
    detail.yaxis.set_major_locator(MaxNLocator(3))
    detail.xaxis.set_minor_locator(AutoMinorLocator(2))
    detail.set_facecolor('white')
    detail._curve_detail = dict(kind='zoom', xlim=list(xlim), ylim=list(ylim),
                               methods=[line.get_label() for line in selected],
                               coordinates='Exact copies of displayed EMA line coordinates')
    return detail


def curve_records(group, method, seed, metric, steps, seconds, raw, smooth, *, startup_s=0.):
    return [dict(group=group, method=method, seed=seed, metric=metric,
                 real_steps=int(s), training_s=float(t), raw_value=float(y),
                 smoothed_value=float(z), half_life_real_steps=HALF_LIFE_STEPS,
                 startup_s=float(startup_s), plot_training_s=float(t)-float(startup_s),
                 included=bool(float(t) >= float(startup_s)))
            for s, t, y, z in zip(steps, seconds, raw, smooth)]
