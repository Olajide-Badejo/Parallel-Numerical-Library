#!/usr/bin/env python3
"""Regenerate every figure and table in the report from summary.csv.

Idempotent by construction: the only inputs are experiments/results/summary.csv
and the session manifest that names the commit those rows carry, and every
output is overwritten in full. Running it twice produces identical files, and
running it after a partial sweep produces figures for the rows that exist and a
stated gap for the rest. No number in the report is typed by hand.

Which rows are published is decided by the commit column and never by row order.
The summary accumulates generations, one per commit that measured it, and this
refuses to run when more than one is present rather than picking the one that
happens to be last in the file: that selector shipped release 1.0.0 and it chose
between two 425 row generations on the strength of the order they were appended
in. `measured_at` is what orders rows inside a generation.

Design notes on the figures, since the reasoning is not visible in the output:

  Form follows the data's job. Speedup against worker count is change over a
  continuous axis, so it is a line. Time per backend is a comparison of
  magnitudes across one categorical axis, so it is a bar in a single hue, not
  eight colours: the backend names are the identity channel and the length is
  the measurement. Efficiency against each device's own peak is two entities
  compared across sizes, so it is a grouped bar with two hues.

  At most three colour coded series appear in any one figure. The palette's
  first three slots are the ones that clear the separation floors with every
  pair visible at once, which is the case for a scatter or a chart a reader
  scans rather than follows. Line style and marker shape carry the same
  distinction again, so the figures survive greyscale printing and colour vision
  deficiency without depending on hue at all.

  No figure has two y axes. Where two quantities of different scale belong
  together they are either normalised to a common base or drawn as two panels.

Dispersion is not decoration here, it is ground rule 7. `seconds_min` and
`seconds_max` are recorded on every row, so every timing table carries a spread
column, every timing figure carries min to max whiskers, and no difference
between two configurations is printed as a number unless it is larger than the
spread of the rows it was computed from. Where it is not, the cell carries the
phrase instead. The knee, which is this project's most repeated finding, is
refit on a bootstrap of the repetitions behind each point and reported with an
interval rather than as a single worker count.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"
SUMMARY = RESULTS / "summary.csv"
# Sessions write manifest-<commit>-<timestamp>.json, with the dot of a .dirty
# stamp written as a dash. A single session_manifest.json was the previous
# arrangement and it could not say which session it described; the one this
# repository shipped described a later re run and not the sweep that was
# published from. See PROV-04 in the engineering log.
MANIFEST_PREFIX = "manifest-"
ARCHIVE = "archive"
FIGURES = ROOT / "report" / "figures"
TABLES = ROOT / "report" / "tables"
# Portable copies for the repository landing page. GitHub cannot display a PDF
# inline in Markdown, so every figure is also written as a PNG here.
ASSET_FIGURES = ROOT / "assets" / "figures"

# Two selected themes rather than one flipped automatically. The categorical
# slots are the first three of the reference palette in each mode, because those
# are the ones that clear the separation floors with every pair visible at once;
# a fourth would put yellow beside orange and fail that gate. The dark steps are
# the same hues re-stepped for a dark surface, not inverted.
THEMES = {
    "light": {
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "sequential": "#2a78d6",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "surface": "#fcfcfb",
    },
    "dark": {
        "series": ["#3987e5", "#d95926", "#199e70"],
        "sequential": "#3987e5",
        "ink": "#ffffff",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "surface": "#1a1a19",
    },
}

# Active theme values. Rebound by set_theme so the figure functions can read
# them by name without every one of them taking a palette argument.
SERIES = THEMES["light"]["series"]
SEQUENTIAL = THEMES["light"]["sequential"]
INK = THEMES["light"]["ink"]
INK_SECONDARY = THEMES["light"]["ink_secondary"]
MUTED = THEMES["light"]["muted"]
GRID = THEMES["light"]["grid"]
AXIS = THEMES["light"]["axis"]
SURFACE = THEMES["light"]["surface"]
THEME_NAME = "light"

# Secondary encoding, so identity never rests on hue alone and the figures stay
# readable in greyscale and under colour vision deficiency.
MARKERS = ["o", "s", "^"]
DASHES = [(None, None), (5, 2), (1.5, 1.5)]


def set_theme(name: str) -> None:
    """Activate a theme for every figure drawn after this call."""
    global SERIES, SEQUENTIAL, INK, INK_SECONDARY, MUTED, GRID, AXIS, SURFACE, THEME_NAME
    theme = THEMES[name]
    SERIES = theme["series"]
    SEQUENTIAL = theme["sequential"]
    INK = theme["ink"]
    INK_SECONDARY = theme["ink_secondary"]
    MUTED = theme["muted"]
    GRID = theme["grid"]
    AXIS = theme["axis"]
    SURFACE = theme["surface"]
    THEME_NAME = name

    matplotlib.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "normal",
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "legend.labelcolor": INK,
        "lines.linewidth": 1.6,
        "lines.markersize": 4.5,
        "figure.dpi": 150,
    })


set_theme("light")


def style_axes(ax: Any) -> None:
    """Recessive chrome: no box, hairline grid, muted ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(length=3, width=0.8)


def note(ax: Any, text: str) -> None:
    """A muted note under the axes, for a convention a figure has to state.

    Every timing figure carries whiskers and some of them carry whiskers on a
    derived quantity, so the rule that produced them travels with the image
    rather than living only in the caption of whichever document embeds it.
    """
    ax.annotate(textwrap.fill(text, 108), xy=(0.0, -0.24), xycoords="axes fraction",
                fontsize=7, color=MUTED, va="top", ha="left")


def save(fig: Any, name: str) -> None:
    """Write a figure once as PDF for LaTeX and once as PNG for the landing page.

    The PDF is vector and is what the report embeds, so it is written only for
    the light theme, which is what a printed page is. The PNG exists because
    GitHub cannot display a PDF inline in Markdown, and is written for both
    themes so the landing page can select the right one from the reader's
    colour scheme rather than showing a white card on a dark page.
    """
    stem = Path(name).stem
    if THEME_NAME == "light":
        FIGURES.mkdir(parents=True, exist_ok=True)
        pdf_path = FIGURES / f"{stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)

    ASSET_FIGURES.mkdir(parents=True, exist_ok=True)
    png_path = ASSET_FIGURES / f"{stem}-{THEME_NAME}.png"
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.06, dpi=200)
    plt.close(fig)
    print(f"  figure  {png_path.relative_to(ROOT)}")


def latex_escape(text: str) -> str:
    """Escape a value for LaTeX, and never emit a character the dash rule bans."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = "".join(replacements.get(char, char) for char in str(text))
    # A literal double hyphen would typeset as an en dash, which ground rule 1
    # forbids. This can only arise from data, never from the prose here.
    return out.replace("--", r"-{}-")


def write_table(name: str, header: list[str], rows: list[list[Any]], caption: str,
                label: str, column_spec: str | None = None) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    spec = column_spec or ("l" + "r" * (len(header) - 1))
    lines = [
        "% Generated by scripts/gen_report_assets.py. Do not edit by hand.",
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        " & ".join(latex_escape(h) for h in header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path = TABLES / name
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  table   {path.relative_to(ROOT)}")


# The text a cell carries where the counted bandwidth cannot be derived.
PREDATES = "predates the column"


def counted_gib_per_second(row: Any) -> float | None:
    """Achieved bandwidth under the read for ownership traffic model.

    dram_bytes_per_unknown_per_sweep x passes x unknowns x iterations, over the
    median time. It differs from the `gib_per_second` column in two ways at
    once, and both are deliberate. The byte count is the one with read for
    ownership charged on the array a pass writes without reading first, which
    is the open question of Section 4.2. The work unit is `passes`, streams
    over memory, where `gib_per_second` uses `sweeps`, updates per unknown;
    the two agree for every method except the red black pair, which does one
    sweep of work in two passes.

    Returns None, never a guess, when either column is empty. A row written
    before phase A2 has no `passes` and a row written before phase A3a has no
    byte count, and assuming one pass for such a row would halve the figure for
    exactly the two methods that carry the device comparison.
    """
    values = {}
    for column in ("dram_bytes_per_unknown_per_sweep", "passes", "unknowns",
                   "iterations", "seconds_median"):
        values[column] = pd.to_numeric(row.get(column), errors="coerce")
    if any(pd.isna(value) for value in values.values()):
        return None
    if values["seconds_median"] <= 0:
        return None
    moved = (values["dram_bytes_per_unknown_per_sweep"] * values["passes"] *
             values["unknowns"] * values["iterations"])
    return float(moved / values["seconds_median"] / (1024.0 ** 3))


def fmt(value: Any, places: int = 3) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "pending"
    if isinstance(value, float):
        if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e6):
            return f"{value:.{places}e}"
        return f"{value:.{places}f}"
    return str(value)


# ---------------------------------------------------------------------------
# Dispersion, and the two rules that rest on it
# ---------------------------------------------------------------------------

# What a cell says when ground rule 7 fires: the difference between two
# configurations is smaller than the run to run spread of the rows it was
# computed from, so the data cannot tell them apart and a percentage would be a
# claim the measurement does not support.
NOT_SEPARABLE = "not separable at this precision"

# How a derived quantity gets its bounds, stated once here and repeated in every
# caption that carries one. A speedup, an efficiency or a ratio is a function of
# measured times, and each one is evaluated twice more: once at the corner of the
# inputs' min to max intervals that makes it largest, and once at the corner that
# makes it smallest. For a speedup that pairs the baseline's slowest run with
# this point's fastest, and then the opposite. It is the widest interval the
# inputs allow rather than the narrowest, which is the only direction that cannot
# understate the uncertainty of a derived number.
DERIVED_BOUNDS = (
    "Whiskers on a derived quantity are taken at the corners of its inputs' min "
    "to max intervals, the pairing that makes it largest and the pairing that "
    "makes it smallest, which is the widest interval those inputs allow rather "
    "than the narrowest."
)

# The bootstrap behind the knee interval. Both are recorded in the table so the
# interval can be reproduced exactly.
KNEE_BOOTSTRAP_SAMPLES = 2000
KNEE_BOOTSTRAP_SEED = 20260906

# Which resampling a knee interval came from. Never mixed within one backend and
# always named in the table, because an interval from the recorded repetitions
# and an interval from an assumed shape are not the same kind of statement.
MEASURED_REPS = "measured repetitions"
TRIANGULAR = "triangular fallback"


def spread(row: Any) -> float | None:
    """Run to run spread of one measured row, (max - min) / median.

    One definition, used by every table, every figure and both rules below. It
    is a range rather than a standard deviation because the harness records the
    extremes of the repetitions and not their variance, and because at fifteen
    repetitions a range is the honest summary of what was seen.

    Returns None when a column is missing or the median is not positive, so a
    caller prints "pending" rather than a number derived from nothing.
    """
    values = {}
    for column in ("seconds_median", "seconds_min", "seconds_max"):
        values[column] = pd.to_numeric(row.get(column), errors="coerce")
    if any(pd.isna(value) for value in values.values()):
        return None
    if values["seconds_median"] <= 0:
        return None
    return float((values["seconds_max"] - values["seconds_min"]) / values["seconds_median"])


def fmt_spread(row: Any) -> str:
    """The spread of a row as a percentage, for a table cell."""
    value = spread(row)
    return "pending" if value is None else f"{value * 100:.1f}"


def seconds_bounds(row: Any) -> tuple[float, float, float] | None:
    """The (min, median, max) seconds of one row, or None if any is missing."""
    values = []
    for column in ("seconds_min", "seconds_median", "seconds_max"):
        value = pd.to_numeric(row.get(column), errors="coerce")
        if pd.isna(value) or value <= 0:
            return None
        values.append(float(value))
    return values[0], values[1], values[2]


def speedup_bounds(base: Any, row: Any) -> tuple[float, float] | None:
    """Conservative bounds on the speedup of `row` against baseline `base`.

    The corners of the two min to max intervals: the largest speedup this data
    allows pairs the baseline's slowest run with this point's fastest, and the
    smallest pairs the baseline's fastest with this point's slowest.
    """
    baseline = seconds_bounds(base)
    point = seconds_bounds(row)
    if baseline is None or point is None:
        return None
    return baseline[0] / point[2], baseline[2] / point[0]


def efficiency_whisker(row: Any, value: float) -> tuple[float, float]:
    """How far an achieved bandwidth figure moves with the time it was divided by.

    Achieved bandwidth is a fixed byte count over the measured seconds, so the
    fastest repetition gives the largest figure and the slowest the smallest.
    Returned as the (below, above) lengths matplotlib wants rather than as the
    bounds themselves. A row with no recorded extremes gets no whisker.
    """
    bounds = seconds_bounds(row)
    if bounds is None:
        return 0.0, 0.0
    low, middle, high = bounds
    return max(0.0, value - value * middle / high), max(0.0, value * middle / low - value)


def separable_effect(a: Any, b: Any) -> float | str:
    """The relative effect of configuration `a` against baseline `b`, or the phrase.

    Ground rule 7. The effect is `|a - b| / b` on the medians, and it is
    reported only when it exceeds the larger of the two rows' spreads. Otherwise
    the two configurations cannot be told apart by this data and the caller
    prints NOT_SEPARABLE where a percentage would have gone.

    The value returned is signed so that a caller can say which way the effect
    runs; the comparison that decides whether to return it at all is on the
    magnitude, which is what the rule is written over. A row with no recorded
    minimum or maximum has no spread to compare against, and the phrase is
    returned rather than a number, because an effect whose noise floor is
    unknown is exactly the case the rule exists for.
    """
    a_median = pd.to_numeric(a.get("seconds_median"), errors="coerce")
    b_median = pd.to_numeric(b.get("seconds_median"), errors="coerce")
    if pd.isna(a_median) or pd.isna(b_median) or b_median <= 0:
        return NOT_SEPARABLE
    spreads = [value for value in (spread(a), spread(b)) if value is not None]
    if not spreads:
        return NOT_SEPARABLE
    effect = float(a_median - b_median) / float(b_median)
    if abs(effect) <= max(spreads):
        return NOT_SEPARABLE
    return effect


def fmt_effect(effect: float | str) -> str:
    """A signed percentage, or the phrase, for a table cell."""
    if isinstance(effect, str):
        return effect
    return f"{effect * 100:+.1f}"


def wider_spread(a: Any, b: Any) -> float | None:
    """The larger of two rows' spreads, which is what the rule is tested against."""
    spreads = [value for value in (spread(a), spread(b)) if value is not None]
    return max(spreads) if spreads else None


def effect_sentence(subject: str, baseline: str, effect: float | str,
                    a: Any, b: Any) -> str:
    """One sentence stating a comparison, or stating that it cannot be made."""
    widest = wider_spread(a, b)
    noise = "unknown" if widest is None else f"{widest * 100:.1f} percent"
    if isinstance(effect, str):
        a_median = pd.to_numeric(a.get("seconds_median"), errors="coerce")
        b_median = pd.to_numeric(b.get("seconds_median"), errors="coerce")
        gap = ("unknown" if pd.isna(a_median) or pd.isna(b_median) or b_median <= 0
               else f"{abs(float(a_median) / float(b_median) - 1.0) * 100:.1f} percent")
        return (f"{subject} against {baseline} is {NOT_SEPARABLE}: the medians differ by "
                f"{gap} and the wider of the two spreads is {noise}.")
    direction = "slower" if effect > 0 else "faster"
    return (f"{subject} is {abs(effect) * 100:.1f} percent {direction} than {baseline}, "
            f"against a wider spread of {noise}.")


def write_fragment(name: str, intro: str, sentences: list[str]) -> None:
    """A verdict fragment: one sentence per comparison, for results.tex to input.

    Not a float and not a table. It sits beside the table it belongs to so that
    the prose quotes what the generator computed from the data rather than a
    number somebody typed while reading the table.
    """
    TABLES.mkdir(parents=True, exist_ok=True)
    body = sentences or ["No comparison in this block had both configurations present."]
    lines = [
        "% Generated by scripts/gen_report_assets.py. Do not edit by hand.",
        intro,
        r"\begin{itemize}",
    ]
    lines += [rf"\item {latex_escape(sentence[:1].upper() + sentence[1:])}"
              for sentence in body]
    lines += [r"\end{itemize}", ""]
    path = TABLES / name
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  verdict {path.relative_to(ROOT)}")


def repetitions_text(block: pd.DataFrame) -> str:
    """How many repetitions the rows of a block carry, for a caption.

    Read from the data rather than written down, because the count changes with
    the sweep matrix and a caption that says "median of five" outlives the five.
    """
    if "reps" not in block.columns:
        return "median of the recorded repetitions"
    values = sorted({int(v) for v in pd.to_numeric(block["reps"], errors="coerce").dropna()})
    if len(values) == 1:
        return f"median of {values[0]} repetitions"
    return "median of the repetitions recorded in the reps column"


def repetition_seconds(row: Any) -> list[float] | None:
    """Every timed repetition of one row, from `seconds_reps`, or None.

    The column arrived in phase A1.5 and is empty on every row measured before
    it, which is every row of the 1.0.0 generation. A caller that gets None must
    fall back to an assumed shape and must say in the caption that it did.
    """
    raw = row.get("seconds_reps")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return None
    values: list[float] = []
    for part in text.split(";"):
        piece = part.strip()
        if not piece:
            continue
        try:
            values.append(float(piece))
        except ValueError:
            return None
    return values or None


def resampled_median(row: Any, rng: random.Random, method: str) -> float | None:
    """One bootstrap draw of a row's median time.

    Under MEASURED_REPS the recorded repetitions are resampled with replacement
    and the median of the draw is returned, which is the ordinary bootstrap of
    the statistic the report quotes. Under TRIANGULAR there are no repetitions
    to resample, so the same number of draws is taken from a triangular
    distribution on (min, median, max): the crudest shape that respects the
    three numbers the row does carry, and honest only because the table says it
    was used.
    """
    if method == MEASURED_REPS:
        reps = repetition_seconds(row)
        if not reps:
            return None
        draw = [reps[rng.randrange(len(reps))] for _ in reps]
        return statistics.median(draw)
    bounds = seconds_bounds(row)
    if bounds is None:
        return None
    low, middle, high = bounds
    count = pd.to_numeric(row.get("reps"), errors="coerce")
    count = 5 if pd.isna(count) or count < 1 else int(count)
    draw = [rng.triangular(low, high, middle) for _ in range(count)]
    return statistics.median(draw)


def resampling_method(series: pd.DataFrame) -> str:
    """Which resampling a whole series must use, so the two are never mixed.

    A series takes the measured repetitions only if every one of its rows has
    them. One row without is enough to put the whole backend on the fallback,
    because an interval built from measured repetitions at some worker counts
    and from an assumed shape at others describes neither.
    """
    for _, row in series.iterrows():
        if not repetition_seconds(row):
            return TRIANGULAR
    return MEASURED_REPS


def percentile(values: list[float], fraction: float) -> float | None:
    """An order statistic, without interpolating between the values observed.

    A knee is a worker count, so an interpolated percentile would report a knee
    at 4.7 workers, which is not a position the fit can return. The nearest rank
    is an outcome the bootstrap actually produced.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = round(fraction * (len(ordered) - 1))
    return ordered[max(0, min(index, len(ordered) - 1))]


def bootstrap_knee(series: pd.DataFrame, samples: int = KNEE_BOOTSTRAP_SAMPLES,
                   seed: int = KNEE_BOOTSTRAP_SEED) -> dict[str, Any]:
    """The knee of one backend's scaling curve, with a percentile interval.

    The two segment least squares fit of find_knee is unchanged and still
    supplies the point estimate, computed from the medians the report quotes.
    What is added around it is a bootstrap: each draw resamples every worker
    count's repetitions, recomputes that point's median, rebuilds the speedup
    curve against the resampled one worker baseline, and refits. The 2.5 and
    97.5 percentiles of the refitted knee positions are the interval.

    The interval is over the sampling of the repetitions and nothing else. It
    does not cover a machine that was busy during one worker count and idle
    during another, which is the error this measurement is most exposed to.
    """
    ordered = series.sort_values("workers")
    rows = [ordered.iloc[i] for i in range(len(ordered))]
    if len(rows) < 5:
        return {}
    workers = [float(row["workers"]) for row in rows]
    baseline_index = next((i for i, w in enumerate(workers) if w == 1), None)
    if baseline_index is None:
        return {}
    medians = [pd.to_numeric(row.get("seconds_median"), errors="coerce") for row in rows]
    if any(pd.isna(value) or value <= 0 for value in medians):
        return {}
    base = float(medians[baseline_index])
    fitted = find_knee(workers, [base / float(value) for value in medians])
    if not fitted:
        return {}

    method = resampling_method(ordered)
    rng = random.Random(seed)
    positions: list[float] = []
    for _ in range(samples):
        drawn = [resampled_median(row, rng, method) for row in rows]
        if any(value is None or value <= 0 for value in drawn):
            positions = []
            break
        drawn_base = drawn[baseline_index]
        knee = find_knee(workers, [drawn_base / value for value in drawn])
        if knee:
            positions.append(float(knee["workers"]))

    return {
        **fitted,
        "low": percentile(positions, 0.025),
        "high": percentile(positions, 0.975),
        "method": method,
        "samples": len(positions),
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_scaling(data: pd.DataFrame) -> dict[str, Any]:
    """Speedup against worker count, and the knee where fast cores run out."""
    block = data[(data["label"] == "scaling") & (data["solver"] == "jacobi")]
    if block.empty:
        return {}

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    backends = [b for b in ("openmp", "pthreads", "jthread") if b in set(block["backend"])]
    knees: dict[str, Any] = {}
    measured_peak = 1.0

    for index, backend in enumerate(backends):
        series = block[block["backend"] == backend].sort_values("workers")
        if series.empty:
            continue
        baseline = series[series["workers"] == 1]
        if baseline.empty:
            continue
        base_row = baseline.iloc[0]
        base = float(base_row["seconds_median"])
        speedup = [base / float(row["seconds_median"]) for _, row in series.iterrows()]
        # Whiskers on a speedup, which is derived, so its bounds come from the
        # corners of the two inputs' min to max intervals. See DERIVED_BOUNDS.
        below, above = [], []
        for value, (_, row) in zip(speedup, series.iterrows(), strict=True):
            bounds = speedup_bounds(base_row, row)
            below.append(0.0 if bounds is None else max(0.0, value - bounds[0]))
            above.append(0.0 if bounds is None else max(0.0, bounds[1] - value))
        tops = [value + tail for value, tail in zip(speedup, above, strict=True)]
        if tops:
            measured_peak = max(measured_peak, *tops)
        ax.errorbar(series["workers"], speedup, yerr=[below, above],
                    color=SERIES[index], marker=MARKERS[index],
                    dashes=DASHES[index] if DASHES[index][0] else (None, None),
                    label=backend, markeredgecolor=SURFACE, markeredgewidth=0.6,
                    ecolor=SERIES[index], elinewidth=0.8, capsize=2.0, capthick=0.8)
        knees[backend] = bootstrap_knee(series)

    # Ideal scaling is reference chrome, not a series, so it is muted and
    # unmarked. It is deliberately allowed to leave the top of the axes: this
    # workload saturates at a speedup below two, and scaling the y axis to fit a
    # line that reaches 28 would compress every measured curve into an
    # unreadable band at the bottom. The point of the figure is the shape of the
    # measurements, not the size of the gap.
    top = int(block["workers"].max())
    ax.plot([1, top], [1, top], color=MUTED, linewidth=0.9, dashes=(2, 3), zorder=0)

    ceiling = max(2.0, float(measured_peak) * 1.35)
    ax.set_ylim(0, ceiling)
    ax.set_xlim(0, top + 1)
    # Label the reference line where it actually leaves the plot.
    exit_x = min(ceiling, top)
    ax.annotate("ideal scaling", xy=(exit_x, ceiling), xytext=(4, -12),
                textcoords="offset points", color=MUTED, fontsize=8)

    ax.set_xlabel("workers")
    ax.set_ylabel("speedup against one worker")
    ax.set_title("Jacobi sweep scaling, 1023 by 1023 grid", color=INK, loc="left")
    ax.legend(loc="upper right")
    style_axes(ax)
    note(ax, "Whiskers span the minimum to the maximum of the recorded repetitions. "
             + DERIVED_BOUNDS)
    save(fig, "scaling_speedup.pdf")
    return knees


def find_knee(workers: list[float], speedup: list[float]) -> dict[str, Any]:
    """Locate the change of slope by the best two segment least squares split.

    The guest cannot label its processors as performance or efficiency cores
    (ENV-04), so the knee is read from the aggregate curve, which depends only
    on how many cores are engaged. The split minimising total residual is the
    worker count past which adding workers buys measurably less.
    """
    if len(workers) < 5:
        return {}
    best = None
    for cut in range(2, len(workers) - 2):
        left = fit_line(workers[:cut + 1], speedup[:cut + 1])
        right = fit_line(workers[cut:], speedup[cut:])
        total = left["residual"] + right["residual"]
        if best is None or total < best["residual"]:
            best = {
                "workers": workers[cut],
                "residual": total,
                "slope_before": left["slope"],
                "slope_after": right["slope"],
            }
    return best or {}


def fit_line(xs: list[float], ys: list[float]) -> dict[str, float]:
    n = len(xs)
    if n < 2:
        return {"slope": 0.0, "residual": 0.0}
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx if sxx else 0.0
    intercept = mean_y - slope * mean_x
    residual = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True))
    return {"slope": slope, "residual": residual}


def figure_backend_cost(data: pd.DataFrame) -> None:
    """Time per backend on identical work: one categorical axis, one hue."""
    block = data[(data["label"] == "backend_cost") & (data["solver"] == "jacobi")]
    if block.empty:
        return
    size = sorted(block["unknowns"].unique())[-1]
    block = block[block["unknowns"] == size].sort_values("seconds_median")
    if block.empty:
        return

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    names = block["backend"].tolist()
    times = block["seconds_median"].astype(float).tolist()
    positions = range(len(names))
    # Whiskers straight from the recorded extremes: this figure plots the measured
    # time itself, so nothing is derived and no corner rule is needed.
    below, above = [], []
    for _, row in block.iterrows():
        bounds = seconds_bounds(row)
        below.append(0.0 if bounds is None else bounds[1] - bounds[0])
        above.append(0.0 if bounds is None else bounds[2] - bounds[1])
    # Magnitude on one categorical axis: a single hue, length carries the value.
    ax.barh(list(positions), times, color=SEQUENTIAL, height=0.62,
            xerr=[below, above],
            error_kw={"ecolor": INK_SECONDARY, "elinewidth": 0.9, "capsize": 2.0,
                      "capthick": 0.9})
    ax.set_yticks(list(positions), names)
    ax.invert_yaxis()
    ax.set_xlabel(f"seconds, {repetitions_text(block)}")
    ax.set_title(f"200 Jacobi sweeps at {int(size):,} unknowns, 20 workers",
                 color=INK, loc="left")
    # Direct labels rather than a value on every gridline.
    for position, value, tail in zip(positions, times, above, strict=True):
        ax.annotate(f"{value:.3f}", xy=(value + tail, position), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=INK_SECONDARY)
    ax.set_xlim(0, max(t + a for t, a in zip(times, above, strict=True)) * 1.24)
    ax.grid(axis="y", visible=False)
    style_axes(ax)
    note(ax, "Whiskers span the minimum to the maximum of the recorded repetitions.")
    save(fig, "backend_cost.pdf")


def figure_device_efficiency(data: pd.DataFrame, bandwidth: dict[str, Any]) -> None:
    """Percent of each device's own measured triad bandwidth.

    This is the fair comparison of Section 8.3: a dimensionless ratio that says
    how well each device is being used, rather than which device is faster.
    """
    block = data[data["label"] == "device_comparison"]
    if block.empty:
        return
    host_peak = (bandwidth.get("host") or {}).get("gib_per_second")
    device_peak = (bandwidth.get("gpu") or {}).get("gib_per_second")
    if not host_peak or not device_peak:
        return

    # Compare methods at the largest size where both devices are genuinely
    # streaming, rather than one method across sizes. Below its cache a device
    # is not bandwidth bound and the ratio would exceed one, which would be an
    # artefact of the roofline assumption rather than a measurement; that leaves
    # a single usable size, and a bar chart with one category on its axis wastes
    # the comparison. Methods on the axis is the informative cut: it shows that
    # for two of them the devices are used equally well.
    streaming = [
        s for s in sorted(block["unknowns"].dropna().unique())
        if working_set_mib(float(s)) > STREAMING_MARGIN * max(CACHE_MIB.values())
    ]
    if not streaming:
        return
    size = streaming[-1]

    methods: list[str] = []
    cpu_efficiency: list[float] = []
    gpu_efficiency: list[float] = []
    # The same two bars under the read for ownership model, drawn as an outline
    # over the solid bar rather than as two more hues: the device is the
    # identity channel and the traffic model is a second encoding on top of it,
    # so the figure still carries two colours and survives greyscale. A value of
    # zero means the row predates the columns the counted model needs, and it is
    # not drawn.
    cpu_counted: list[float] = []
    gpu_counted: list[float] = []
    # Whiskers on an efficiency, which is derived. Achieved bandwidth is a fixed
    # byte count over the measured time, so it moves inversely with the time and
    # the corners are the fastest and the slowest recorded repetition. See
    # DERIVED_BOUNDS.
    cpu_bars: list[tuple[float, float]] = []
    gpu_bars: list[tuple[float, float]] = []
    for solver in dict.fromkeys(block["solver"]):
        cpu = block[(block["backend"] == "openmp") & (block["solver"] == solver) &
                    (block["unknowns"] == size)]
        gpu = block[(block["backend"] == "cuda") & (block["solver"] == solver) &
                    (block["unknowns"] == size)]
        if cpu.empty or gpu.empty:
            continue
        methods.append(str(solver).replace("_", " "))
        cpu_efficiency.append(float(cpu["gib_per_second"].iloc[0]) / host_peak * 100)
        gpu_efficiency.append(float(gpu["gib_per_second"].iloc[0]) / device_peak * 100)
        cpu_bars.append(efficiency_whisker(cpu.iloc[0], cpu_efficiency[-1]))
        gpu_bars.append(efficiency_whisker(gpu.iloc[0], gpu_efficiency[-1]))
        host_counted = counted_gib_per_second(cpu.iloc[0])
        gpu_counted_value = counted_gib_per_second(gpu.iloc[0])
        cpu_counted.append(0.0 if host_counted is None else host_counted / host_peak * 100)
        gpu_counted.append(
            0.0 if gpu_counted_value is None else gpu_counted_value / device_peak * 100)
    if not methods:
        return

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    width = 0.36
    positions = list(range(len(methods)))
    left = [p - width / 2 - 0.012 for p in positions]
    right = [p + width / 2 + 0.012 for p in positions]
    error_style = {"ecolor": INK_SECONDARY, "elinewidth": 0.9, "capsize": 2.0,
                   "capthick": 0.9}
    ax.bar(left, cpu_efficiency, width, color=SERIES[0], label="host, 20 threads",
           yerr=[[b for b, _ in cpu_bars], [a for _, a in cpu_bars]],
           error_kw=error_style)
    ax.bar(right, gpu_efficiency, width, color=SERIES[1], label="RTX 5070",
           yerr=[[b for b, _ in gpu_bars], [a for _, a in gpu_bars]],
           error_kw=error_style)

    counted_drawn = any(cpu_counted) or any(gpu_counted)
    if counted_drawn:
        for xs, values in ((left, cpu_counted), (right, gpu_counted)):
            for x, value in zip(xs, values, strict=True):
                if value <= 0.0:
                    continue
                ax.bar(x, value, width, facecolor="none", edgecolor=INK,
                       linewidth=1.0, linestyle="--", zorder=3)
        ax.bar(positions[0], 0.0, width, facecolor="none", edgecolor=INK,
               linewidth=1.0, linestyle="--", label="read for ownership counted")

    for xs, values, bars in ((left, cpu_efficiency, cpu_bars),
                             (right, gpu_efficiency, gpu_bars)):
        for x, value, (_, tail) in zip(xs, values, bars, strict=True):
            ax.annotate(f"{value:.0f}%", xy=(x, value + tail), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8,
                        color=INK_SECONDARY)

    ax.set_xticks(positions, methods)
    ax.set_ylabel("percent of own measured peak")
    subtitle = "declared byte model" if not counted_drawn else "both byte models"
    ax.set_title(f"Efficiency against each device's own bandwidth, "
                 f"{int(size):,} unknowns, {subtitle}", color=INK, loc="left")
    tops = ([value + tail for value, (_, tail) in zip(cpu_efficiency, cpu_bars, strict=True)] +
            [value + tail for value, (_, tail) in zip(gpu_efficiency, gpu_bars, strict=True)] +
            cpu_counted + gpu_counted)
    ax.set_ylim(0, max(tops) * 1.28)
    ax.legend(loc="upper right", ncols=2)
    ax.grid(axis="x", visible=False)
    style_axes(ax)
    note(ax, "Whiskers span the minimum to the maximum of the recorded repetitions. "
             + DERIVED_BOUNDS + " The outlined bars are the read for ownership byte "
             "count over the same times and carry the same relative spread.")
    save(fig, "device_efficiency.pdf")
    if not counted_drawn:
        print("  note    device_efficiency.pdf carries the declared model only: "
              "these rows predate the passes and dram bytes columns")


def figure_iteration_counts(data: pd.DataFrame) -> None:
    """Iterations to tolerance per method: hardware free, and reported first."""
    block = data[data["label"] == "convergence_counts"]
    if block.empty:
        return
    size = sorted(block["unknowns"].unique())[-1]
    block = block[(block["unknowns"] == size) & (block["converged"] == 1)]
    block = block.sort_values("iterations", ascending=False)
    if block.empty:
        return

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    names = block["solver"].tolist()
    counts = block["iterations"].astype(float).tolist()
    positions = list(range(len(names)))
    ax.barh(positions, counts, color=SEQUENTIAL, height=0.66)
    ax.set_yticks(positions, names)
    ax.set_xscale("log")
    ax.set_xlabel("iterations to relative residual 1e-8, log scale")
    side = round(float(size) ** 0.5)
    ax.set_title(f"Iterations to tolerance, {side} by {side} grid", color=INK, loc="left")
    for position, value in zip(positions, counts, strict=True):
        ax.annotate(f"{int(value):,}", xy=(value, position), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=INK_SECONDARY)
    ax.grid(axis="y", visible=False)
    style_axes(ax)
    save(fig, "iteration_counts.pdf")


def figure_convergence_growth(data: pd.DataFrame) -> None:
    """Iteration count against grid size: the change in growth order, measured."""
    block = data[data["label"].isin(["convergence_counts", "convergence_counts_large"])]
    block = block[block["converged"] == 1]
    if block.empty:
        return

    chosen = [s for s in ("jacobi", "gauss_seidel_f", "cg") if s in set(block["solver"])]
    if not chosen:
        return

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for index, solver in enumerate(chosen):
        series = block[block["solver"] == solver].sort_values("unknowns")
        if series.empty:
            continue
        sides = [round(float(u) ** 0.5) for u in series["unknowns"]]
        ax.plot(sides, series["iterations"].astype(float),
                color=SERIES[index], marker=MARKERS[index],
                dashes=DASHES[index] if DASHES[index][0] else (None, None),
                label=solver, markeredgecolor=SURFACE, markeredgewidth=0.6)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("grid points per side")
    ax.set_ylabel("iterations to 1e-8, log scale")
    ax.set_title("Iteration count growth with grid size", color=INK, loc="left")
    ax.legend(loc="upper left")
    style_axes(ax)
    save(fig, "convergence_growth.pdf")


def figure_mpi_communication(data: pd.DataFrame) -> None:
    """How MPI scales, and where conjugate gradient loses to Jacobi."""
    block = data[(data["label"] == "mpi_scaling") & (data["backend"] == "mpi")]
    if block.empty:
        return
    chosen = [s for s in ("jacobi", "cg", "gauss_seidel_f") if s in set(block["solver"])]
    if not chosen:
        return

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for index, solver in enumerate(chosen[:3]):
        series = block[block["solver"] == solver].sort_values("workers")
        baseline = series[series["workers"] == 1]
        if baseline.empty:
            continue
        base_row = baseline.iloc[0]
        base = float(base_row["seconds_median"])
        speedup = [base / float(row["seconds_median"]) for _, row in series.iterrows()]
        below, above = [], []
        for value, (_, row) in zip(speedup, series.iterrows(), strict=True):
            bounds = speedup_bounds(base_row, row)
            below.append(0.0 if bounds is None else max(0.0, value - bounds[0]))
            above.append(0.0 if bounds is None else max(0.0, bounds[1] - value))
        ax.errorbar(series["workers"], speedup, yerr=[below, above],
                    color=SERIES[index], marker=MARKERS[index],
                    dashes=DASHES[index] if DASHES[index][0] else (None, None),
                    label=solver, markeredgecolor=SURFACE, markeredgewidth=0.6,
                    ecolor=SERIES[index], elinewidth=0.8, capsize=2.0, capthick=0.8)

    top = int(block["workers"].max())
    ax.plot([1, top], [1, top], color=MUTED, linewidth=0.9, dashes=(2, 3), zorder=0)
    ax.annotate("ideal", xy=(top * 0.62, top * 0.66), color=MUTED, fontsize=8)
    ax.set_xlabel("ranks")
    ax.set_ylabel("speedup against one rank")
    ax.set_title("MPI scaling by method, 1023 by 1023 grid", color=INK, loc="left")
    ax.legend(loc="upper left")
    style_axes(ax)
    note(ax, "Whiskers span the minimum to the maximum of the recorded repetitions. "
             + DERIVED_BOUNDS)
    save(fig, "mpi_scaling.pdf")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_convergence(data: pd.DataFrame) -> None:
    block = data[data["label"] == "convergence_counts"]
    if block.empty:
        return
    sizes = sorted(block["unknowns"].unique())
    solvers = list(dict.fromkeys(block["solver"]))
    header = ["method"] + [f"{round(float(s) ** 0.5)} by {round(float(s) ** 0.5)}"
                           for s in sizes]
    rows: list[list[Any]] = []
    for solver in solvers:
        row: list[Any] = [solver]
        for size in sizes:
            match = block[(block["solver"] == solver) & (block["unknowns"] == size)]
            if match.empty:
                row.append("pending")
            elif int(match["converged"].iloc[0]) == 0:
                row.append("cap")
            else:
                row.append(f"{int(match['iterations'].iloc[0]):,}")
        rows.append(row)
    write_table(
        "convergence.tex", header, rows,
        "Iterations to a relative residual of 1e-8 on the 2D Poisson problem with a "
        "spectrally rich source. These counts are properties of the mathematics and "
        "carry to any machine, which is why Section 8.3 reports them before any timing.",
        "tab:convergence",
    )


def table_backend_cost(data: pd.DataFrame) -> None:
    """Seconds per backend and method, each with its spread and its verdict.

    One row per measured configuration rather than a grid of backends against
    methods. The grid had no room for a spread beside each number without eleven
    columns of it, and ground rule 7 needs the spread beside the number it
    qualifies rather than in a separate table the reader has to join by hand.
    """
    block = data[data["label"] == "backend_cost"]
    if block.empty:
        return
    size = sorted(block["unknowns"].unique())[-1]
    block = block[block["unknowns"] == size]
    backends = list(dict.fromkeys(block["backend"]))
    solvers = list(dict.fromkeys(block["solver"]))
    baseline_name = "serial" if "serial" in backends else backends[0]
    header = ["backend", "method", "seconds", "spread (percent)",
              f"against {baseline_name} (percent)"]
    rows: list[list[Any]] = []
    sentences: list[str] = []
    for backend in backends:
        for solver in solvers:
            match = block[(block["backend"] == backend) & (block["solver"] == solver)]
            if match.empty:
                rows.append([backend, solver, "pending", "pending", "pending"])
                continue
            row = match.iloc[0]
            base = block[(block["backend"] == baseline_name) &
                         (block["solver"] == solver)]
            if backend == baseline_name:
                verdict = "baseline"
            elif base.empty:
                verdict = "pending"
            else:
                effect = separable_effect(row, base.iloc[0])
                verdict = fmt_effect(effect)
                sentences.append(effect_sentence(
                    f"{backend} on {solver}", f"{baseline_name} on {solver}",
                    effect, row, base.iloc[0]))
            rows.append([backend, solver, fmt(float(row["seconds_median"])),
                         fmt_spread(row), verdict])
    write_table(
        "backend_cost.tex", header, rows,
        f"Seconds for a fixed sweep count at {int(size):,} unknowns with 20 workers, "
        f"{repetitions_text(block)}. Every backend performs bit identical arithmetic, "
        "so the only variable is the execution model. The spread column is "
        f"(max - min) / median over those repetitions. The last column compares each "
        f"backend against {baseline_name} on the same method and reports "
        f"``{NOT_SEPARABLE}'' wherever the difference between the two medians is "
        "smaller than the larger of the two spreads, which is ground rule 7: an "
        "effect inside the noise is not a measurement of anything.",
        "tab:backend-cost", column_spec="llrrl",
    )
    write_fragment(
        "backend_cost_verdicts.tex",
        f"Read against {baseline_name} on the same method, at "
        f"{int(size):,} unknowns:",
        sentences,
    )


# Last level cache of each device, in MiB. The working set of a sweep is
# compared against these so a ratio computed under the streaming assumption is
# never presented as a DRAM efficiency when the problem in fact fits in cache.
# The RTX 5070 figure is its L2; the host figure is the L3 reported by sysfs.
CACHE_MIB = {"host": 33.0, "gpu": 48.0}

# How many times the last level cache the working set must exceed before a
# streaming efficiency is quoted. A working set merely larger than the cache is
# not enough: at twice the cache a good fraction of the traffic is still served
# from it, and the ratio comes out above one, which is the model announcing that
# it does not apply rather than a result. Four times is where the measured
# ratios settle below one and stay there.
STREAMING_MARGIN = 4.0


def working_set_mib(unknowns: float, vectors: int = 3) -> float:
    """Bytes a sweep touches, in MiB: the iterate, the previous iterate and the
    right hand side."""
    return unknowns * vectors * 8.0 / (1024.0 * 1024.0)


def cache_regime(mib: float, cache: float) -> str:
    """Which regime a working set is in: streaming, partly cached, or resident."""
    if mib < cache:
        return "resident"
    if mib < STREAMING_MARGIN * cache:
        return "partial"
    return "streaming"


def table_device_comparison(data: pd.DataFrame, bandwidth: dict[str, Any]) -> None:
    block = data[data["label"] == "device_comparison"]
    if block.empty:
        return
    host_peak = (bandwidth.get("host") or {}).get("gib_per_second")
    device_peak = (bandwidth.get("gpu") or {}).get("gib_per_second")

    header = ["method", "unknowns", "device", "working set", "GiB/s declared",
              "GiB/s counted", "percent declared", "percent counted", "seconds",
              "spread (percent)"]
    rows: list[list[Any]] = []
    cache_bound_seen = False
    predates_seen = False
    for solver in dict.fromkeys(block["solver"]):
        for size in sorted(block["unknowns"].unique()):
            for backend, peak, name, cache in (
                ("openmp", host_peak, "host, 20 threads", CACHE_MIB["host"]),
                ("cuda", device_peak, "RTX 5070", CACHE_MIB["gpu"]),
            ):
                match = block[(block["solver"] == solver) & (block["unknowns"] == size) &
                              (block["backend"] == backend)]
                if match.empty:
                    continue
                achieved = float(match["gib_per_second"].iloc[0])
                # Both traffic models, side by side, which is ground rule 9.
                # Neither replaces the other and the report says which is which.
                counted = counted_gib_per_second(match.iloc[0])
                if counted is None:
                    predates_seen = True
                mib = working_set_mib(float(size))
                # A working set that is not comfortably larger than the last
                # level cache means the sweep is not purely streaming from
                # memory, so the ratio is not a DRAM efficiency and must not be
                # presented as one.
                regime = cache_regime(mib, cache)
                if regime != "streaming":
                    cache_bound_seen = True
                if regime == "resident":
                    efficiency = "cache resident"
                    efficiency_counted = "cache resident"
                elif regime == "partial":
                    efficiency = "partly cached"
                    efficiency_counted = "partly cached"
                else:
                    efficiency = f"{achieved / peak * 100:.1f}" if peak else "pending"
                    if counted is None:
                        efficiency_counted = PREDATES
                    elif peak:
                        efficiency_counted = f"{counted / peak * 100:.1f}"
                    else:
                        efficiency_counted = "pending"
                rows.append([solver, f"{int(size):,}", name, f"{mib:.0f} MiB",
                             f"{achieved:.1f}",
                             PREDATES if counted is None else f"{counted:.1f}",
                             efficiency, efficiency_counted,
                             fmt(float(match["seconds_median"].iloc[0])),
                             fmt_spread(match.iloc[0])])

    caption = (
        "Achieved bandwidth, efficiency against each device's own measured STREAM triad, "
        "and absolute time. The efficiency columns are the comparable ones: they are "
        "dimensionless and say how well each device is used. The seconds column is a "
        "property of this particular pair of devices and of nothing else. "
        "Declared and counted are the two traffic models of Section 4.2, both published "
        "because neither has yet been selected: declared divides the sweeps of each "
        "iteration by the conservative byte count, which charges a read for a read and a "
        "write for a write, and counted divides the passes over memory by the same count "
        "with read for ownership charged on the one array a pass writes without reading "
        "first. The non temporal triad in the bandwidth table is the instrument that "
        "chooses between them and the rule is fixed in benchmarks/sweep\\_matrix.yaml "
        "before the measurement."
    )
    if predates_seen:
        caption += (
            f" A cell reading {PREDATES} belongs to a row measured before the passes and "
            "dram bytes columns existed. The counted figure needs both and is left "
            "underived rather than assumed: guessing one pass would halve the figure for "
            "the two red black methods, which are the ones the comparison turns on."
        )
    if cache_bound_seen:
        caption += (
            " A streaming efficiency is quoted only where the working set exceeds that "
            "device's last level cache by a factor of four. Rows marked cache resident "
            "have a working set smaller than the cache and rows marked partly cached "
            "have one within that factor; in both cases a good share of the traffic never "
            "reaches memory, so the sweep runs faster than the memory system alone would "
            "allow and a percentage would exceed one hundred. That would be the roofline "
            "assumption announcing that it does not apply, not a result. The comparison "
            "should therefore be read from the largest size, where both devices are "
            "unambiguously bandwidth bound."
        )
    caption += (
        " The spread column is (max - min) / median over the repetitions of that row, "
        "and both bandwidth columns and both percentage columns move inversely with it: "
        "a row whose spread is ten percent has a bandwidth figure good to about ten "
        "percent, whichever byte count it is divided by."
    )
    write_table("device_comparison.tex", header, rows, caption, "tab:device-comparison",
                column_spec="llrrrrrrrr")


def table_bandwidth(bandwidth: dict[str, Any]) -> None:
    rows = []
    for key, label in (("host", "host, all threads, plain stores"),
                       ("host_nontemporal", "host, all threads, non temporal stores"),
                       ("gpu", "RTX 5070")):
        entry = bandwidth.get(key) or {}
        value = entry.get("gib_per_second")
        rows.append([label, fmt(value, 1) if value else "pending",
                     entry.get("detail", "").strip()])
    # Arithmetic on the rows above, kept visibly apart from them. The manifest
    # files these under bandwidth.derived for the same reason.
    for name, item in (bandwidth.get("derived") or {}).items():
        value = item.get("value")
        rows.append([f"derived, {name.replace('_', ' ')}",
                     fmt(value, 3) if value else "pending",
                     str(item.get("note", "")).strip()])
    write_table(
        "bandwidth.tex", ["device", "GiB/s", "measurement"], rows,
        "Measured STREAM triad bandwidth. Every efficiency figure in this report "
        "divides by these values, which were measured on this machine, and never by a "
        "manufacturer's specification. The plain probe is a C++ loop, so the compiler "
        "emits ordinary stores into an array the loop never reads; the non temporal probe "
        "writes the same triad through streaming stores, which move 24 bytes per element "
        "for real. Both report against the same declared 24, so whether the plain loop "
        "also fetches each output line before overwriting it, and moves 32 where it "
        "declares 24, is exactly what their ratio measures. That ratio selects between "
        "the two traffic models of Section 4.2 under the rule fixed in "
        "benchmarks/sweep\\_matrix.yaml before the measurement was taken. The rows "
        "marked derived are arithmetic on the rows above and were not measured. No "
        "memory speed is recorded anywhere in this repository, so none of these figures "
        "is compared against a theoretical peak.",
        "tab:bandwidth",
        column_spec="lrp{0.5\\textwidth}",
    )


def table_reduction_cost(data: pd.DataFrame) -> None:
    block = data[data["label"] == "reduction_cost"]
    if block.empty:
        return
    backends = list(dict.fromkeys(block["backend"]))
    rows: list[list[Any]] = []
    sentences: list[str] = []
    for backend in backends:
        deterministic = block[(block["backend"] == backend) &
                              (block["reduction"] == "deterministic")]
        native = block[(block["backend"] == backend) & (block["reduction"] == "native")]
        if deterministic.empty or native.empty:
            continue
        d_row = deterministic.iloc[0]
        n_row = native.iloc[0]
        effect = separable_effect(d_row, n_row)
        rows.append([backend, fmt(float(d_row["seconds_median"])), fmt_spread(d_row),
                     fmt(float(n_row["seconds_median"])), fmt_spread(n_row),
                     fmt_effect(effect)])
        sentences.append(effect_sentence(
            f"the deterministic reduction on {backend}",
            "the model's native reduction", effect, d_row, n_row))
    write_table(
        "reduction_cost.tex",
        ["backend", "deterministic (s)", "spread", "native (s)", "spread",
         "overhead (percent)"], rows,
        "The price of reproducibility. The deterministic mode combines partial sums in a "
        "fixed order that does not depend on the worker count, which is what lets the "
        "equivalence suite assert bit identical results; the native mode uses the "
        "model's own reduction. Conjugate gradient is used because its two global "
        "reductions per iteration make it the method most exposed to the difference. "
        "Each spread column is (max - min) / median over that configuration's "
        f"repetitions, {repetitions_text(block)}. The overhead column reads "
        f"``{NOT_SEPARABLE}'' wherever the difference between the two medians is "
        "smaller than the larger of the two spreads. This is the table ground rule 7 was "
        "written for: at 1.0.0 it reported effects of minus 5.5 to plus 3.1 percent on "
        "runs of seventy two milliseconds, against a block whose median spread was 9.3 "
        "percent, and the README quoted those effects as what reproducibility is worth.",
        "tab:reduction-cost", column_spec="lrrrrl",
    )
    write_fragment(
        "reduction_cost_verdicts.tex",
        "What the deterministic reduction costs, per backend:", sentences,
    )


def table_pinning(data: pd.DataFrame) -> None:
    block = data[data["label"] == "pinning"]
    if block.empty:
        return
    policies = list(dict.fromkeys(block["pinning"]))
    backends = list(dict.fromkeys(block["backend"]))
    workers = sorted(block["workers"].unique())
    baseline_policy = "none" if "none" in policies else policies[0]
    header = ["backend", "workers", "policy", "seconds", "spread (percent)",
              f"against {baseline_policy} (percent)"]
    rows: list[list[Any]] = []
    sentences: list[str] = []
    for backend in backends:
        for worker_count in workers:
            for policy in policies:
                match = block[(block["backend"] == backend) &
                              (block["workers"] == worker_count) &
                              (block["pinning"] == policy)]
                label = [backend, str(int(worker_count)), policy]
                if match.empty:
                    rows.append([*label, "pending", "pending", "pending"])
                    continue
                row = match.iloc[0]
                base = block[(block["backend"] == backend) &
                             (block["workers"] == worker_count) &
                             (block["pinning"] == baseline_policy)]
                if policy == baseline_policy:
                    verdict = "baseline"
                elif base.empty:
                    verdict = "pending"
                else:
                    effect = separable_effect(row, base.iloc[0])
                    verdict = fmt_effect(effect)
                    sentences.append(effect_sentence(
                        f"{policy} pinning on {backend} at {int(worker_count)} workers",
                        f"{baseline_policy} pinning on the same configuration",
                        effect, row, base.iloc[0]))
                rows.append([*label, fmt(float(row["seconds_median"])),
                             fmt_spread(row), verdict])
    compared = [row[-1] for row in rows if row[-1] not in ("baseline", "pending")]
    inside = sum(1 for value in compared if value == NOT_SEPARABLE)
    write_table(
        "pinning.tex", header, rows,
        "Seconds by thread binding policy. Under WSL2 an affinity request binds a thread "
        "to a guest virtual processor and the hypervisor remains free to place that "
        "processor on any host core, so the expectation here is a smaller effect than the "
        "same experiment would show on bare metal. The spread column is "
        f"(max - min) / median over that row's repetitions, {repetitions_text(block)}, "
        f"and the last column reads ``{NOT_SEPARABLE}'' wherever a policy's difference "
        f"from {baseline_policy} is smaller than the larger of the two spreads. That is "
        f"{inside} of the {len(compared)} comparisons this block supports, which is the "
        "result rather than a gap in it.",
        "tab:pinning", column_spec="llrrrl",
    )
    write_fragment(
        "pinning_verdicts.tex",
        f"Each binding policy read against {baseline_policy} at the same backend and "
        "worker count:", sentences,
    )


def table_schedule(data: pd.DataFrame) -> None:
    block = data[data["label"] == "schedule_cost"]
    if block.empty:
        return
    rows: list[list[Any]] = []
    sentences: list[str] = []
    for backend in dict.fromkeys(block["backend"]):
        static = block[(block["backend"] == backend) & (block["schedule"] == "static")]
        dynamic = block[(block["backend"] == backend) & (block["schedule"] == "dynamic")]
        if static.empty or dynamic.empty:
            continue
        s_row = static.iloc[0]
        d_row = dynamic.iloc[0]
        effect = separable_effect(d_row, s_row)
        rows.append([backend, fmt(float(s_row["seconds_median"])), fmt_spread(s_row),
                     fmt(float(d_row["seconds_median"])), fmt_spread(d_row),
                     fmt_effect(effect)])
        sentences.append(effect_sentence(
            f"a dynamic schedule on {backend}", "a static one on the same backend",
            effect, d_row, s_row))
    write_table(
        "schedule_cost.tex",
        ["backend", "static (s)", "spread", "dynamic (s)", "spread",
         "dynamic overhead (percent)"], rows,
        "What a dynamic schedule costs on uniform work. This is the measurement behind "
        "leaving work stealing out of the jthread pool: on a balanced stencil sweep the "
        "extra bookkeeping buys nothing. Each spread column is (max - min) / median over "
        f"that configuration's repetitions, {repetitions_text(block)}, and the overhead "
        f"column reads ``{NOT_SEPARABLE}'' wherever the difference between the two "
        "medians is smaller than the larger of the two spreads. A row that reads the "
        "phrase is not evidence that a dynamic schedule is free; it is evidence that this "
        "block cannot price it, and the argument for leaving work stealing out has to "
        "rest on the rows that do separate.",
        "tab:schedule-cost", column_spec="lrrrrl",
    )
    write_fragment(
        "schedule_cost_verdicts.tex",
        "What a dynamic schedule costs against a static one, per backend:", sentences,
    )


def interval_text(knee: dict[str, Any]) -> str:
    """The bootstrap interval of one knee, as a table cell."""
    low, high = knee.get("low"), knee.get("high")
    if low is None or high is None:
        return "pending"
    return f"{int(low)} to {int(high)}"


def table_knee(knees: dict[str, Any]) -> None:
    fitted = {backend: knee for backend, knee in knees.items() if knee}
    if not fitted:
        return
    rows = [
        [backend, str(int(knee.get("workers", 0))), interval_text(knee),
         fmt(knee.get("slope_before"), 2), fmt(knee.get("slope_after"), 2),
         str(knee.get("method", "pending"))]
        for backend, knee in fitted.items()
    ]

    methods = sorted({str(knee.get("method")) for knee in fitted.values()})
    seeds = sorted({int(knee.get("seed", KNEE_BOOTSTRAP_SEED)) for knee in fitted.values()})
    samples = sorted({int(knee.get("samples", 0)) for knee in fitted.values()})
    caption = (
        "The knee in the scaling curve, found as the two segment split that minimises "
        "total least squares residual. The guest cannot identify which logical processors "
        "are performance cores, so this is read from the aggregate curve, which depends "
        "only on how many cores are engaged and is therefore valid under virtualisation. "
        f"The interval is the 2.5 to 97.5 percentile of {KNEE_BOOTSTRAP_SAMPLES} "
        f"bootstrap refits at seed {', '.join(str(s) for s in seeds)}, "
        f"of which {', '.join(str(s) for s in samples)} returned a fit: each draw "
        "resamples the repetitions behind every worker count, takes the median of the "
        "draw, rebuilds the speedup curve against the resampled one worker baseline and "
        "refits. It covers the sampling of the repetitions and nothing else, so it says "
        "nothing about a machine that was busy at one worker count and idle at another."
    )
    if MEASURED_REPS in methods:
        caption += (
            f" A row marked {MEASURED_REPS} resamples the individual timings recorded in "
            "the seconds\\_reps column."
        )
    if TRIANGULAR in methods:
        caption += (
            f" A row marked {TRIANGULAR} has no such column, which is every row measured "
            "before phase A1.5 added it, and its repetitions are drawn instead from a "
            "triangular distribution on the recorded minimum, median and maximum. That is "
            "an assumed shape rather than a measured one, and its interval is worth less "
            "than the other. The two are never mixed inside one backend, which is why the "
            "method is a column and not a footnote."
        )
    write_table(
        "knee.tex",
        ["backend", "knee at workers", "95 percent interval",
         "speedup per worker before", "after", "resampling"],
        rows, caption, "tab:knee", column_spec="lllrrl",
    )

    sentences = [
        (f"On {backend} the fit puts the knee at {int(knee.get('workers', 0))} workers, "
         f"with a bootstrap interval of {interval_text(knee)} workers "
         f"({knee.get('method')}).")
        for backend, knee in fitted.items()
    ]
    positions = {int(knee.get("workers", 0)) for knee in fitted.values()}
    lows = [knee.get("low") for knee in fitted.values()]
    highs = [knee.get("high") for knee in fitted.values()]
    if len(positions) > 1:
        overlap = (None not in lows and None not in highs and
                   max(lows) <= min(highs))
        if overlap:
            sentences.append(
                "The backends do not agree on a single knee position, but every one of "
                f"their intervals contains {int(max(lows))} to {int(min(highs))} workers, "
                "so the disagreement between the point estimates is inside what this data "
                "can resolve and is not by itself a finding.")
        else:
            sentences.append(
                "No worker count lies inside every backend's interval, so the "
                "disagreement survives the bootstrap: it is a result and not a fitting "
                "artefact, and the point past which adding workers buys measurably less "
                "is genuinely not the same for these thread models on this machine.")
    else:
        sentences.append(
            "Every backend puts the knee at the same worker count, which is what the "
            "aggregate curve reading assumes and does not always deliver.")
    write_fragment(
        "knee_verdicts.tex",
        "Where the scaling curve changes slope, and how well that position is pinned "
        "down:", sentences,
    )


def table_dispersion(data: pd.DataFrame) -> None:
    """How much the timings move between repetitions, per block of the matrix.

    Deliberately n, median and max, and no p90. At eight rows in the reduction
    cost block and six in the schedule cost block a ninetieth percentile is the
    single worst row wearing the name of a statistic, and quoting it as a
    dispersion figure is exactly the over claiming that ground rule 7 exists to
    forbid.
    """
    rows: list[list[Any]] = []
    smallest: int | None = None
    single = []
    for label in dict.fromkeys(data["label"]):
        if not label:
            continue
        block = data[data["label"] == label]
        reps = pd.to_numeric(block["reps"], errors="coerce").dropna() \
            if "reps" in block.columns else pd.Series(dtype=float)
        if len(reps) and int(reps.max()) <= 1:
            # Measured once, so there is no dispersion to report: an iteration
            # count is deterministic and repeating it measures nothing.
            single.append(str(label).replace("_", " "))
            continue
        values = [value for value in (spread(row) for _, row in block.iterrows())
                  if value is not None]
        if not values:
            continue
        smallest = len(values) if smallest is None else min(smallest, len(values))
        rows.append([str(label).replace("_", " "), str(len(values)),
                     f"{statistics.median(values) * 100:.1f}",
                     f"{max(values) * 100:.1f}"])
    if not rows:
        return
    caption = (
        "Run to run spread by block of the sweep matrix, defined throughout this report "
        "as (max - min) / median over the repetitions of one configuration. Every timing "
        "table in this chapter carries this quantity beside the number it qualifies, and "
        "every timing figure draws it as a whisker; this table is the summary. "
        f"There is deliberately no p90 column. The smallest block here has {smallest} "
        "rows, and a ninetieth percentile over that many is the single worst row wearing "
        "the name of a statistic, which is the over claiming that ground rule 7 exists to "
        "forbid. n, the median and the maximum are what these sample sizes support."
    )
    if single:
        caption += (
            f" Measured once and therefore absent: {', '.join(single)}. An iteration "
            "count is a property of the mathematics and repeating it measures nothing."
        )
    write_table("dispersion.tex", ["block", "n", "median spread (percent)",
                                   "max (percent)"], rows, caption, "tab:dispersion")


def manifest_slug(commit: str) -> str:
    """The commit as it appears inside a manifest file name."""
    return (commit or "unknown").replace(".", "-")


def manifests_for(results: Path, commit: str) -> list[Path]:
    """Every manifest in `results` written for `commit`, oldest first.

    Anchored rather than globbed, because the slug of a clean commit is a prefix
    of the slug of its dirty twin and a glob would let the wrong one answer. The
    timestamp is optional so a manifest archived under its bare name is still
    recognised as belonging to its commit.
    """
    pattern = re.compile(
        rf"^{re.escape(MANIFEST_PREFIX)}{re.escape(manifest_slug(commit))}"
        r"(?:-(\d{8}T\d{6}Z))?\.json$"
    )
    found: list[tuple[str, Path]] = []
    if results.is_dir():
        for path in results.glob(f"{MANIFEST_PREFIX}*.json"):
            match = pattern.match(path.name)
            if match:
                found.append((match.group(1) or "", path))
    return [path for _, path in sorted(found)]


def select_generation(data: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    """The rows of the one generation in the summary, and which commit it is.

    Returns the rows, the commit, and a refusal message which is empty when
    there is exactly one. Nothing is dropped and no commit is preferred: two
    generations in one file is a state the archive procedure exists to resolve,
    and choosing between them here is what release 1.0.0 did by accident.
    """
    if "commit" not in data.columns or not len(data):
        return data, "", ""
    commits = sorted(data["commit"].astype(str).unique())
    if len(commits) > 1:
        counts = ", ".join(
            f"{commit} ({int((data['commit'].astype(str) == commit).sum())} rows)"
            for commit in commits
        )
        return data, "", (
            f"gen_report_assets: refusing to build assets from {len(commits)} generations "
            f"in one summary: {counts}. A report whose rows came from different binaries "
            f"is not one measurement. Move the superseded rows to "
            f"experiments/results/archive/summary-<commit>.csv, as "
            f"experiments/results/archive/README.md describes, and leave one generation "
            f"here."
        )
    commit = commits[0]
    # measured_at orders rows inside the generation, and row order orders
    # nothing. Sorting here is what makes any later "latest" read the clock
    # rather than the append order that put two generations in one file.
    if "measured_at" in data.columns:
        data = data.sort_values("measured_at", kind="stable")
    return data, commit, ""


def select_manifest(results: Path, commit: str, allow_dirty: bool) -> tuple[dict[str, Any], str]:
    """The session manifest for `commit`, and a refusal message if there is none.

    Searched in the results directory first and in its archive second. An
    archived manifest is announced as archived on every run, because a manifest
    that has been archived is one that no longer describes the session whose
    rows are being published, and that is exactly the confusion finding 4.3 was
    made of. It is only ever reached with --allow-dirty, which has already said
    that nothing built from this run is publishable.
    """
    candidates = manifests_for(results, commit)
    if candidates:
        if len(candidates) > 1:
            print(f"gen_report_assets: {len(candidates)} manifests carry commit {commit}: "
                  + ", ".join(path.name for path in candidates)
                  + f". Using the latest, {candidates[-1].name}.")
        chosen = candidates[-1]
        print(f"gen_report_assets: manifest {chosen.name}")
        return json.loads(chosen.read_text(encoding="utf-8")), ""

    archived = manifests_for(results / ARCHIVE, commit)
    if archived and allow_dirty:
        chosen = archived[-1]
        print(f"gen_report_assets: no manifest for commit {commit} in "
              f"{results.name}, falling back to the archived "
              f"{ARCHIVE}/{chosen.name}. An archived manifest does not describe the "
              f"session that measured these rows, so every bandwidth figure below is "
              f"provenance this generation does not have.")
        return json.loads(chosen.read_text(encoding="utf-8")), ""

    message = (
        f"gen_report_assets: no session manifest for commit {commit}. Every efficiency "
        f"figure divides by a bandwidth this machine measured, and without the manifest "
        f"of the session that measured these rows there is no such number. Run the sweep, "
        f"or point --results-dir at the directory holding "
        f"{MANIFEST_PREFIX}{manifest_slug(commit)}-<timestamp>.json."
    )
    if archived:
        message += (f" One exists under {ARCHIVE}/, and --allow-dirty will fall back to "
                    f"it and say so.")
    return {}, message


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate every figure and table in the report from summary.csv.")
    parser.add_argument(
        "--allow-dirty", action="store_true",
        help="build assets from rows whose commit stamp ends in .dirty. Ground rule 6 "
             "says no published number may come from a tree that is not in git history, "
             "so without this flag the generator refuses and names how many such rows it "
             "found. Every row in the committed summary carries a .dirty stamp today, so "
             "every command in phase A7 passes this flag, and the CI reports job passes "
             "it until phase A8b lands a generation measured from a clean tree. It also "
             "lets an archived manifest stand in for a missing one, which the published "
             "generation needs because its manifest describes a different session. For "
             "development only: an asset built with this flag is not publishable.")
    parser.add_argument(
        "--results-dir", type=Path, default=None,
        help="read summary.csv and the session manifest from here instead of "
             "experiments/results. experiments/results/interim is where a sweep that "
             "proves the pipeline writes, so this is how its assets are generated "
             "without touching the generation the report is built from.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # The module level paths stay the defaults, so the self test can point them
    # at a workspace, and --results-dir overrides both together: a summary from
    # one directory and a manifest from another would be two sessions again.
    if args.results_dir:
        results = Path(args.results_dir)
        summary = results / "summary.csv"
    else:
        summary = SUMMARY
        results = SUMMARY.parent
    if not summary.exists():
        print(f"gen_report_assets: {summary} not found. Run make sweep first.",
              file=sys.stderr)
        return 2

    data = pd.read_csv(summary)
    for column in ("unknowns", "workers", "iterations", "converged", "seconds_median",
                   "gib_per_second", "updates_per_second"):
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["label"] = data["label"].fillna("").astype(str).str.split().str[0]

    # One generation, and no choosing between two.
    #
    # The summary accumulates across commits by design: a row is keyed partly on
    # the commit that produced it, so a rebuild adds rows rather than replacing
    # them and the history is preserved. That is right for the data file and
    # wrong for the report, where mixing a stale build with the current one would
    # produce a table whose rows came from different binaries. Release 1.0.0
    # resolved that by taking the commit of the last row in the file, which is a
    # statement about the order two sweeps were appended in and about nothing
    # else. It is a refusal now, and the archive is where the other generation
    # goes.
    data, commit, refusal = select_generation(data)
    if refusal:
        print(refusal, file=sys.stderr)
        return 4

    # Ground rule 6, checked over the rows this run would actually publish. A
    # .dirty stamp means the exact source that produced the number is not in git
    # history, so the figure is unreproducible by anyone including its author.
    dirty = 0
    if "commit" in data.columns and len(data):
        dirty = int(data["commit"].astype(str).str.endswith(".dirty").sum())
    if dirty and not args.allow_dirty:
        print(f"gen_report_assets: refusing to build a published asset from {dirty} of "
              f"{len(data)} row(s) whose commit stamp ends in .dirty. The source that "
              f"produced those numbers is not in git history. Re measure from a clean "
              f"tree, or pass --allow-dirty to generate anyway, which is for development "
              f"only and produces nothing publishable.", file=sys.stderr)
        return 3
    if dirty:
        print(f"gen_report_assets: --allow-dirty, {dirty} of {len(data)} row(s) carry a "
              f".dirty commit stamp and nothing built from them is publishable")

    manifest, missing = select_manifest(results, commit, args.allow_dirty)
    if missing:
        print(missing, file=sys.stderr)
        return 5
    bandwidth: dict[str, Any] = manifest.get("bandwidth", {})

    where = summary.relative_to(ROOT) if summary.is_relative_to(ROOT) else summary
    print(f"gen_report_assets: {len(data)} rows at commit {commit or 'unknown'} from {where}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    # Every figure is drawn once per selected theme. The knee fit is a property
    # of the data rather than the palette, so the first pass is the one whose
    # result is kept.
    knees: dict[str, Any] = {}
    for theme in THEMES:
        set_theme(theme)
        fitted = figure_scaling(data)
        knees = knees or fitted
        figure_backend_cost(data)
        figure_device_efficiency(data, bandwidth)
        figure_iteration_counts(data)
        figure_convergence_growth(data)
        figure_mpi_communication(data)
    set_theme("light")

    table_convergence(data)
    table_backend_cost(data)
    table_device_comparison(data, bandwidth)
    table_bandwidth(bandwidth)
    table_reduction_cost(data)
    table_pinning(data)
    table_schedule(data)
    table_knee(knees)
    table_dispersion(data)

    print("gen_report_assets: done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
