"""Render the valuation dashboard: one self-contained HTML file.

Everything - styles, script, data - is inlined, so the file opens from disk
with no server and no network, and a copy emailed to someone else still works.
It is regenerated on every run alongside the tracker workbook.

Layout, top to bottom:

* a filter row that scopes everything beneath it
* headline figures for the filtered set
* **where the price sits against estimated value** - one row per company,
  normalised to its own median estimate so a $10 stock and a $300 stock share
  an axis; the band is the spread of the value models, the dot is the price
* **what the price assumes** - the reverse DCF against the growth the record
  supports, which is the one valuation number that cannot be tuned
* the scorecard, each row expanding to every test, every model, and the
  earnings history behind them
* where each threshold comes from

Chart conventions follow the dataviz method: palettes validated in both modes
(blue/red diverging pair: CVD dE 21.6 light, 19.2 dark; one-hue dumbbell
steps: monotone, light end 2.06:1), thin marks, hairline grid, status colours
only for status and always with a glyph and a label, every chart with a table
twin, and every string from a filing inserted as text, never as markup.
"""

from __future__ import annotations

import json
import math
import re
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

from .buffett import REVERSE_DCF_BOUNDS
from .config import (CONFIG_VERSION, PROVENANCE, REPORT_DIR, ROOT,
                     SOURCE_TIERS, ValuationConfig)
from .score import VERDICTS

DASHBOARD_PATH = ROOT / 'valuation_dashboard.html'
# Runs of price-to-value history drawn per company; a longer record is thinned
# evenly, always keeping the latest run.
HISTORY_POINTS = 400
# Archived copies kept for every day this recent; older ones only as the
# first copy of each month.
ARCHIVE_DAYS = 30
# What the page reads of each criterion. Its citation is looked up from the
# provenance table by key rather than repeated on every row.
CRITERION_FIELDS = ('key', 'number', 'name', 'verdict', 'display', 'threshold',
                    'detail', 'source_tier', 'structural')
# Above this many companies, each one's detail - its tests, models, earnings
# record and history - goes into a small script file of its own beside the
# page, loaded when its row is opened. The S&P 500 inline would be a page of
# well over 10 MB; a watchlist inline stays one self-contained file.
INLINE_LIMIT = 150
DETAIL_KEYS = ('criteria', 'models', 'consensus_note', 'mos_note', 'caveats', 'fund',
               'history', 'growth_source')
DETAILS_DIR = 'valuation_details'
# The change log shown on the page.
CHANGES_DAYS = 30


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------

def _j(v):
    """Anything -> what json.dumps accepts: NaN -> null, floats to 6 figures."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return float(f'{f:.6g}') if math.isfinite(f) else None
    if isinstance(v, (pd.Timestamp, datetime)):
        return None if pd.isna(v) else v.strftime('%Y-%m-%d')
    if isinstance(v, dict):
        return {str(k): _j(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_j(x) for x in v]
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _tally(t: dict) -> list:
    """[passed, applicable, not measurable] - what the scorecard meter reads."""
    return [t['passed'], t['applicable'], t['na']]


def _history_by_ticker(history: pd.DataFrame | None) -> dict:
    """``{ticker: [{'d', 'p', 'v', 'p2v', 'verdict'}, ...]}``, grouped once.

    It was filtered from the whole history table once per company - a cost of
    companies x runs that grew with every day the tracker ran.
    """
    cols = ['Ticker', 'date', 'Price', 'IntrinsicValue', 'PriceToValue', 'Verdict']
    if history is None or history.empty or not {'Ticker', 'date'} <= set(history.columns):
        return {}
    h = history.reindex(columns=cols).copy()
    h['date'] = pd.to_datetime(h['date'])
    out = {}
    for t, g in h.sort_values('date', kind='stable').groupby('Ticker', sort=False):
        rows = list(g.itertuples(index=False))
        if len(rows) > HISTORY_POINTS:
            rows = rows[::-1][::math.ceil(len(rows) / HISTORY_POINTS)][::-1]
        out[t] = [{'d': r.date, 'p': r.Price, 'v': r.IntrinsicValue,
                   'p2v': r.PriceToValue, 'verdict': r.Verdict} for r in rows]
    return out


def _display_name(name):
    """SEC entity names carry a state-of-incorporation suffix ('US BANCORP \\DE\\')."""
    if not name:
        return name
    return re.sub(r'\s*/?\\[A-Z]{2}\\/?\s*$', '', str(name)).strip()


def _company(t: str, r: dict, panel: pd.DataFrame, history: list) -> dict:
    """One company as the page reads it - only the fields the script uses."""
    cons = r.get('consensus') or {}
    mos = r.get('margin_of_safety') or {}
    s = r.get('summary') or {}
    models = r.get('models') or {}
    value = cons.get('value')
    low, high = cons.get('low'), cons.get('high')

    def rel(x):
        return (x / value) if (x is not None and value) else None

    rev = models.get('reverse_dcf') or {}
    dcf_d = (models.get('dcf') or {}).get('detail') or {}
    used = cons.get('models_used') or []
    order = ['dcf', 'epv', 'earnings_multiple', 'justified_pb', 'graham_number',
             'ncav', 'ncav_two_thirds', 'growth_formula', 'reverse_dcf']
    model_rows = [{
        'label': m.get('label'), 'role': m.get('role'), 'value': m.get('value'),
        'applicable': bool(m.get('applicable')), 'in_consensus': k in used,
        'note': m.get('note'), 'is_rate': bool(m.get('is_rate')),
    } for k in order + [k for k in models if k not in order] if (m := models.get(k))]

    tail = panel.tail(20) if panel is not None and not panel.empty else None
    return {
        't': t, 'name': _display_name(r.get('name')), 'group': r.get('group'),
        'industry': r.get('sic_description'),
        'price': r.get('price'), 'price_date': r.get('price_date'),
        'verdict': r.get('verdict'), 'completeness': r.get('completeness'),
        'value': value, 'low': low, 'high': high,
        'low_rel': rel(low), 'high_rel': rel(high),
        'p2v': mos.get('price_to_value'),
        'mos_measured': mos.get('mos') if mos.get('verdict') == 'MEASURED' else None,
        'mos_verdict': mos.get('verdict'), 'mos_band': mos.get('band'),
        'n_models': cons.get('n_models'),
        'implied_growth': rev.get('value') if rev.get('applicable') else None,
        'implied_at_bound': bool((rev.get('detail') or {}).get('at_bound')),
        'implied_range': (rev.get('detail') or {}).get('range') if rev.get('applicable') else None,
        'equity_rate': (rev.get('detail') or {}).get('rate') or dcf_d.get('rate'),
        'credited_growth': dcf_d.get('growth_stage1'),
        'oe_growth': s.get('owner_earnings_cagr'),
        'rev_growth': s.get('revenue_cagr'),
        'growth_source': dcf_d.get('growth_source'),
        'tally': {'def': _tally(r['defensive_tally']),
                  'ent': _tally(r['enterprising_tally']),
                  'qual': _tally(r['quality_tally'])},
        'criteria': {block: [{k: row[k] for k in CRITERION_FIELDS}
                             for row in (c.to_row() for c in r[block])]
                     for block in ('defensive', 'enterprising', 'quality')},
        'models': model_rows,
        'consensus_note': cons.get('note'), 'mos_note': mos.get('note'),
        'caveats': r.get('caveats') or [],
        'n_years': r.get('n_years'), 'last_fy': r.get('last_fy'),
        'fund': ({'years': list(tail.index),
                  'eps': tail['eps'].tolist() if 'eps' in tail.columns else []}
                 if tail is not None else {}),
        'history': history,
    }


def _recent_changes(changes: pd.DataFrame | None, days: int = CHANGES_DAYS,
                    cap: int = 400) -> list:
    """The change log's last `days` days, newest first, without first evaluations."""
    if changes is None or changes.empty or 'date' not in changes.columns:
        return []
    c = changes.reindex(columns=['date', 'Ticker', 'change', 'before', 'after', 'detail']).copy()
    c['date'] = pd.to_datetime(c['date'])
    c = c[(c['date'] >= pd.Timestamp.today().normalize() - pd.Timedelta(days=days))
          & (c['change'] != 'new')]
    c = c.sort_values('date', ascending=False, kind='stable').head(cap)
    return [{'d': r.date, 't': r.Ticker, 'change': r.change, 'before': r.before,
             'after': r.after, 'detail': r.detail} for r in c.itertuples()]


def build_payload(results: dict, panels: dict, macro: dict,
                  history: pd.DataFrame | None, cfg: ValuationConfig,
                  changes: pd.DataFrame | None = None, lists: dict | None = None) -> dict:
    """The page's data. `history` and `changes` are the tracker store's
    tables; `lists` maps each ticker to the watchlists it came from."""
    hist = _history_by_ticker(history)
    companies = []
    for t, r in results.items():
        panel = r.get('panel_used')
        if panel is None:
            panel = panels.get(t)
        c = _company(t, r, panel, hist.get(t, []))
        c['lists'] = (lists or {}).get(t, [])
        companies.append(c)
    return _j({
        # UTC, stated: the page is built on a Singapore laptop and on a cloud
        # runner, and an unlabelled time is wrong for one of them.
        'generated': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'config_version': CONFIG_VERSION,
        'macro': {
            'aaa': macro.get('aaa_yield'), 'aaa_asof': macro.get('aaa_as_of'),
            'treasury_10y': macro.get('treasury_10y'),
            'cpi_ratio': macro.get('cpi_ratio'), 'cpi_asof': macro.get('cpi_as_of'),
            'cpi_base_year': macro.get('cpi_base_year', cfg.graham.base_year),
        },
        'params': {
            'discount_rate': cfg.buffett.discount_rate,
            'stage1_years': cfg.buffett.stage1_years,
            'max_dispersion': cfg.score.max_model_dispersion,
            'max_stage1_growth': cfg.buffett.max_stage1_growth,
            'mos_strong': cfg.score.mos_strong,
            'solver': list(REVERSE_DCF_BOUNDS),
        },
        'verdicts': [{'label': label, 'status': st, 'candidate': cand}
                     for label, _, st, cand in VERDICTS],
        'changes': _recent_changes(changes),
        'companies': companies,
        'tiers': SOURCE_TIERS,
        'provenance': [{'key': k, 'tier': v[0], 'citation': v[1]}
                       for k, v in PROVENANCE.items()],
    })


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def render(payload: dict, path=DASHBOARD_PATH, archive: bool = True):
    """Write the dashboard. Returns the path written.

    The JSON is escaped for embedding inside a <script> block: every `<`
    becomes \\u003c, so a company name containing `</script>` - or anything a
    filing might plausibly carry - cannot terminate the block early.
    """
    payload = dict(payload)
    lazy = len(payload.get('companies') or []) > INLINE_LIMIT
    details_dir = path.parent / DETAILS_DIR
    if lazy:
        payload['companies'] = _write_details(payload['companies'], details_dir)
        payload['details'] = DETAILS_DIR + '/'
    blob = json.dumps(payload, separators=(',', ':'), ensure_ascii=False)
    blob = blob.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    shell = _TEMPLATE.replace('/*__CSS__*/', _CSS).replace('/*__JS__*/', _JS)

    def page(workbook_href):
        # The workbook sits beside the dashboard (and beside the published
        # page); the archived copies are one folder down.
        return shell.replace('__WORKBOOK__', workbook_href).replace('/*__DATA__*/', blob)

    path.write_text(page('valuation_tracker.xlsx'), encoding='utf-8')
    # An archived copy must stand on its own; with its details in files that
    # the next run overwrites, it would show today's detail under an old
    # date. The history store keeps every run either way.
    if archive and not lazy:
        stamp = datetime.now().strftime('%Y%m%d')
        (REPORT_DIR / f'valuation_dashboard_{stamp}.html').write_text(
            page('../valuation_tracker.xlsx'), encoding='utf-8')
    if archive:
        prune_archive(REPORT_DIR)
    return path


def _write_details(companies: list, folder) -> list:
    """Move each company's detail into `folder/<ticker>.js` and return the
    summaries. A script file, not JSON, so the page can load it with a
    <script> tag - which, unlike fetch(), also works opened from disk."""
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob('*.js'):
        old.unlink()
    out = []
    for c in companies:
        c = dict(c)
        detail = {k: c.pop(k) for k in DETAIL_KEYS if k in c}
        c['file'] = re.sub(r'[^A-Za-z0-9_-]', '_', c['t']) + '.js'
        body = json.dumps(detail, separators=(',', ':'), ensure_ascii=False)
        body = body.replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')
        (folder / c['file']).write_text(
            f'window.VT_DETAIL({json.dumps(c["t"])},{body});\n', encoding='utf-8')
        out.append(c)
    return out


def prune_archive(folder, keep_days: int = ARCHIVE_DAYS, today: date | None = None) -> list:
    """Delete archived dashboards older than `keep_days`, except the first
    copy of each month. Returns what was deleted.

    One self-contained copy a day is ~1-2 MB for a large watchlist, which is
    half a gigabyte a year in a folder nobody prunes by hand.
    """
    today = today or date.today()
    dated = {}
    for p in folder.glob('valuation_dashboard_*.html'):
        m = re.fullmatch(r'valuation_dashboard_(\d{8})\.html', p.name)
        if m:
            try:
                dated[p] = datetime.strptime(m.group(1), '%Y%m%d').date()
            except ValueError:
                continue
    first_of_month = {}
    for p, d in sorted(dated.items(), key=lambda kv: kv[1]):
        first_of_month.setdefault((d.year, d.month), p)
    keep = set(first_of_month.values())
    removed = []
    for p, d in dated.items():
        if (today - d).days < keep_days or p in keep:
            continue
        try:
            p.unlink()
            removed.append(p)
        except OSError:
            pass
    return removed


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Value Tracker</title>
<style>/*__CSS__*/</style>
</head>
<body>
<div class="page">
  <header class="top">
    <div class="title">
      <h1>Value tracker</h1>
      <p class="sub">Graham's defensive and 1976 criteria, Buffett's quality tests, and a range of intrinsic-value estimates, from SEC filings.</p>
    </div>
    <button type="button" id="theme" class="ghost" aria-label="Toggle light or dark theme">Theme</button>
  </header>
  <p class="asof" id="asof"></p>
  <div class="notice stale" id="stale" role="status" hidden></div>

  <section class="filters" aria-label="Filters">
    <label class="f-search"><span class="vh">Search</span>
      <input id="f-q" type="search" placeholder="Ticker or name" autocomplete="off"></label>
    <label id="f-list-wrap" hidden><span class="f-lbl">List</span><select id="f-list"></select></label>
    <label><span class="f-lbl">Verdict</span><select id="f-verdict"></select></label>
    <label><span class="f-lbl">Type</span><select id="f-group"></select></label>
    <label class="f-check"><input id="f-measured" type="checkbox"> Margin measured only</label>
    <span class="f-count" id="f-count" aria-live="polite"></span>
  </section>

  <section class="kpis" id="kpis" aria-label="Headline figures"></section>

  <section class="card" id="card-changes">
    <header class="card-h">
      <div>
        <h2>What changed</h2>
        <p class="card-sub">The last 30 days of the change log for the companies in view: verdicts, margin bands, and the price crossing its estimated value, the Graham Number or two-thirds of net current assets. A change caused by new rules rather than by the company is marked as such.</p>
      </div>
    </header>
    <div class="table-wrap" id="changes"></div>
  </section>

  <section class="card" id="card-value">
    <header class="card-h">
      <div>
        <h2>Where the price sits against estimated value</h2>
        <p class="card-sub">Price as a multiple of each company's median value estimate. Left of 1&times; is a discount; left of &#8532;&times; clears Graham's one-third margin of safety.</p>
      </div>
      <button type="button" class="ghost tv" data-for="value">Table view</button>
    </header>
    <div class="key" id="key-value"></div>
    <div class="chart" id="chart-value"></div>
    <div class="tableview" id="tv-value" hidden></div>
  </section>

  <section class="card" id="card-growth">
    <header class="card-h">
      <div>
        <h2>What the price assumes</h2>
        <p class="card-sub" id="growth-sub"></p>
      </div>
      <button type="button" class="ghost tv" data-for="growth">Table view</button>
    </header>
    <div class="key" id="key-growth"></div>
    <div class="chart" id="chart-growth"></div>
    <div class="tableview" id="tv-growth" hidden></div>
  </section>

  <section class="card" id="card-score">
    <header class="card-h">
      <div>
        <h2>Scorecard</h2>
        <p class="card-sub">Select a company for every test, every model and its earnings record. Sorting by a column header is available throughout.</p>
      </div>
    </header>
    <div class="table-wrap"><table class="score" id="score"></table></div>
  </section>

  <section class="card method" id="method"></section>

  <footer class="foot">Research tooling. Every figure here applies a stated rule to public filings; none of it is investment advice, and an estimate of value is an opinion with arithmetic attached. Every test, model and past run is in the workbook: <a href="__WORKBOOK__" download>valuation_tracker.xlsx</a>.</footer>
</div>
<div id="tip" role="tooltip" hidden></div>
<script type="application/json" id="data">/*__DATA__*/</script>
<script>/*__JS__*/</script>
</body>
</html>
"""


_CSS = r"""
:root {
  color-scheme: light;
  --page: #f9f9f7;
  --surface: #fcfcfb;
  --ink-1: #0b0b0b;
  --ink-2: #52514e;
  --ink-3: #898781;
  --grid: #e1e0d9;
  --axis: #c3c2b7;
  --hair: rgba(11,11,11,0.10);
  --hover: rgba(11,11,11,0.04);
  --accent: #2a78d6;
  --neg: #e34948;
  --band: #b7d3f6;
  --zone: rgba(42,120,214,0.07);
  --hist: #86b6ef;
  --impl: #1c5cab;
  --track: #cde2fb;
  --good: #0ca30c;
  --warning: #fab219;
  --serious: #ec835a;
  --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19;
    --ink-1: #f0efec; --ink-2: #c3c2b7; --ink-3: #898781;
    --grid: #2c2c2a; --axis: #383835;
    --hair: rgba(255,255,255,0.10); --hover: rgba(255,255,255,0.05);
    --accent: #3987e5; --neg: #e66767;
    --band: #184f95; --zone: rgba(57,135,229,0.10);
    --hist: #256abf; --impl: #9ec5f4; --track: #104281;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19;
  --ink-1: #f0efec; --ink-2: #c3c2b7; --ink-3: #898781;
  --grid: #2c2c2a; --axis: #383835;
  --hair: rgba(255,255,255,0.10); --hover: rgba(255,255,255,0.05);
  --accent: #3987e5; --neg: #e66767;
  --band: #184f95; --zone: rgba(57,135,229,0.10);
  --hist: #256abf; --impl: #9ec5f4; --track: #104281;
}

* { box-sizing: border-box; }
html, body { margin: 0; }
body {
  background: var(--page); color: var(--ink-1);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
  -webkit-font-smoothing: antialiased;
}
.page { max-width: 1240px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; font-weight: 650; margin: 0; letter-spacing: -0.01em; }
h2 { font-size: 15px; font-weight: 600; margin: 0; }
h3 { font-size: 13px; font-weight: 600; margin: 0 0 6px; }
.sub, .card-sub { color: var(--ink-2); margin: 4px 0 0; }
.card-sub { font-size: 13px; max-width: 72ch; }
.asof { color: var(--ink-3); font-size: 12px; margin: 8px 0 16px; }
.vh { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); }

.top { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
button.ghost {
  font: inherit; font-size: 12px; color: var(--ink-2); background: transparent;
  border: 1px solid var(--hair); border-radius: 6px; padding: 5px 10px; cursor: pointer;
}
button.ghost:hover { background: var(--hover); color: var(--ink-1); }
button:focus-visible, input:focus-visible, select:focus-visible, [tabindex]:focus-visible,
tr.srow:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

.filters { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; margin-bottom: 16px; }
.filters label { display: inline-flex; align-items: center; gap: 6px; color: var(--ink-2); font-size: 13px; }
.f-lbl { color: var(--ink-3); font-size: 12px; }
.filters input[type=search], .filters select {
  font: inherit; font-size: 13px; color: var(--ink-1); background: var(--surface);
  border: 1px solid var(--hair); border-radius: 6px; padding: 5px 8px;
}
.filters input[type=search] { width: 200px; }
.f-count { color: var(--ink-3); font-size: 12px; margin-left: auto; }

.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 16px; }
.kpi { background: var(--surface); border: 1px solid var(--hair); border-radius: 10px; padding: 14px 16px; }
.kpi .k-l { color: var(--ink-2); font-size: 12px; }
.kpi .k-v { font-size: 26px; font-weight: 600; margin-top: 2px; letter-spacing: -0.01em; }
.kpi .k-s { color: var(--ink-3); font-size: 12px; margin-top: 2px; }

.card { background: var(--surface); border: 1px solid var(--hair); border-radius: 12px; padding: 18px 18px 14px; margin-bottom: 16px; }
.card-h { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 10px; }
.key { display: flex; flex-wrap: wrap; gap: 6px 16px; color: var(--ink-2); font-size: 12px; margin: 2px 0 6px; }
.key span { display: inline-flex; align-items: center; gap: 6px; }
.key svg { flex: none; }
.chart { width: 100%; }
.chart svg { display: block; overflow: visible; }
.empty { color: var(--ink-3); font-size: 13px; padding: 12px 0; }

/* svg marks */
.gridline { stroke: var(--grid); stroke-width: 1; shape-rendering: crispEdges; }
.axisline { stroke: var(--axis); stroke-width: 1; shape-rendering: crispEdges; }
.refline { stroke: var(--axis); stroke-width: 1.5; }
.tick { fill: var(--ink-3); font-size: 11px; font-variant-numeric: tabular-nums; }
.reflbl { fill: var(--ink-2); font-size: 11px; }
.zone { fill: var(--zone); }
.lbl-t { fill: var(--ink-1); font-size: 12px; font-weight: 600; }
.lbl-n { fill: var(--ink-3); font-size: 12px; }
.val { fill: var(--ink-2); font-size: 12px; font-variant-numeric: tabular-nums; }
.band { fill: var(--band); }
.dot { fill: var(--accent); stroke: var(--surface); stroke-width: 2; }
.dot-open { fill: var(--surface); stroke: var(--ink-3); stroke-width: 2; }
.off { fill: var(--ink-3); font-size: 13px; }
.conn { stroke: var(--axis); stroke-width: 2; stroke-linecap: round; }
.d-hist { fill: var(--hist); stroke: var(--surface); stroke-width: 2; }
.d-impl { fill: var(--impl); stroke: var(--surface); stroke-width: 2; }
.col-pos { fill: var(--accent); }
.col-neg { fill: var(--neg); }
.line { fill: none; stroke: var(--accent); stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
.hit { fill: transparent; }
g.vrow { cursor: pointer; outline: none; }
g.vrow:hover .hit, g.vrow:focus .hit { fill: var(--hover); }
g.vrow:focus-visible .hit { stroke: var(--accent); stroke-width: 1.5; }
g.colg { cursor: default; outline: none; }
g.colg:hover path, g.colg:focus path { opacity: 0.82; }

/* tooltip */
#tip {
  position: fixed; z-index: 20; pointer-events: none; max-width: 300px;
  background: var(--surface); color: var(--ink-1); border: 1px solid var(--hair);
  border-radius: 8px; padding: 8px 10px; font-size: 12px;
  box-shadow: 0 6px 24px rgba(0,0,0,0.12);
}
.tip-row { display: flex; align-items: baseline; gap: 8px; padding: 1px 0; }
.tip-head { font-weight: 600; margin-bottom: 3px; }
.tip-v { font-weight: 600; font-variant-numeric: tabular-nums; }
.tip-l { color: var(--ink-2); }
.tip-key { width: 12px; height: 2px; border-radius: 1px; align-self: center; flex: none; }
.tip-note { color: var(--ink-3); margin-top: 4px; }

/* status glyph + label */
.st { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.st svg { flex: none; }
.st-good { fill: var(--good); }
.st-warning { fill: var(--warning); }
.st-serious { fill: var(--serious); }
.st-critical { fill: none; stroke: var(--critical); stroke-width: 2.2; stroke-linecap: round; }
.st-neutral { fill: var(--ink-3); }
.st-none { fill: none; stroke: var(--ink-3); stroke-width: 1.5; }

/* tables */
.table-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; font-size: 12px; white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
table.score th[data-k] { cursor: pointer; user-select: none; }
table.score th[data-k]:hover { color: var(--ink-1); }
table.score th .arrow { color: var(--ink-3); font-size: 10px; margin-left: 3px; }
tr.srow { cursor: pointer; }
tr.srow:hover td { background: var(--hover); }
tr.srow.open td { background: var(--hover); border-bottom-color: transparent; }
.tk { font-weight: 600; }
.nm { color: var(--ink-3); font-size: 12px; max-width: 220px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.nw { white-space: nowrap; }
.muted { color: var(--ink-3); }
.small { font-size: 12px; }
.tally { display: inline-flex; flex-direction: column; gap: 4px; min-width: 64px; }
.tally .tv-t { font-variant-numeric: tabular-nums; }
.meter { width: 64px; height: 4px; border-radius: 2px; background: var(--track); overflow: hidden; }
.meter > i { display: block; height: 100%; background: var(--accent); border-radius: 2px; }
.tableview table { margin-top: 6px; }

/* detail */
tr.drow > td { padding: 0; background: var(--page); }
.detail { padding: 16px 14px 18px; display: grid; gap: 18px; }
.d-head { display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px; }
.d-head .d-name { font-size: 15px; font-weight: 600; }
.d-meta { color: var(--ink-3); font-size: 12px; }
.notice { border: 1px solid var(--hair); border-left: 3px solid var(--warning); border-radius: 6px; padding: 8px 10px; background: var(--surface); font-size: 12px; color: var(--ink-2); }
.notice b { color: var(--ink-1); font-weight: 600; }
.d-grid { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr); gap: 18px; }
.panel { background: var(--surface); border: 1px solid var(--hair); border-radius: 10px; padding: 12px 14px; }
.panel p { margin: 6px 0 0; font-size: 12px; color: var(--ink-2); }
.panel p b { color: var(--ink-1); font-weight: 600; }
table.crit td, table.models td { font-size: 12px; }
table.crit td.why, table.models td.why { color: var(--ink-2); min-width: 220px; }
.tier { color: var(--ink-3); font-size: 11px; white-space: nowrap; border: 1px solid var(--hair); border-radius: 4px; padding: 1px 5px; }
.role { color: var(--ink-3); font-size: 11px; white-space: nowrap; }
tr.na td { color: var(--ink-3); }
tr.na td.why { color: var(--ink-3); }

.method h2 { margin-bottom: 8px; }
.method ol { margin: 6px 0 14px; padding-left: 20px; color: var(--ink-2); }
.method li { margin: 4px 0; }
.method li b { color: var(--ink-1); font-weight: 600; }
.foot { color: var(--ink-3); font-size: 12px; margin-top: 18px; max-width: 80ch; }
.foot a { color: var(--accent); }
.notice.stale { margin: -6px 0 16px; border-left-color: var(--critical); }
table.chg td { font-size: 12px; }
table.chg a { color: var(--accent); font-weight: 600; text-decoration: none; }
table.chg a:hover { text-decoration: underline; }
tr.chg-rule td { color: var(--ink-3); }
.chg-more { color: var(--ink-3); font-size: 12px; margin: 8px 0 0; }

@media (max-width: 860px) {
  .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .d-grid { grid-template-columns: minmax(0, 1fr); }
  .f-count { margin-left: 0; width: 100%; }
}
@media (max-width: 520px) {
  .filters input[type=search] { width: 100%; }
  .f-search { width: 100%; }
  .kpi .k-v { font-size: 22px; }
}
@media print {
  .filters, button.ghost, #tip { display: none !important; }
  .card { break-inside: avoid; }
}
"""


_JS = r"""
(function () {
  'use strict';
  var DATA = JSON.parse(document.getElementById('data').textContent);
  var ALL = DATA.companies || [];
  var SVGNS = 'http://www.w3.org/2000/svg';
  var CHART_ROWS = 60;

  // ---------- dom helpers: every string goes in as text, never markup ----------
  function h(tag, attrs) {
    var el = document.createElement(tag);
    setAttrs(el, attrs);
    for (var i = 2; i < arguments.length; i++) append(el, arguments[i]);
    return el;
  }
  function s(tag, attrs) {
    var el = document.createElementNS(SVGNS, tag);
    setAttrs(el, attrs);
    for (var i = 2; i < arguments.length; i++) append(el, arguments[i]);
    return el;
  }
  function setAttrs(el, attrs) {
    if (!attrs) return;
    Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (v === null || v === undefined || v === false) return;
      if (k === 'text') el.textContent = String(v);
      else if (k === 'class') el.setAttribute('class', v);
      else if (k.slice(0, 2) === 'on' && typeof v === 'function') el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? '' : String(v));
    });
  }
  function append(el, kid) {
    if (kid === null || kid === undefined || kid === false) return;
    if (Array.isArray(kid)) { kid.forEach(function (k) { append(el, k); }); return; }
    el.appendChild(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  function $(sel) { return document.querySelector(sel); }
  function isNum(v) { return typeof v === 'number' && isFinite(v); }
  // The model-spread limit as a ratio: dispersion d = (hi - lo) / (hi + lo),
  // so hi / lo = (1 + d) / (1 - d) - 3x at the default 0.5.
  function spreadLimit() {
    var d = (DATA.params || {}).max_dispersion || 0.5;
    return ((1 + d) / (1 - d)).toFixed(0);
  }

  // ---------- formatting ----------
  var nf2 = new Intl.NumberFormat('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  var fmt = {
    money: function (v) {
      if (!isNum(v)) return '—';
      var a = Math.abs(v), sign = v < 0 ? '−' : '';
      return sign + '$' + (a >= 1000 ? Math.round(a).toLocaleString('en-US') : nf2.format(a));
    },
    x: function (v) {
      if (!isNum(v)) return '—';
      return (v >= 10 ? v.toFixed(1) : v.toFixed(2)) + '×';
    },
    pct: function (v, signed) {
      if (!isNum(v)) return '—';
      var t = (Math.abs(v) * 100).toFixed(1) + '%';
      if (v < 0) return '−' + t;
      return (signed && v > 0 ? '+' : '') + t;
    },
    pts: function (v) {
      if (!isNum(v)) return '—';
      var t = (Math.abs(v) * 100).toFixed(1) + ' pts';
      return (v < 0 ? '−' : '+') + t;
    },
    year: function (d) { return d ? String(d).slice(0, 4) : '—'; }
  };

  // ---------- status: colour + shape + label, never colour alone ----------
  // Verdicts, their order and their glyphs come from the engine's own list
  // (score.VERDICTS), so a renamed or added verdict cannot fall out of the
  // filters or the counts here.
  var VERDICT = {};
  (DATA.verdicts || []).forEach(function (v, i) { VERDICT[v.label] = { st: v.status, order: i, candidate: !!v.candidate }; });
  var GROUP_LABEL = { operating: 'Operating company', bank: 'Bank', insurance: 'Insurer', reit: 'REIT', utility: 'Utility', financial: 'Other financial' };
  var CITE = {};
  (DATA.provenance || []).forEach(function (p) { CITE[p.key] = p.citation; });
  function glyph(st, size) {
    size = size || 10;
    var g = s('svg', { width: size, height: size, viewBox: '0 0 10 10', 'aria-hidden': 'true' });
    if (st === 'good') g.appendChild(s('circle', { cx: 5, cy: 5, r: 4.5, class: 'st-good' }));
    else if (st === 'warning') g.appendChild(s('path', { d: 'M5 0.6 L9.6 9.2 L0.4 9.2 Z', class: 'st-warning' }));
    else if (st === 'serious') g.appendChild(s('rect', { x: 0.8, y: 0.8, width: 8.4, height: 8.4, rx: 1.5, class: 'st-serious' }));
    else if (st === 'critical') g.appendChild(s('path', { d: 'M1 1 L9 9 M9 1 L1 9', class: 'st-critical' }));
    else if (st === 'neutral') g.appendChild(s('rect', { x: 1.5, y: 1.5, width: 7, height: 7, rx: 1, class: 'st-neutral' }));
    else g.appendChild(s('circle', { cx: 5, cy: 5, r: 3.8, class: 'st-none' }));
    return g;
  }
  function verdictBadge(v) {
    var meta = VERDICT[v] || { st: 'none' };
    return h('span', { class: 'st' }, glyph(meta.st), h('span', { text: v || '—' }));
  }
  function resultBadge(v, structural) {
    if (v === 'PASS') return h('span', { class: 'st' }, glyph('good'), h('span', { text: 'Pass' }));
    if (v === 'FAIL') return h('span', { class: 'st' }, glyph('critical'), h('span', { text: 'Fail' }));
    return h('span', { class: 'st' }, glyph('none'),
      h('span', { text: structural ? 'Does not apply' : 'Not measurable' }));
  }
  function bandBadge(c) {
    if (c.mos_verdict === 'MEASURED') {
      var st = { STRONG: 'good', ADEQUATE: 'good', THIN: 'warning', PREMIUM: 'serious' }[c.mos_band] || 'neutral';
      var txt = c.mos_band === 'PREMIUM' ? 'Premium' : (c.mos_band || '').charAt(0) + (c.mos_band || '').slice(1).toLowerCase();
      return h('span', { class: 'st' }, glyph(st), h('span', { text: txt }));
    }
    if (c.mos_verdict === 'UNRELIABLE') return h('span', { class: 'st' }, glyph('none'), h('span', { text: 'Withheld' }));
    return h('span', { class: 'st' }, glyph('none'), h('span', { text: 'Unknown' }));
  }
  var TIER_SHORT = {
    'graham': 'Graham', 'graham-derived': 'Graham, derived', 'graham-disavowed': 'Graham, disavowed',
    'buffett': 'Buffett', 'buffett-paraphrase': 'Buffett, paraphrased', 'third-party': 'Third-party', 'derived': 'Proxy'
  };

  // ---------- tooltip ----------
  var tip = document.getElementById('tip');
  function showTip(evt, anchor, head, rows, note) {
    tip.textContent = '';
    tip.appendChild(h('div', { class: 'tip-row tip-head', text: head }));
    rows.forEach(function (r) {
      var row = h('div', { class: 'tip-row' });
      if (r.key) { var k = h('span', { class: 'tip-key' }); k.style.background = r.key; row.appendChild(k); }
      row.appendChild(h('span', { class: 'tip-v', text: r.v }));
      if (r.l) row.appendChild(h('span', { class: 'tip-l', text: r.l }));
      tip.appendChild(row);
    });
    if (note) tip.appendChild(h('div', { class: 'tip-note', text: note }));
    tip.hidden = false;
    var x, y;
    if (evt && isNum(evt.clientX) && evt.type && evt.type.indexOf('pointer') === 0) { x = evt.clientX; y = evt.clientY; }
    else { var r = anchor.getBoundingClientRect(); x = r.left + Math.min(r.width / 2, 160); y = r.top + r.height / 2; }
    var tw = tip.offsetWidth, th = tip.offsetHeight, pad = 14;
    var left = x + pad, top = y + pad;
    if (left + tw > window.innerWidth - 8) left = x - tw - pad;
    if (top + th > window.innerHeight - 8) top = y - th - pad;
    tip.style.left = Math.max(8, left) + 'px';
    tip.style.top = Math.max(8, top) + 'px';
  }
  function hideTip() { tip.hidden = true; }
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  // Tooltip on hover and focus; `open` (optional) on click and Enter/Space.
  function wire(el, over, open) {
    el.addEventListener('pointerenter', over); el.addEventListener('pointermove', over);
    el.addEventListener('pointerleave', hideTip); el.addEventListener('focus', over); el.addEventListener('blur', hideTip);
    if (!open) return;
    el.addEventListener('click', open);
    el.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
  }

  // ---------- state ----------
  var state = { q: '', list: '', verdict: '', group: '', measured: false, sortKey: 'p2v', sortDir: 1, open: null };
  function filtered() {
    var q = state.q.trim().toLowerCase();
    return ALL.filter(function (c) {
      if (q && (c.t || '').toLowerCase().indexOf(q) < 0 && (c.name || '').toLowerCase().indexOf(q) < 0) return false;
      if (state.list && (c.lists || []).indexOf(state.list) < 0) return false;
      if (state.verdict && c.verdict !== state.verdict) return false;
      if (state.group && c.group !== state.group) return false;
      if (state.measured && c.mos_verdict !== 'MEASURED') return false;
      return true;
    });
  }

  // ---------- header ----------
  function renderAsOf() {
    var m = DATA.macro || {}, p = DATA.params || {};
    var parts = ['Generated ' + DATA.generated];
    // Dated, so a series that could not be refreshed shows its age; missing,
    // so it says which tests went unmeasured rather than vanishing quietly.
    parts.push(isNum(m.aaa) ? 'AAA corporate yield ' + m.aaa.toFixed(2) + '%' + (m.aaa_asof ? ' (' + m.aaa_asof + ')' : '')
      : 'AAA yield unavailable, so Graham’s yield tests are not measured');
    if (isNum(m.treasury_10y)) parts.push('10-year Treasury ' + m.treasury_10y.toFixed(2) + '%');
    parts.push(isNum(m.cpi_ratio) ? 'CPI ×' + m.cpi_ratio.toFixed(1) + ' since ' + m.cpi_base_year + (m.cpi_asof ? ' (' + String(m.cpi_asof).slice(0, 7) + ')' : '')
      : 'CPI unavailable, so Graham’s size test is not measured');
    if (isNum(p.discount_rate)) parts.push('required return ' + (p.discount_rate * 100).toFixed(0) + '% before debt');
    parts.push('config ' + DATA.config_version);
    $('#asof').textContent = parts.join(' · ');
    $('#growth-sub').textContent =
      'The reverse DCF: the growth rate today’s price needs for ' + (p.stage1_years || 10) +
      ' years at each company’s equity rate (' + ((p.discount_rate || 0.09) * 100).toFixed(0) + '% required of the business, more where it carries debt), against the growth the record supports ' +
      '(the lower of owner-earnings and revenue growth, never crediting growth that came from buybacks, capped at ' + Math.round((p.max_stage1_growth || 0.15) * 100) + '%). Sorted by the gap. It needs no growth forecast, but it does move with the discount rate: each row’s tooltip gives the rate one point either side.';
  }

  // A page that stopped updating keeps showing its numbers as if current;
  // past four days - a weekend and then some - it says so.
  function renderStale() {
    var m = String(DATA.generated || '').match(/^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})/);
    if (!m) return;
    var days = (Date.now() - new Date(m[1] + 'T' + m[2] + ':00Z').getTime()) / 86400000;
    if (!(days >= 4)) return;
    var el = $('#stale');
    el.textContent = '';
    el.appendChild(h('b', { text: 'Out of date: ' }));
    el.appendChild(document.createTextNode('these figures were generated ' + Math.floor(days) + ' days ago, on ' + m[1] +
      '. The tracker normally runs every weekday, so its scheduled run has stopped or is failing.'));
    el.hidden = false;
  }

  function setupFilters() {
    var verdicts = Object.keys(VERDICT).filter(function (v) { return ALL.some(function (c) { return c.verdict === v; }); });
    var vs = $('#f-verdict');
    vs.appendChild(h('option', { value: '', text: 'All' }));
    verdicts.forEach(function (v) { vs.appendChild(h('option', { value: v, text: v.charAt(0) + v.slice(1).toLowerCase() })); });
    var groups = Array.from(new Set(ALL.map(function (c) { return c.group; }).filter(Boolean))).sort();
    var gs = $('#f-group');
    gs.appendChild(h('option', { value: '', text: 'All' }));
    groups.forEach(function (g) { gs.appendChild(h('option', { value: g, text: GROUP_LABEL[g] || g })); });
    // Lists the companies came from (the watchlist, an index); shown only
    // when there is more than one to choose between.
    var lists = Array.from(new Set([].concat.apply([], ALL.map(function (c) { return c.lists || []; }))))
      .sort(function (a, b) { return (a === 'Watchlist' ? -1 : b === 'Watchlist' ? 1 : a.localeCompare(b)); });
    var ls = $('#f-list');
    ls.appendChild(h('option', { value: '', text: 'All' }));
    lists.forEach(function (l) { ls.appendChild(h('option', { value: l, text: l })); });
    $('#f-list-wrap').hidden = lists.length < 2;
    ls.addEventListener('change', function (e) { state.list = e.target.value; renderAll(); });
    $('#f-q').addEventListener('input', function (e) { state.q = e.target.value; renderAll(); });
    vs.addEventListener('change', function (e) { state.verdict = e.target.value; renderAll(); });
    gs.addEventListener('change', function (e) { state.group = e.target.value; renderAll(); });
    $('#f-measured').addEventListener('change', function (e) { state.measured = e.target.checked; renderAll(); });
  }

  function renderKpis(rows) {
    var host = $('#kpis'); host.textContent = '';
    var cand = rows.filter(function (c) { return (VERDICT[c.verdict] || {}).candidate; }).length;
    var measured = rows.filter(function (c) { return c.mos_verdict === 'MEASURED'; });
    // Measured margins only: a withheld margin's price/value is the number
    // the withholding exists to keep out of rankings and counts.
    var p2v = measured.map(function (c) { return c.p2v; }).filter(isNum).sort(function (a, b) { return a - b; });
    var med = p2v.length ? (p2v.length % 2 ? p2v[(p2v.length - 1) / 2] : (p2v[p2v.length / 2 - 1] + p2v[p2v.length / 2]) / 2) : null;
    var below = p2v.filter(function (v) { return v < 1; }).length;
    var tiles = [
      { l: 'Companies shown', v: String(rows.length), s: 'of ' + ALL.length + ' evaluated' },
      { l: 'Candidates', v: String(cand), s: 'quality passes and a measured margin of safety' },
      { l: 'Margin of safety measurable', v: String(measured.length), s: 'value models within ' + spreadLimit() + '× of each other' },
      { l: 'Median price to value', v: med === null ? '—' : fmt.x(med), s: below + ' of ' + p2v.length + ' measured are below their estimate' }
    ];
    tiles.forEach(function (t) {
      host.appendChild(h('div', { class: 'kpi' },
        h('div', { class: 'k-l', text: t.l }), h('div', { class: 'k-v', text: t.v }), h('div', { class: 'k-s', text: t.s })));
    });
    $('#f-count').textContent = rows.length === ALL.length ? '' : rows.length + ' of ' + ALL.length + ' shown';
  }

  // ---------- chart: price against estimated value ----------
  function keyItem(svgEl, label) { return h('span', null, svgEl, h('span', { text: label })); }
  function renderValueKey() {
    var host = $('#key-value'); host.textContent = '';
    var b = s('svg', { width: 22, height: 10 }); b.appendChild(s('rect', { x: 0, y: 1, width: 22, height: 8, rx: 4, class: 'band' }));
    var d = s('svg', { width: 12, height: 12 }); d.appendChild(s('circle', { cx: 6, cy: 6, r: 4.5, class: 'dot' }));
    var o = s('svg', { width: 12, height: 12 }); o.appendChild(s('circle', { cx: 6, cy: 6, r: 4, class: 'dot-open' }));
    var l = s('svg', { width: 4, height: 14 }); l.appendChild(s('line', { x1: 2, x2: 2, y1: 0, y2: 14, class: 'refline' }));
    host.appendChild(keyItem(b, 'range of the value models'));
    host.appendChild(keyItem(l, 'median estimate (1×)'));
    host.appendChild(keyItem(d, 'price, margin measured'));
    host.appendChild(keyItem(o, 'price, models disagree so margin withheld'));
  }

  function renderValueChart(rows) {
    var host = $('#chart-value'); host.textContent = '';
    var items = rows.filter(function (c) { return isNum(c.p2v) && isNum(c.value); })
      .sort(function (a, b) { return a.p2v - b.p2v; });
    var skipped = rows.length - items.length;
    if (!items.length) { host.appendChild(h('p', { class: 'empty', text: 'No company in this view has both a price and a value estimate.' })); renderValueTable(items); return; }
    // One row per company stops being readable long before 500 companies.
    var all = items;
    items = items.slice(0, CHART_ROWS);

    var W = Math.max(320, host.clientWidth || 800);
    var narrow = W < 600;
    var labelW = narrow ? 58 : 190, valW = 56, rowH = 30, top = 30, bottom = 30;
    var L = labelW + 10, R = W - valW - 10;
    var H = top + items.length * rowH + bottom;
    var lo = 0.25, hi = 4;
    items.forEach(function (c) {
      [c.p2v, c.low_rel, c.high_rel].forEach(function (v) { if (isNum(v) && v > 0) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });
    });
    lo = Math.max(lo * 0.9, 0.1); hi = Math.min(hi * 1.1, 12);
    var LL = Math.log(lo), LH = Math.log(hi);
    function X(v) { var c = Math.min(Math.max(v, lo), hi); return L + (Math.log(c) - LL) / (LH - LL) * (R - L); }

    var svg = s('svg', { width: W, height: H, viewBox: '0 0 ' + W + ' ' + H, role: 'group',
      'aria-label': 'Price as a multiple of estimated value for ' + items.length + ' companies' });
    // margin-of-safety zone and grid
    // Graham's margin zone, from the same threshold the scorer uses.
    var m23 = 1 - ((DATA.params || {}).mos_strong || 1 / 3);
    if (lo < m23) svg.appendChild(s('rect', { class: 'zone', x: L, y: top - 6, width: X(m23) - L, height: H - top - bottom + 6 }));
    var CAND = [0.1, 0.2, 0.25, 0.33, 0.5, 1, 1.5, 2, 3, 4, 5, 7, 10];
    var lastPx = -1e9;
    CAND.filter(function (t) { return t >= lo && t <= hi; }).forEach(function (t) {
      var px = X(t);
      if (px - lastPx < 40) return;
      lastPx = px;
      if (Math.abs(px - X(1)) > 1 && Math.abs(px - X(m23)) > 18)
        svg.appendChild(s('line', { class: 'gridline', x1: px, x2: px, y1: top - 6, y2: H - bottom }));
      svg.appendChild(s('text', { class: 'tick', x: px, y: H - bottom + 16, 'text-anchor': 'middle', text: String(t) + '×' }));
    });
    svg.appendChild(s('line', { class: 'axisline', x1: L, x2: R, y1: H - bottom, y2: H - bottom }));
    // reference lines
    svg.appendChild(s('line', { class: 'refline', x1: X(1), x2: X(1), y1: top - 10, y2: H - bottom }));
    svg.appendChild(s('text', { class: 'reflbl', x: X(1), y: top - 16, 'text-anchor': 'middle', text: narrow ? '1×' : 'estimated value' }));
    if (lo < m23) {
      svg.appendChild(s('line', { class: 'axisline', x1: X(m23), x2: X(m23), y1: top - 6, y2: H - bottom }));
      if (X(1) - X(m23) > (narrow ? 26 : 110))
        svg.appendChild(s('text', { class: 'reflbl', x: X(m23) - 4, y: top - 16, 'text-anchor': 'end', text: narrow ? '⅔' : '⅔ · Graham’s margin' }));
    }

    items.forEach(function (c, i) {
      var yc = top + i * rowH + rowH / 2;
      var measured = c.mos_verdict === 'MEASURED';
      var g = s('g', { class: 'vrow', tabindex: 0, role: 'button',
        'aria-label': c.t + ', price ' + fmt.x(c.p2v) + ' its estimated value, ' + (measured ? 'margin measured' : 'margin withheld') + '. Open detail.' });
      g.appendChild(s('rect', { class: 'hit', x: 0, y: yc - rowH / 2, width: W, height: rowH, rx: 4 }));
      g.appendChild(s('text', { class: 'lbl-t', x: 2, y: yc + 4, text: c.t }));
      if (!narrow) {
        var nm = c.name || '';
        if (nm.length > 22) nm = nm.slice(0, 21) + '…';
        g.appendChild(s('text', { class: 'lbl-n', x: 56, y: yc + 4, text: nm }));
      }
      if (isNum(c.low_rel) && isNum(c.high_rel)) {
        var x1 = X(c.low_rel), x2 = X(c.high_rel);
        g.appendChild(s('rect', { class: 'band', x: x1, y: yc - 4, width: Math.max(4, x2 - x1), height: 8, rx: 4 }));
      }
      var px = X(c.p2v);
      g.appendChild(s('circle', { class: measured ? 'dot' : 'dot-open', cx: px, cy: yc, r: measured ? 5 : 4.5 }));
      if (c.p2v > hi || c.p2v < lo) g.appendChild(s('text', { class: 'off', x: c.p2v > hi ? px + 7 : px - 13, y: yc + 4, text: c.p2v > hi ? '›' : '‹' }));
      g.appendChild(s('text', { class: 'val', x: W - 2, y: yc + 4, 'text-anchor': 'end', text: fmt.x(c.p2v) }));
      function over(e) {
        showTip(e, g, c.t + ' · ' + (c.name || ''), [
          { v: fmt.money(c.price), l: 'price' },
          { v: fmt.money(c.value), l: 'median estimate' + (isNum(c.n_models) ? ' (' + c.n_models + ' model' + (c.n_models === 1 ? '' : 's') + ')' : '') },
          { v: fmt.money(c.low) + ' – ' + fmt.money(c.high), l: 'model range' },
          { v: fmt.x(c.p2v), l: 'price to value' },
          { v: measured ? fmt.pct(c.mos_measured, true) : 'withheld', l: 'margin of safety' }
        ], measured ? null : (c.n_models < 2 ? 'Only one model applies, so there is no range to measure a margin against.' : 'Models disagree by more than the limit, so no margin is claimed.'));
      }
      wire(g, over, function () { openDetail(c.t); });
      svg.appendChild(g);
    });
    host.appendChild(svg);
    if (all.length > items.length) host.appendChild(h('p', { class: 'empty', text: 'Showing the ' + items.length + ' lowest price to value of ' + all.length + '. The table view and the scorecard have every one.' }));
    if (skipped) host.appendChild(h('p', { class: 'empty', text: skipped + ' compan' + (skipped === 1 ? 'y has' : 'ies have') + ' no value estimate or no price and ' + (skipped === 1 ? 'is' : 'are') + ' listed only in the scorecard.' }));
    renderValueTable(all);
  }

  function simpleTable(cols, rows) {
    var t = h('table');
    var tr = h('tr');
    cols.forEach(function (c) { tr.appendChild(h('th', { class: c.num ? 'num' : null, text: c.l })); });
    t.appendChild(h('thead', null, tr));
    var tb = h('tbody');
    rows.forEach(function (r) {
      var row = h('tr');
      cols.forEach(function (c) { row.appendChild(h('td', { class: c.num ? 'num' : null, text: c.f(r) })); });
      tb.appendChild(row);
    });
    t.appendChild(tb);
    return t;
  }
  function renderValueTable(items) {
    var host = $('#tv-value'); host.textContent = '';
    host.appendChild(simpleTable([
      { l: 'Ticker', f: function (c) { return c.t; } },
      { l: 'Price', num: 1, f: function (c) { return fmt.money(c.price); } },
      { l: 'Low', num: 1, f: function (c) { return fmt.money(c.low); } },
      { l: 'Median estimate', num: 1, f: function (c) { return fmt.money(c.value); } },
      { l: 'High', num: 1, f: function (c) { return fmt.money(c.high); } },
      { l: 'Price / value', num: 1, f: function (c) { return fmt.x(c.p2v); } },
      { l: 'Margin', f: function (c) { return c.mos_verdict === 'MEASURED' ? fmt.pct(c.mos_measured, true) : 'withheld'; } }
    ], items));
  }

  // ---------- chart: what the price assumes ----------
  function renderGrowthKey() {
    var host = $('#key-growth'); host.textContent = '';
    var a = s('svg', { width: 12, height: 12 }); a.appendChild(s('circle', { cx: 6, cy: 6, r: 4.5, class: 'd-hist' }));
    var b = s('svg', { width: 12, height: 12 }); b.appendChild(s('circle', { cx: 6, cy: 6, r: 4.5, class: 'd-impl' }));
    host.appendChild(keyItem(a, 'growth the record supports'));
    host.appendChild(keyItem(b, 'growth the price implies'));
  }
  function niceStep(span, target) {
    var raw = span / target, mag = Math.pow(10, Math.floor(Math.log10(raw))), e = raw / mag;
    return (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1) * mag;
  }
  function renderGrowthChart(rows) {
    var host = $('#chart-growth'); host.textContent = '';
    var items = rows.filter(function (c) { return isNum(c.implied_growth) && isNum(c.credited_growth); })
      .map(function (c) { return { c: c, gap: c.implied_growth - c.credited_growth }; })
      .sort(function (a, b) { return b.gap - a.gap; });
    if (!items.length) { host.appendChild(h('p', { class: 'empty', text: 'No company in this view has both a reverse DCF and a supported growth rate.' })); renderGrowthTable(items); return; }
    var all = items;
    items = items.slice(0, CHART_ROWS);
    var W = Math.max(320, host.clientWidth || 800);
    var narrow = W < 600;
    var labelW = narrow ? 58 : 120, valW = 74, rowH = 28, top = 14, bottom = 30;
    var L = labelW + 10, R = W - valW - 10, H = top + items.length * rowH + bottom;
    var lo = -0.05, hi = 0.2, solver = (DATA.params || {}).solver || [-0.6, 1.0];
    items.forEach(function (it) { lo = Math.min(lo, it.c.implied_growth, it.c.credited_growth); hi = Math.max(hi, it.c.implied_growth, it.c.credited_growth); });
    lo = Math.max(lo, solver[0]); hi = Math.min(hi, solver[1]);
    var step = niceStep(hi - lo, narrow ? 4 : 7);
    lo = Math.floor(lo / step) * step; hi = Math.ceil(hi / step) * step;
    function X(v) { var c = Math.min(Math.max(v, lo), hi); return L + (c - lo) / (hi - lo) * (R - L); }
    var svg = s('svg', { width: W, height: H, viewBox: '0 0 ' + W + ' ' + H, role: 'group',
      'aria-label': 'Implied versus supported growth for ' + items.length + ' companies' });
    for (var t = lo; t <= hi + step / 2; t += step) {
      var tv = Math.round(t / step) * step, px = X(tv);
      if (Math.abs(tv) > 1e-9) svg.appendChild(s('line', { class: 'gridline', x1: px, x2: px, y1: top - 4, y2: H - bottom }));
      svg.appendChild(s('text', { class: 'tick', x: px, y: H - bottom + 16, 'text-anchor': 'middle', text: Math.round(tv * 100) + '%' }));
    }
    svg.appendChild(s('line', { class: 'refline', x1: X(0), x2: X(0), y1: top - 4, y2: H - bottom }));
    svg.appendChild(s('line', { class: 'axisline', x1: L, x2: R, y1: H - bottom, y2: H - bottom }));
    var hist = cssVar('--hist'), impl = cssVar('--impl');
    items.forEach(function (it, i) {
      var c = it.c, yc = top + i * rowH + rowH / 2;
      var g = s('g', { class: 'vrow', tabindex: 0, role: 'button',
        'aria-label': c.t + ': price implies ' + fmt.pct(c.implied_growth) + ' a year against ' + fmt.pct(c.credited_growth) + ' the record supports. Open detail.' });
      g.appendChild(s('rect', { class: 'hit', x: 0, y: yc - rowH / 2, width: W, height: rowH, rx: 4 }));
      g.appendChild(s('text', { class: 'lbl-t', x: 2, y: yc + 4, text: c.t }));
      var xa = X(c.credited_growth), xb = X(c.implied_growth);
      if (Math.abs(xb - xa) > 1) g.appendChild(s('line', { class: 'conn', x1: xa, x2: xb, y1: yc, y2: yc }));
      g.appendChild(s('circle', { class: 'd-hist', cx: xa, cy: yc, r: 5 }));
      g.appendChild(s('circle', { class: 'd-impl', cx: xb, cy: yc, r: 5 }));
      g.appendChild(s('text', { class: 'val', x: W - 2, y: yc + 4, 'text-anchor': 'end', text: fmt.pts(it.gap) }));
      function over(e) {
        var r = isNum(c.equity_rate) ? c.equity_rate : ((DATA.params || {}).discount_rate || 0.09);
        showTip(e, g, c.t + ' · ' + (c.name || ''), [
          { v: (c.implied_at_bound ? (c.implied_growth > 0 ? '≥ ' : '≤ ') : '') + fmt.pct(c.implied_growth, true), l: 'implied by the price at ' + fmt.pct(r), key: impl },
          c.implied_range ? { v: fmt.pct(c.implied_range[0], true) + ' to ' + fmt.pct(c.implied_range[1], true),
            l: 'implied at ' + fmt.pct(r - 0.01) + ' to ' + fmt.pct(r + 0.01) } : null,
          { v: fmt.pct(c.credited_growth, true), l: 'supported by the record', key: hist },
          { v: fmt.pts(it.gap), l: 'gap' },
          { v: fmt.pct(c.oe_growth, true), l: 'owner earnings, before the cap' },
          { v: fmt.pct(c.rev_growth, true), l: 'revenue' }
        ].filter(Boolean), c.implied_at_bound ? 'At the edge of the solver’s range.' : null);
      }
      wire(g, over, function () { openDetail(c.t); });
      svg.appendChild(g);
    });
    host.appendChild(svg);
    if (all.length > items.length) host.appendChild(h('p', { class: 'empty', text: 'Showing the ' + items.length + ' largest gaps of ' + all.length + '. The table view has every one.' }));
    renderGrowthTable(all);
  }
  function renderGrowthTable(items) {
    var host = $('#tv-growth'); host.textContent = '';
    host.appendChild(simpleTable([
      { l: 'Ticker', f: function (it) { return it.c.t; } },
      { l: 'Implied by price', num: 1, f: function (it) { return fmt.pct(it.c.implied_growth, true); } },
      { l: 'Record supports', num: 1, f: function (it) { return fmt.pct(it.c.credited_growth, true); } },
      { l: 'Gap', num: 1, f: function (it) { return fmt.pts(it.gap); } },
      { l: 'Implied at ±1 pt', num: 1, f: function (it) { return it.c.implied_range ? fmt.pct(it.c.implied_range[0], true) + ' to ' + fmt.pct(it.c.implied_range[1], true) : '—'; } },
      { l: 'Owner-earnings growth', num: 1, f: function (it) { return fmt.pct(it.c.oe_growth, true); } },
      { l: 'Revenue growth', num: 1, f: function (it) { return fmt.pct(it.c.rev_growth, true); } }
    ], items));
  }

  // ---------- scorecard ----------
  function rate(tl) { return tl && tl[1] ? tl[0] / tl[1] : -1; }
  function tallyCell(tl, label) {
    if (!tl || !tl[1]) return h('span', { class: 'muted small', text: 'none measurable' });
    var m = h('div', { class: 'meter' }); var i = h('i'); i.style.width = Math.round(100 * tl[0] / tl[1]) + '%'; m.appendChild(i);
    var extra = tl[2] ? ' (' + tl[2] + ' N/A)' : '';
    return h('span', { class: 'tally', title: tl[0] + ' of ' + tl[1] + ' applicable ' + label + ' tests passed' + extra },
      h('span', { class: 'tv-t', text: tl[0] + ' / ' + tl[1] }), m);
  }
  var COLS = [
    { k: 'company', l: 'Company', v: function (c) { return c.t; } },
    { k: 'verdict', l: 'Verdict', v: function (c) { return (VERDICT[c.verdict] || { order: 9 }).order; } },
    { k: 'price', l: 'Price', num: 1, v: function (c) { return c.price; } },
    { k: 'value', l: 'Estimate', num: 1, v: function (c) { return c.value; } },
    { k: 'p2v', l: 'Price / value', num: 1, v: function (c) { return c.mos_verdict === 'MEASURED' ? c.p2v : null; } },
    { k: 'mos', l: 'Margin', v: function (c) { return c.mos_measured; } },
    { k: 'def', l: 'Graham 7', v: function (c) { return rate(c.tally.def); } },
    { k: 'ent', l: 'Graham ’76', v: function (c) { return rate(c.tally.ent); } },
    { k: 'qual', l: 'Buffett', v: function (c) { return rate(c.tally.qual); } },
    { k: 'implied', l: 'Implied growth', num: 1, v: function (c) { return c.implied_growth; } },
    { k: 'data', l: 'Data', num: 1, v: function (c) { return c.completeness; } }
  ];
  function sortRows(rows) {
    var col = COLS.filter(function (c) { return c.k === state.sortKey; })[0] || COLS[4];
    return rows.slice().sort(function (a, b) {
      var va = col.v(a), vb = col.v(b);
      var na = va === null || va === undefined || (typeof va === 'number' && !isFinite(va));
      var nb = vb === null || vb === undefined || (typeof vb === 'number' && !isFinite(vb));
      if (na && nb) return 0; if (na) return 1; if (nb) return -1;   // missing always last
      if (typeof va === 'string') return state.sortDir * va.localeCompare(vb);
      return state.sortDir * (va - vb);
    });
  }
  function renderScore(rows) {
    var table = $('#score'); table.textContent = '';
    var tr = h('tr');
    tr.appendChild(h('th', { class: 'vh', text: 'Expand' }));
    COLS.forEach(function (c) {
      var active = state.sortKey === c.k;
      var th = h('th', { 'data-k': c.k, class: c.num ? 'num' : null, scope: 'col', tabindex: 0,
        'aria-sort': active ? (state.sortDir > 0 ? 'ascending' : 'descending') : 'none' },
        c.l, active ? h('span', { class: 'arrow', 'aria-hidden': 'true', text: state.sortDir > 0 ? '▲' : '▼' }) : null);
      function go() { if (state.sortKey === c.k) state.sortDir *= -1; else { state.sortKey = c.k; state.sortDir = (c.k === 'company' || c.k === 'verdict' || c.k === 'p2v') ? 1 : -1; } renderScore(filtered()); }
      th.addEventListener('click', go);
      th.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
      tr.appendChild(th);
    });
    table.appendChild(h('thead', null, tr));
    var tb = h('tbody');
    sortRows(rows).forEach(function (c) {
      var open = state.open === c.t;
      var row = h('tr', { class: 'srow' + (open ? ' open' : ''), tabindex: 0, 'data-t': c.t, 'aria-expanded': open ? 'true' : 'false' });
      row.appendChild(h('td', { class: 'muted', 'aria-hidden': 'true', text: open ? '▾' : '▸' }));
      row.appendChild(h('td', null, h('div', { class: 'tk', text: c.t }), h('div', { class: 'nm', text: c.name || '' })));
      row.appendChild(h('td', null, verdictBadge(c.verdict)));
      row.appendChild(h('td', { class: 'num', text: fmt.money(c.price) }));
      row.appendChild(h('td', { class: 'num' }, h('div', { text: fmt.money(c.value) }),
        isNum(c.low) && c.low !== c.high ? h('div', { class: 'muted small nw', text: fmt.money(c.low) + '–' + fmt.money(c.high) }) : null));
      row.appendChild(h('td', { class: 'num', text: fmt.x(c.p2v) }));
      row.appendChild(h('td', null, bandBadge(c),
        c.mos_verdict === 'MEASURED' ? h('div', { class: 'muted small', text: fmt.pct(c.mos_measured, true) }) : null));
      row.appendChild(h('td', null, tallyCell(c.tally.def, 'Graham defensive')));
      row.appendChild(h('td', null, tallyCell(c.tally.ent, 'Graham 1976')));
      row.appendChild(h('td', null, tallyCell(c.tally.qual, 'Buffett quality')));
      row.appendChild(h('td', { class: 'num', text: fmt.pct(c.implied_growth, true) }));
      row.appendChild(h('td', { class: 'num', text: isNum(c.completeness) ? Math.round(c.completeness * 100) + '%' : '—' }));
      function toggle() { state.open = state.open === c.t ? null : c.t; renderScore(filtered()); var r = document.querySelector('tr.srow[data-t="' + CSS.escape(c.t) + '"]'); if (r) r.focus(); }
      row.addEventListener('click', toggle);
      row.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
      tb.appendChild(row);
      if (open) {
        var d = h('tr', { class: 'drow' }); var td = h('td', { colspan: COLS.length + 1 });
        if (needsDetail(c)) {
          td.appendChild(h('p', { class: 'empty', style: 'padding: 12px 14px', text: 'Loading ' + c.t + '…' }));
          loadDetail(c, function () { if (state.open === c.t) renderScore(filtered()); });
        } else {
          td.appendChild(detail(c));
        }
        d.appendChild(td); tb.appendChild(d);
      }
    });
    if (!rows.length) tb.appendChild(h('tr', null, h('td', { colspan: COLS.length + 1, class: 'muted', text: 'No company matches the filters.' })));
    table.appendChild(tb);
    // charts inside an open detail need the laid-out width
    if (state.open) {
      var oc = ALL.filter(function (x) { return x.t === state.open; })[0];
      if (oc && !needsDetail(oc) && !oc.detail_error) { renderEps(document.getElementById('eps-' + oc.t), oc); renderHistory(document.getElementById('hist-' + oc.t), oc); }
    }
  }

  // ---------- details on demand ----------
  // With many companies each one's detail sits in its own script file
  // (DATA.details), which calls VT_DETAIL when it loads. A <script> tag
  // rather than fetch(), so it also works with the page opened from disk.
  var waiting = {};
  window.VT_DETAIL = function (t, d) {
    var c = ALL.filter(function (x) { return x.t === t; })[0];
    if (c) Object.keys(d).forEach(function (k) { c[k] = d[k]; });
    var done = waiting[t]; delete waiting[t];
    if (done) done();
  };
  function needsDetail(c) { return !!(DATA.details && c.file && !c.criteria && !c.detail_error); }
  function loadDetail(c, done) {
    if (waiting[c.t]) { waiting[c.t] = done; return; }
    waiting[c.t] = done;
    var el = document.createElement('script');
    el.src = DATA.details + c.file;
    el.onerror = function () { c.detail_error = true; var f = waiting[c.t]; delete waiting[c.t]; if (f) f(); };
    document.head.appendChild(el);
  }
  // Open a company's detail from a chart, clearing any filter that hides it.
  function openDetail(t) {
    state.open = t;
    var c = ALL.filter(function (x) { return x.t === t; })[0];
    if (c && filtered().indexOf(c) < 0) { state.q = ''; state.list = ''; state.verdict = ''; state.group = ''; state.measured = false; $('#f-q').value = ''; $('#f-list').value = ''; $('#f-verdict').value = ''; $('#f-group').value = ''; $('#f-measured').checked = false; renderAll(); }
    else renderScore(filtered());
    var r = document.querySelector('tr.srow[data-t="' + CSS.escape(t) + '"]');
    if (r) { r.scrollIntoView({ behavior: 'smooth', block: 'start' }); r.focus({ preventScroll: true }); }
  }

  // ---------- detail panel ----------
  function detail(c) {
    var wrap = h('div', { class: 'detail' });
    wrap.appendChild(h('div', { class: 'd-head' },
      h('div', null, h('div', { class: 'd-name', text: c.t + ' · ' + (c.name || '') }),
        h('div', { class: 'd-meta', text: [c.industry, GROUP_LABEL[c.group] || c.group,
          (c.n_years || 0) + ' fiscal years on file, latest ' + fmt.year(c.last_fy),
          'price as of ' + (c.price_date || '—')].filter(Boolean).join(' · ') })),
      verdictBadge(c.verdict)));
    if (c.detail_error) {
      wrap.appendChild(h('div', { class: 'notice' }, h('b', { text: 'Detail unavailable: ' }),
        'the file with this company’s tests and models (' + DATA.details + c.file + ') could not be loaded. It is written beside the page on every run.'));
      return wrap;
    }
    if (c.caveats && c.caveats.length) {
      var n = h('div', { class: 'notice' }, h('b', { text: 'Read with care: ' }), c.caveats.join('; ') + '.');
      wrap.appendChild(n);
    }
    var grid = h('div', { class: 'd-grid' });
    // valuation panel
    var vp = h('div', { class: 'panel' }, h('h3', { text: 'Valuation' }));
    var mt = h('table', { class: 'models' });
    mt.appendChild(h('thead', null, h('tr', null, h('th', { text: 'Model' }), h('th', { text: 'Role' }), h('th', { class: 'num', text: 'Value' }), h('th', { text: 'How' }))));
    var mb = h('tbody');
    var ROLE = { value: 'value estimate', price_limit: 'Graham price limit', reference: 'shown, not relied on', rate: 'implied rate' };
    (c.models || []).forEach(function (m) {
      var v = m.is_rate ? fmt.pct(m.value, true) : fmt.money(m.value);
      var role = (ROLE[m.role] || m.role) + (m.in_consensus ? ' · in median' : '');
      mb.appendChild(h('tr', { class: m.applicable ? null : 'na' },
        h('td', { text: m.label }), h('td', null, h('span', { class: 'role', text: role })),
        h('td', { class: 'num', text: m.applicable ? v : '—' }), h('td', { class: 'why', text: m.note || '' })));
    });
    mt.appendChild(mb);
    vp.appendChild(h('div', { class: 'table-wrap' }, mt));
    if (c.consensus_note) vp.appendChild(h('p', null, h('b', { text: 'Estimate: ' }), c.consensus_note + '.'));
    if (c.mos_note) vp.appendChild(h('p', null, h('b', { text: 'Margin of safety: ' }), c.mos_note + '.'));
    if (c.growth_source) vp.appendChild(h('p', null, h('b', { text: 'DCF growth: ' }), c.growth_source + '.'));
    grid.appendChild(vp);
    // record panel
    var rp = h('div', { class: 'panel' }, h('h3', { text: 'Earnings record' }),
      h('p', { text: 'Diluted EPS by fiscal year, restated onto today’s share basis. Losses sit below the line in red. Graham’s tests 3 and 5 are read off this series.' }));
    rp.appendChild(h('div', { class: 'chart', id: 'eps-' + c.t }));
    rp.appendChild(h('h3', { text: 'Price to value over time', style: 'margin-top:14px' }));
    rp.appendChild(h('div', { class: 'chart', id: 'hist-' + c.t }));
    grid.appendChild(rp);
    wrap.appendChild(grid);
    wrap.appendChild(critPanel('Graham — the defensive investor’s seven tests', 'The Intelligent Investor, chapter 14. Dollar size limits inflated from 1973 by CPI.', c.criteria.defensive, true));
    wrap.appendChild(critPanel('Graham — the 1976 ten-point screen', 'Five tests of reward (R) and five of risk (K). Much stricter than chapter 14 by design.', c.criteria.enterprising, true));
    wrap.appendChild(critPanel('Buffett — quality proxies', 'Buffett’s criteria are mostly qualitative. These are proxies, and the source column says how close each is to his own words.', c.criteria.quality, false));
    return wrap;
  }
  function critPanel(title, sub, rows, numbered) {
    var p = h('div', { class: 'panel' }, h('h3', { text: title }), h('p', { text: sub, style: 'margin:0 0 6px' }));
    var t = h('table', { class: 'crit' });
    var head = h('tr');
    if (numbered) head.appendChild(h('th', { text: '#' }));
    ['Test', 'Result', 'Value', 'Needs', 'Why', 'Source'].forEach(function (x, i) { head.appendChild(h('th', { class: i === 2 ? 'num' : null, text: x })); });
    t.appendChild(h('thead', null, head));
    var tb = h('tbody');
    (rows || []).forEach(function (r) {
      var tr = h('tr', { class: r.verdict === 'N/A' ? 'na' : null });
      if (numbered) tr.appendChild(h('td', { class: 'muted', text: r.number || '' }));
      tr.appendChild(h('td', { text: r.name }));
      tr.appendChild(h('td', null, resultBadge(r.verdict, r.structural)));
      tr.appendChild(h('td', { class: 'num', text: r.display }));
      tr.appendChild(h('td', { class: 'muted', text: r.threshold || '' }));
      tr.appendChild(h('td', { class: 'why', text: r.detail || '' }));
      tr.appendChild(h('td', null, h('span', { class: 'tier', title: CITE[r.key] || '', text: TIER_SHORT[r.source_tier] || r.source_tier || '' })));
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    p.appendChild(h('div', { class: 'table-wrap' }, t));
    return p;
  }

  // column with a 4px rounded data-end and a square baseline end
  function colPath(x, w, yb, ye, r) {
    var up = ye < yb, ht = Math.abs(yb - ye);
    r = Math.min(r, w / 2, ht);
    if (ht < 0.5) return 'M' + x + ',' + yb + 'h' + w;
    if (up) return 'M' + x + ',' + yb + 'V' + (ye + r) + 'Q' + x + ',' + ye + ' ' + (x + r) + ',' + ye + 'H' + (x + w - r) + 'Q' + (x + w) + ',' + ye + ' ' + (x + w) + ',' + (ye + r) + 'V' + yb + 'Z';
    return 'M' + x + ',' + yb + 'V' + (ye - r) + 'Q' + x + ',' + ye + ' ' + (x + r) + ',' + ye + 'H' + (x + w - r) + 'Q' + (x + w) + ',' + ye + ' ' + (x + w) + ',' + (ye - r) + 'V' + yb + 'Z';
  }
  function niceTicks(lo, hi, n) {
    if (lo === hi) hi = lo + 1;
    var st = niceStep(hi - lo, n), a = Math.floor(lo / st) * st, b = Math.ceil(hi / st) * st, out = [];
    for (var v = a; v <= b + st / 2; v += st) out.push(Math.round(v / st) * st);
    return out;
  }
  function renderEps(host, c) {
    if (!host) return;
    host.textContent = '';
    var f = c.fund || {}, pts = [];
    (f.years || []).forEach(function (y, i) { if (isNum((f.eps || [])[i])) pts.push({ y: y, v: f.eps[i] }); });
    if (pts.length < 2) { host.appendChild(h('p', { class: 'empty', text: 'Not enough EPS history to chart.' })); return; }
    var W = Math.max(280, host.clientWidth || 480), H = 190, padL = 44, padR = 6, padT = 18, padB = 24;
    var vmin = Math.min(0, Math.min.apply(null, pts.map(function (p) { return p.v; })));
    var vmax = Math.max(0, Math.max.apply(null, pts.map(function (p) { return p.v; })));
    var ticks = niceTicks(vmin, vmax, 4), t0 = ticks[0], t1 = ticks[ticks.length - 1];
    function Y(v) { return padT + (t1 - v) / (t1 - t0) * (H - padT - padB); }
    var svg = s('svg', { width: W, height: H, viewBox: '0 0 ' + W + ' ' + H, role: 'group', 'aria-label': 'Diluted EPS by fiscal year for ' + c.t });
    ticks.forEach(function (t) {
      if (Math.abs(t) > 1e-12) svg.appendChild(s('line', { class: 'gridline', x1: padL, x2: W - padR, y1: Y(t), y2: Y(t) }));
      svg.appendChild(s('text', { class: 'tick', x: padL - 6, y: Y(t) + 4, 'text-anchor': 'end', text: (t < 0 ? '−' : '') + '$' + Math.abs(t).toFixed(Math.abs(t1 - t0) < 4 ? 2 : (Math.abs(t1 - t0) < 20 ? 1 : 0)) }));
    });
    var band = (W - padL - padR) / pts.length, bw = Math.min(24, band * 0.62);
    var every = Math.max(1, Math.ceil(36 / band));
    var minI = 0, lastI = pts.length - 1;
    pts.forEach(function (p, i) { if (p.v < pts[minI].v) minI = i; });
    pts.forEach(function (p, i) {
      var cx = padL + band * i + band / 2;
      var g = s('g', { class: 'colg', tabindex: 0, 'aria-label': fmt.year(p.y) + ': EPS ' + fmt.money(p.v) });
      g.appendChild(s('rect', { class: 'hit', x: cx - band / 2, y: padT, width: band, height: H - padT - padB }));
      g.appendChild(s('path', { class: p.v >= 0 ? 'col-pos' : 'col-neg', d: colPath(cx - bw / 2, bw, Y(0), Y(p.v), 4) }));
      wire(g, function (e) { showTip(e, g, 'Fiscal ' + fmt.year(p.y), [{ v: fmt.money(p.v), l: 'diluted EPS' }]); });
      svg.appendChild(g);
      if ((pts.length - 1 - i) % every === 0) svg.appendChild(s('text', { class: 'tick', x: cx, y: H - padB + 15, 'text-anchor': 'middle', text: "'" + String(p.y).slice(2, 4) }));
      if (i === lastI || (i === minI && p.v < 0)) {
        var ly = p.v >= 0 ? Y(p.v) - 5 : Y(p.v) + 13;
        svg.appendChild(s('text', { class: 'val', x: cx, y: ly, 'text-anchor': 'middle', text: fmt.money(p.v) }));
      }
    });
    svg.appendChild(s('line', { class: 'axisline', x1: padL, x2: W - padR, y1: Y(0), y2: Y(0) }));
    host.appendChild(svg);
  }
  function renderHistory(host, c) {
    if (!host) return;
    host.textContent = '';
    var pts = (c.history || []).filter(function (p) { return isNum(p.p2v) && p.d; });
    if (pts.length < 2) {
      host.appendChild(h('p', { class: 'empty', text: pts.length ? 'One run on record (' + pts[0].d + ', ' + fmt.x(pts[0].p2v) + '). The line draws once the tracker has a second run.' : 'No price-to-value history yet.' }));
      return;
    }
    var W = Math.max(280, host.clientWidth || 480), H = 150, padL = 44, padR = 44, padT = 14, padB = 24;
    var ts = pts.map(function (p) { return new Date(p.d + 'T00:00:00').getTime(); });
    var vs = pts.map(function (p) { return p.p2v; });
    var ticks = niceTicks(Math.min(0.5, Math.min.apply(null, vs)), Math.max(1.5, Math.max.apply(null, vs)), 4);
    var t0 = ticks[0], t1 = ticks[ticks.length - 1], x0 = ts[0], x1 = ts[ts.length - 1];
    function X(t) { return padL + (x1 === x0 ? 0.5 : (t - x0) / (x1 - x0)) * (W - padL - padR); }
    function Y(v) { return padT + (t1 - v) / (t1 - t0) * (H - padT - padB); }
    var svg = s('svg', { width: W, height: H, viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Price to value across ' + pts.length + ' runs for ' + c.t });
    ticks.forEach(function (t) {
      svg.appendChild(s('line', { class: Math.abs(t - 1) < 1e-9 ? 'refline' : 'gridline', x1: padL, x2: W - padR, y1: Y(t), y2: Y(t) }));
      svg.appendChild(s('text', { class: 'tick', x: padL - 6, y: Y(t) + 4, 'text-anchor': 'end', text: String(+t.toFixed(4)) + '×' }));
    });
    var d = pts.map(function (p, i) { return (i ? 'L' : 'M') + X(ts[i]) + ',' + Y(p.p2v); }).join('');
    svg.appendChild(s('path', { class: 'line', d: d }));
    var li = pts.length - 1;
    svg.appendChild(s('circle', { class: 'dot', cx: X(ts[li]), cy: Y(vs[li]), r: 4 }));
    svg.appendChild(s('text', { class: 'val', x: X(ts[li]) + 8, y: Y(vs[li]) + 4, text: fmt.x(vs[li]) }));
    svg.appendChild(s('text', { class: 'tick', x: padL, y: H - 6, text: pts[0].d }));
    svg.appendChild(s('text', { class: 'tick', x: W - padR, y: H - 6, 'text-anchor': 'end', text: pts[li].d }));
    // crosshair: snap to the nearest run
    var cross = s('line', { class: 'axisline', x1: 0, x2: 0, y1: padT, y2: H - padB, visibility: 'hidden' });
    svg.appendChild(cross);
    var hit = s('rect', { class: 'hit', x: padL, y: padT, width: W - padL - padR, height: H - padT - padB });
    hit.addEventListener('pointermove', function (e) {
      var r = svg.getBoundingClientRect(), mx = (e.clientX - r.left) * (W / r.width), best = 0;
      ts.forEach(function (t, i) { if (Math.abs(X(t) - mx) < Math.abs(X(ts[best]) - mx)) best = i; });
      cross.setAttribute('x1', X(ts[best])); cross.setAttribute('x2', X(ts[best])); cross.setAttribute('visibility', 'visible');
      var p = pts[best];
      showTip(e, hit, p.d, [{ v: fmt.x(p.p2v), l: 'price to value' }, { v: fmt.money(p.p), l: 'price' }, { v: fmt.money(p.v), l: 'estimate' }, { v: p.verdict || '—', l: 'verdict' }]);
    });
    hit.addEventListener('pointerleave', function () { cross.setAttribute('visibility', 'hidden'); hideTip(); });
    svg.appendChild(hit);
    host.appendChild(svg);
  }

  // ---------- method ----------
  function renderMethod() {
    var host = $('#method'); host.textContent = '';
    var p = DATA.params || {};
    host.appendChild(h('h2', { text: 'How to read this' }));
    var ol = h('ol');
    [
      ['Three results, not two. ', 'A test is Pass, Fail, or N/A. N/A means either the data is missing, or the test does not apply to this kind of company — a current ratio for a bank. Neither is ever counted as a failure.'],
      ['The estimate is a median of models that capitalise earnings at the required return. ', 'A DCF on owner earnings, and earnings power value with no growth credited; for banks and insurers, a justified price-to-book and average earnings at the required return. The required return is ' + Math.round(((DATA.params || {}).discount_rate || 0.09) * 100) + '% of the business, and more of the equity of a company with debt. Graham’s ceilings — fifteen times average earnings, the Graham Number, net current assets — are price limits, shown separately, not estimates of worth.'],
      ['A margin of safety has to be earned. ', 'It is reported only when at least two models apply and the highest is within ' + spreadLimit() + '× of the lowest. Otherwise it is withheld and scores zero, the same as a premium: a margin that cannot be demonstrated is not a margin.'],
      ['Read “what the price assumes” first. ', 'The forward models depend on a growth forecast; the reverse DCF does not. It states the growth today’s price needs and sets it beside what the business has actually done, never crediting growth that came from buying back its own shares. It does depend on the discount rate, so the rate one point either side is shown too: a conclusion that holds across that range is a finding, one that flips inside it is not.'],
      ['The composite score only sorts the list. ', 'Neither Graham nor Buffett weighted their tests into one number. The per-test tables in each company’s detail are the analysis.']
    ].forEach(function (x) { ol.appendChild(h('li', null, h('b', { text: x[0] }), x[1])); });
    host.appendChild(ol);
    host.appendChild(h('h3', { text: 'Where each threshold comes from' }));
    var t = h('table', { class: 'crit' });
    t.appendChild(h('thead', null, h('tr', null, h('th', { text: 'Source' }), h('th', { text: 'Meaning' }), h('th', { text: 'Tests' }))));
    var tb = h('tbody');
    var byTier = {};
    (DATA.provenance || []).forEach(function (r) { (byTier[r.tier] = byTier[r.tier] || []).push(r); });
    Object.keys(DATA.tiers || {}).forEach(function (tier) {
      var items = byTier[tier] || [];
      if (!items.length) return;
      tb.appendChild(h('tr', null,
        h('td', null, h('span', { class: 'tier', text: TIER_SHORT[tier] || tier })),
        h('td', { class: 'why', text: DATA.tiers[tier] }),
        h('td', { class: 'why', text: items.map(function (r) { return r.citation; }).join(' · ') })));
    });
    t.appendChild(tb);
    host.appendChild(h('div', { class: 'table-wrap' }, t));
    host.appendChild(h('p', { class: 'card-sub', style: 'margin-top:12px',
      text: 'Fundamentals are the SEC’s XBRL filings, which begin around 2009, so Graham’s twenty-year dividend test is usually unverifiable rather than failed. Per-share history is restated for splits using the date each figure was filed; share counts are used as filed, with scale errors repaired and net income over EPS used only to fill a missing year; prices are split-adjusted but not dividend-adjusted, so historical market values are real ones.' }));
  }

  // ---------- wiring ----------
  // ---------- what changed ----------
  function renderChanges(rows) {
    var host = $('#changes'); host.textContent = '';
    var inView = {};
    rows.forEach(function (c) { inView[c.t] = true; });
    var items = (DATA.changes || []).filter(function (x) { return inView[x.t]; });
    if (!items.length) {
      host.appendChild(h('p', { class: 'empty', text: (DATA.changes || []).length
        ? 'No change in the last 30 days for the companies in view.' : 'No change in the last 30 days.' }));
      return;
    }
    var shown = items.slice(0, 40);
    var t = h('table', { class: 'chg' });
    t.appendChild(h('thead', null, h('tr', null, ['Date', 'Company', 'Change', 'Before', 'After', 'Detail']
      .map(function (x) { return h('th', { text: x }); }))));
    var tb = h('tbody');
    shown.forEach(function (x) {
      var rule = String(x.change || '').indexOf('(rules') >= 0;
      tb.appendChild(h('tr', { class: rule ? 'chg-rule' : null },
        h('td', { class: 'nw', text: x.d || '' }),
        h('td', null, h('a', { href: '#', text: x.t, onclick: function (e) { e.preventDefault(); openDetail(x.t); } })),
        h('td', { text: x.change || '' }), h('td', { text: x.before || '—' }),
        h('td', { text: x.after || '—' }), h('td', { class: 'why', text: x.detail || '' })));
    });
    t.appendChild(tb);
    host.appendChild(t);
    if (items.length > shown.length) host.appendChild(h('p', { class: 'chg-more',
      text: (items.length - shown.length) + ' more in the workbook’s Changes sheet.' }));
  }

  function renderAll() {
    var rows = filtered();
    renderKpis(rows);
    renderChanges(rows);
    renderValueChart(rows);
    renderGrowthChart(rows);
    renderScore(rows);
  }
  function setupTheme() {
    var root = document.documentElement;
    try { var saved = localStorage.getItem('vt-theme'); if (saved === 'dark' || saved === 'light') root.setAttribute('data-theme', saved); } catch (e) {}
    $('#theme').addEventListener('click', function () {
      var cur = root.getAttribute('data-theme') || (window.matchMedia && matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      var next = cur === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('vt-theme', next); } catch (e) {}
      renderAll();   // tooltip keys read the live palette
    });
  }
  function setupTableViews() {
    Array.prototype.forEach.call(document.querySelectorAll('button.tv'), function (b) {
      b.addEventListener('click', function () {
        var which = b.getAttribute('data-for');
        var tv = document.getElementById('tv-' + which), ch = document.getElementById('chart-' + which), key = document.getElementById('key-' + which);
        var showTable = tv.hidden;
        tv.hidden = !showTable; ch.hidden = showTable; if (key) key.hidden = showTable;
        b.textContent = showTable ? 'Chart view' : 'Table view';
      });
    });
  }
  var resizeT = null;
  window.addEventListener('resize', function () { clearTimeout(resizeT); resizeT = setTimeout(renderAll, 150); });

  renderAsOf();
  renderStale();
  setupFilters();
  setupTheme();
  setupTableViews();
  renderValueKey();
  renderGrowthKey();
  renderMethod();
  if (!ALL.length) {
    $('#kpis').appendChild(h('p', { class: 'empty', text: 'No companies were evaluated in this run.' }));
  }
  renderAll();
})();
"""
