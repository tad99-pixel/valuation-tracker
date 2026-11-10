"""The valuation tracker: an append-only store, and an Excel view of it.

A single valuation is a snapshot. What makes this a *tracker* is the history:
how intrinsic value estimates drift as filings arrive, when a price crosses
below a Graham threshold, when a verdict changes.

**Where the history lives.** The record of every run is kept in parquet files
next to the workbook (``<workbook>_store/``): ``history``, ``runlog`` and
``changes`` are append-only, ``latest`` is replaced each run. Each run appends
a few kilobytes and rewrites nothing that already exists.

**The workbook is a view.** ``valuation_tracker.xlsx`` is regenerated from the
store on every run, with History and Changes cut to the last
`WORKBOOK_HISTORY_DAYS`. An earlier version kept everything in the workbook,
read every sheet back and rewrote it in full each run - cost that grew with
days x tickers without limit (an estimated three minutes a run after a year of
the S&P 500). It also made the workbook the single point of failure: open in
Excel, the run had to stop or stash. Now an open workbook only means the view
is refreshed next time; the store - the record - is already written.

| Sheet       | Contents                                                       |
|-------------|----------------------------------------------------------------|
| RunLog      | one row per run: universe size, verdict counts, AAA, CPI       |
| Latest      | the current scorecard, one row per company                     |
| Changes     | threshold crossings and verdict changes (recent window)        |
| History     | price, value and margin of safety per company (recent window)  |
| Criteria    | every test, its verdict, value and source tier (latest run)    |
| Models      | every intrinsic-value model and its reasoning (latest run)     |
| Provenance  | where each threshold comes from - quote, paraphrase, or proxy  |

A store that cannot be written raises `TrackerLocked`, and the caller stashes
the run in ``valuation_pending/`` to be merged by the next successful run.
"""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG_VERSION, PROVENANCE, SOURCE_TIERS, TRACKER_PATH, ScoreConfig

SHEET_ORDER = ['RunLog', 'Latest', 'Changes', 'History', 'Criteria', 'Models',
               'Provenance']

# Sheets that accumulate, and the key that makes a re-run on the same day
# replace rather than duplicate its own rows.
APPEND_KEYS = {
    'RunLog': ['run_id'],
    'History': ['date', 'Ticker'],
    'Changes': ['date', 'Ticker', 'change'],
}
# Sheets that describe only the latest run and are replaced wholesale.
REPLACE_SHEETS = {'Latest', 'Criteria', 'Models', 'Provenance'}
# Kept in the store between runs: the append sheets, plus Latest for change
# detection. Criteria and Models are rebuilt every run and only shown.
STORED = ('History', 'RunLog', 'Changes', 'Latest')

# How much of the accumulating sheets the workbook shows. The store keeps all.
# The window shrinks for a large universe so the view stays near
# WORKBOOK_MAX_ROWS rows: 400 days of the S&P 500 would be 200,000, slow to
# write on every run and to open.
WORKBOOK_HISTORY_DAYS = 400
WORKBOOK_MAX_ROWS = 50_000

HISTORY_COLS = ['date', 'run_id', 'Ticker', 'Price', 'IntrinsicValue',
                'IntrinsicLow', 'IntrinsicHigh', 'PriceToValue', 'MarginOfSafety',
                'MoSVerdict', 'MoSBand', 'Verdict', 'Composite', 'Completeness',
                'ImpliedGrowth', 'DCF', 'EPV', 'EarningsMultipleValue',
                'JustifiedPB', 'GrahamNumber', 'GrahamBargainPrice',
                'ConfigVersion']


class TrackerLocked(RuntimeError):
    """The store (or, during migration, the old workbook) cannot be used."""


def new_run_id() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def store_dir(path=TRACKER_PATH) -> Path:
    """The store sits beside its workbook, so a scratch workbook (set through
    VALUATION_TRACKER) never mixes its runs into the real history."""
    return Path(path).parent / f'{Path(path).stem}_store'


def _store_file(path, sheet: str) -> Path:
    return store_dir(path) / f'{sheet.lower()}.parquet'


# ---------------------------------------------------------------------------
# Store I/O
# ---------------------------------------------------------------------------

def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Object columns holding a mix of strings, floats and None break pyarrow;
    store them as text, keeping missing values missing."""
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v))
                              else (v if isinstance(v, str) else str(v)))
    return df


def _write_atomic(df: pd.DataFrame, target: Path) -> None:
    tmp = target.with_suffix('.tmp.parquet')
    _parquet_safe(df).to_parquet(tmp, index=False)
    last = None
    for attempt in range(4):
        try:
            os.replace(tmp, target)
            return
        except OSError as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise TrackerLocked(f'could not replace {target.name} ({last}); '
                        f'the new content is in {tmp.name}')


def load_store(path=TRACKER_PATH) -> dict:
    """The accumulated record: ``{'History', 'RunLog', 'Changes', 'Latest'}``.

    On first use, a store is seeded from an existing workbook so the history it
    holds carries over. A workbook that exists but cannot be read at that
    moment (open in Excel) raises rather than being skipped: starting an empty
    store beside a full workbook would lose the history the next time the view
    was written.
    """
    d = store_dir(path)
    out = {}
    if d.exists() and any(_store_file(path, s).exists() for s in STORED):
        for s in STORED:
            f = _store_file(path, s)
            if f.exists():
                try:
                    out[s] = pd.read_parquet(f)
                except Exception as exc:  # noqa: BLE001
                    raise TrackerLocked(f'store file {f.name} unreadable ({exc}); '
                                        'nothing was overwritten') from exc
        return out

    path = Path(path)
    if not path.exists():
        return {}
    book, last = None, None
    for attempt in range(3):
        try:
            book = pd.read_excel(path, sheet_name=None)
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    if book is None:
        raise TrackerLocked(
            f'{path.name} exists but could not be read to seed the history store '
            f'({last}). It is probably open in Excel - close it and re-run. '
            'Nothing was overwritten.')
    d.mkdir(parents=True, exist_ok=True)
    for s in STORED:
        if s in book and not book[s].empty:
            df = book[s]
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            _write_atomic(df, _store_file(path, s))
            out[s] = df
    print(f'  tracker: history store seeded from {path.name} '
          f'({len(out.get("History", []))} history rows)')
    return out


def _pending_dir(path=TRACKER_PATH):
    d = Path(path).parent / 'valuation_pending'
    d.mkdir(exist_ok=True)
    return d


def stash_pending(updates: dict, path=TRACKER_PATH):
    """Park a run's sheets as parquet when the store cannot be written."""
    out = _pending_dir(path) / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    for sheet, df in updates.items():
        if sheet not in APPEND_KEYS or df is None or df.empty:
            continue
        try:
            _parquet_safe(df).to_parquet(out / f'{sheet}.parquet')
        except Exception:
            df.to_csv(out / f'{sheet}.csv', index=False)
    return out


def _drain_pending(path=TRACKER_PATH) -> tuple:
    """Append-only sheets from any stashed runs, oldest first.

    Returns ``(sheets, drained_folders)``. A folder holding a file that cannot
    be read is left out of `drained_folders`, so it is never deleted - an
    unreadable stash is kept for a person to look at, not silently discarded.
    """
    d = _pending_dir(path)
    merged, drained = {}, []
    for folder in sorted(p for p in d.iterdir() if p.is_dir()):
        ok = True
        for f in sorted(folder.iterdir()):
            if f.stem not in APPEND_KEYS:
                continue          # stale snapshots of replace-sheets are dropped
            try:
                df = pd.read_parquet(f) if f.suffix == '.parquet' else pd.read_csv(f)
            except Exception as exc:  # noqa: BLE001
                print(f'  pending stash {folder.name}/{f.name} unreadable ({exc}); kept')
                ok = False
                continue
            merged[f.stem] = (pd.concat([merged[f.stem], df], ignore_index=True)
                              if f.stem in merged else df)
        if ok:
            drained.append(folder)
    return merged, drained


# ---------------------------------------------------------------------------
# Building the sheets for one run
# ---------------------------------------------------------------------------

def criteria_frame(results: dict, run_date) -> pd.DataFrame:
    rows = []
    for t, r in results.items():
        for block, label in (('defensive', 'Graham defensive'),
                             ('enterprising', 'Graham 1976 screen'),
                             ('quality', 'Buffett quality')):
            for c in r[block]:
                row = c.to_row()
                row.update({'date': run_date, 'Ticker': t, 'Block': label})
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    front = ['date', 'Ticker', 'Block', 'number', 'name', 'verdict', 'display',
             'threshold', 'detail', 'source_tier', 'citation', 'structural', 'value']
    return df[[c for c in front if c in df.columns]]


def models_frame(results: dict, run_date) -> pd.DataFrame:
    rows = []
    for t, r in results.items():
        cons = r.get('consensus') or {}
        for k, m in (r.get('models') or {}).items():
            rows.append({
                'date': run_date, 'Ticker': t, 'model': k,
                'label': m.get('label'), 'role': m.get('role'),
                'value': m.get('value'),
                'applicable': bool(m.get('applicable')),
                'in_consensus': k in (cons.get('models_used') or []),
                'source_tier': m.get('source_tier'),
                'note': m.get('note'),
            })
    return pd.DataFrame(rows)


def provenance_frame() -> pd.DataFrame:
    return pd.DataFrame([
        {'key': k, 'tier': tier, 'tier_meaning': SOURCE_TIERS.get(tier, ''),
         'citation': cite}
        for k, (tier, cite) in PROVENANCE.items()
    ])


# How far past a threshold a move must go to be reported. Across the S&P 500
# about twenty companies sit within 2% of their estimate, their Graham Number or
# a margin band's edge on any day, and ordinary price moves would report them
# crossing back and forth. Each row of `Latest` carries the state last reported
# (the `_ref_*` columns), and a move is measured from there: one that falls
# back inside the buffer is not news, and the next real one is still caught.
CROSSING_BUFFER = 0.02          # of the threshold's own value
BAND_BUFFER = 0.02              # of margin of safety, in points
_BANDS = ('PREMIUM', 'THIN', 'ADEQUATE', 'STRONG')
_CROSSINGS = (('IntrinsicValue', 'intrinsic value'), ('GrahamNumber', 'Graham Number'),
              ('GrahamBargainPrice', '2/3 NCAV'))


def _band_of(margin: float) -> str:
    s = ScoreConfig()
    if margin >= s.mos_strong:
        return 'STRONG'
    if margin >= s.mos_adequate:
        return 'ADEQUATE'
    return 'THIN' if margin >= 0 else 'PREMIUM'


def _band_move_is_clear(ref, now, margin) -> bool:
    """Has the margin left the reported band by more than the buffer?

    A move to or from "withheld" has no margin to measure and always counts.
    """
    if ref not in _BANDS or now not in _BANDS or margin is None:
        return True
    up = _BANDS.index(now) > _BANDS.index(ref)
    return _band_of(margin - BAND_BUFFER if up else margin + BAND_BUFFER) != ref


def detect_changes(previous: pd.DataFrame | None, latest: pd.DataFrame,
                   run_date) -> pd.DataFrame:
    """What moved since the last run - the reason to keep a tracker at all.

    Reported:
      * a verdict change (WATCH -> CANDIDATE)
      * a margin-of-safety band change (THIN -> ADEQUATE)
      * the price crossing below - or back above - the median intrinsic value,
        the Graham Number, or two-thirds of net current asset value. These are
        the thresholds a Graham investor acts on, and a crossing is news in a
        way a level is not.
      * a company appearing for the first time.

    Crossings are only reported when both runs had a usable value. A threshold
    that existed yesterday and is unmeasurable today is reported as exactly
    that, not as a crossing.

    A change of rules is not news about a company. When the two runs used
    different `CONFIG_VERSION`s, the changes are still recorded - a reader
    should see them - but each is labelled with the rule change, so a verdict
    moved by a new threshold is never read as the company having moved. A
    rule change also resets the reported state, without the buffers.

    Changes are measured from the state last reported, not from yesterday's,
    and must clear `CROSSING_BUFFER` or `BAND_BUFFER` (see above). A verdict
    change that comes with a margin band move inside the buffer waits with it.
    The reported state is written back onto `latest` as `_ref_*` columns.
    """
    rows = []
    if latest is None or latest.empty:
        return pd.DataFrame()
    prev = (previous.set_index('Ticker') if previous is not None
            and not previous.empty and 'Ticker' in previous.columns else None)
    refs: dict = {}

    def add(t, change, before, after, detail):
        rows.append({'date': run_date, 'Ticker': t, 'change': change,
                     'before': None if before is None else str(before),
                     'after': None if after is None else str(after), 'detail': detail})

    def num(x):
        try:
            v = float(x)
            return v if np.isfinite(v) else None
        except (TypeError, ValueError):
            return None

    def text(x):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)

    def below(px, v):
        return None if px is None or v is None else px < v

    def current(row):
        px = num(row.get('Price'))
        state = {'verdict': text(row.get('Verdict')), 'band': text(row.get('MoSBand'))}
        state.update({col: below(px, num(row.get(col))) for col, _ in _CROSSINGS})
        return state

    for row in latest.to_dict('records'):
        t = row['Ticker']
        if prev is None or t not in prev.index:
            add(t, 'new', None, row.get('Verdict'), 'first evaluation')
            refs[t] = current(row)
            continue
        p = prev.loc[t]
        if isinstance(p, pd.DataFrame):
            p = p.iloc[-1]
        c0, c1 = text(p.get('ConfigVersion')), text(row.get('ConfigVersion'))
        tag = f' (rules {c0} -> {c1})' if c0 and c1 and c0 != c1 else ''
        now = current(row)
        state = dict(now)

        # The reported state; a row from before it was kept stands in for it.
        ref_verdict = text(p.get('_ref_verdict')) or text(p.get('Verdict'))
        ref_band = text(p.get('_ref_band'))
        ref_band = text(p.get('MoSBand')) if ref_band is None else (
            None if ref_band == 'withheld' else ref_band)
        band_clear = bool(tag) or _band_move_is_clear(
            ref_band, now['band'], num(row.get('MarginOfSafety')))

        if now['verdict'] != ref_verdict:
            if now['band'] == ref_band or band_clear:
                add(t, 'verdict' + tag, ref_verdict, now['verdict'],
                    f'composite {num(p.get("Composite")) or 0:.2f} -> '
                    f'{num(row.get("Composite")) or 0:.2f}')
            else:
                state['verdict'] = ref_verdict
        if now['band'] != ref_band:
            if band_clear:
                add(t, 'margin band' + tag, ref_band or 'withheld', now['band'] or 'withheld',
                    str(row.get('MoSVerdict')))
            else:
                state['band'] = ref_band

        px0, px1 = num(p.get('Price')), num(row.get('Price'))
        for col, label in _CROSSINGS:
            v0, v1 = num(p.get(col)), num(row.get(col))
            if v1 is None or px1 is None:
                if v0 is not None and v1 is None:
                    add(t, f'{label} lost{tag}', f'{v0:,.2f}', None,
                        'not computable under the new rules' if tag
                        else 'no longer computable from the latest filings')
                continue
            held = num(p.get(f'_ref_below_{col}'))
            ref_below = bool(held) if held is not None else below(px0, v0)
            if ref_below is None or now[col] == ref_below:
                continue
            clear = bool(tag) or (px1 <= v1 * (1 - CROSSING_BUFFER) if now[col]
                                  else px1 >= v1 * (1 + CROSSING_BUFFER))
            if not clear:
                state[col] = ref_below
            elif now[col]:
                add(t, f'below {label}{tag}', f'{px0:,.2f} vs {v0:,.2f}' if None not in (px0, v0)
                    else None, f'{px1:,.2f} vs {v1:,.2f}', f'price crossed below {label}')
            else:
                add(t, f'above {label}{tag}', f'{px0:,.2f} vs {v0:,.2f}' if None not in (px0, v0)
                    else None, f'{px1:,.2f} vs {v1:,.2f}', f'price crossed back above {label}')
        refs[t] = state

    latest['_ref_verdict'] = latest['Ticker'].map(lambda t: refs[t]['verdict'])
    latest['_ref_band'] = latest['Ticker'].map(lambda t: refs[t]['band'] or 'withheld')
    for col, _ in _CROSSINGS:
        latest[f'_ref_below_{col}'] = latest['Ticker'].map(
            lambda t, c=col: np.nan if refs[t][c] is None else float(refs[t][c]))
    return pd.DataFrame(rows)


def run_log_row(run_id, run_date, latest: pd.DataFrame, macro: dict,
                runtime_s: float, watchlist: str, n_requested: int) -> pd.DataFrame:
    from .score import VERDICT_SLUGS
    vc = latest['Verdict'].value_counts().to_dict() if not latest.empty else {}
    row = {'run_id': run_id, 'date': run_date, 'watchlist': watchlist,
           'requested': n_requested, 'evaluated': int(len(latest))}
    # One count per verdict, named from the single list in score - a renamed
    # or added verdict can no longer be silently counted as zero here.
    for label, slug in VERDICT_SLUGS.items():
        row[slug] = int(vc.get(label, 0))
    row.update({
        'mos_measured': int((latest.get('MoSVerdict') == 'MEASURED').sum())
        if 'MoSVerdict' in latest else 0,
        'aaa_yield': macro.get('aaa_yield'),
        'treasury_10y': macro.get('treasury_10y'),
        'cpi_ratio': macro.get('cpi_ratio'),
        'config_version': CONFIG_VERSION,
        'runtime_s': round(runtime_s, 1),
    })
    return pd.DataFrame([row])


def run_date_of(latest: pd.DataFrame | None) -> pd.Timestamp:
    """The trading day a run describes: the newest close it priced.

    Runs were dated by the machine's calendar, so a run made after midnight
    UTC - the catch-up schedule, or any run from Singapore the next morning -
    filed Monday's closes under Tuesday, and Monday's cloud runs, made before
    the US open, filed Friday's under Monday. Dated by its prices, a later
    run for the same close replaces the earlier one's rows.
    """
    if latest is not None and 'PriceDate' in latest.columns:
        d = pd.to_datetime(latest['PriceDate'], errors='coerce').max()
        if pd.notna(d):
            return pd.Timestamp(d).normalize()
    return pd.Timestamp.today().normalize()


def close_recorded(store: dict | None, close) -> bool:
    """Whether the newest run in the store already priced `close`."""
    latest = (store or {}).get('Latest')
    if latest is None or latest.empty or 'PriceDate' not in latest.columns:
        return False
    newest = pd.to_datetime(latest['PriceDate'], errors='coerce').max()
    return bool(pd.notna(newest) and newest.normalize() >= pd.Timestamp(close).normalize())


def build_updates(results: dict, latest: pd.DataFrame, macro: dict,
                  previous_latest: pd.DataFrame | None, run_id: str,
                  runtime_s: float, watchlist: str, n_requested: int) -> dict:
    run_date = run_date_of(latest)
    hist = latest.copy()
    hist['date'] = run_date
    hist['run_id'] = run_id
    hist = hist[[c for c in HISTORY_COLS if c in hist.columns]]
    return {
        'RunLog': run_log_row(run_id, run_date, latest, macro, runtime_s,
                              watchlist, n_requested),
        'Latest': latest,
        'Changes': detect_changes(previous_latest, latest, run_date),
        'History': hist,
        'Criteria': criteria_frame(results, run_date),
        'Models': models_frame(results, run_date),
        'Provenance': provenance_frame(),
    }


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _merge_append(old: pd.DataFrame | None, new: pd.DataFrame, sheet: str) -> pd.DataFrame:
    merged = (pd.concat([old, new], ignore_index=True)
              if old is not None and not old.empty else new.copy())
    keys = [k for k in APPEND_KEYS.get(sheet, []) if k in merged.columns]
    if 'date' in merged.columns:
        merged['date'] = pd.to_datetime(merged['date']).dt.normalize()
    if keys:
        # A same-day re-run replaces its own rows rather than duplicating.
        merged = merged.drop_duplicates(subset=keys, keep='last')
    if 'date' in merged.columns:
        merged = merged.sort_values('date', kind='stable')
    return merged.reset_index(drop=True)


def write_tracker(updates: dict, path=TRACKER_PATH, store: dict | None = None) -> dict:
    """Append one run to the store, then refresh the workbook view.

    Returns the updated store. Raises `TrackerLocked` only when the *store*
    cannot be written; a workbook that cannot be written (open in Excel) is
    reported and left for the next run, since nothing in it is the record.
    """
    path = Path(path)
    store = dict(store) if store is not None else load_store(path)
    store_dir(path).mkdir(parents=True, exist_ok=True)

    # Stashed rows go through exactly the same merge as this run's rows. Stash
    # first, this run last: on a key collision the newer row wins.
    pending, folders = _drain_pending(path)
    updates = dict(updates)
    for sheet, df in pending.items():
        new = updates.get(sheet)
        updates[sheet] = (pd.concat([df, new], ignore_index=True)
                          if new is not None and not new.empty else df)

    for sheet in APPEND_KEYS:
        new = updates.get(sheet)
        if new is None or new.empty:
            continue
        store[sheet] = _merge_append(store.get(sheet), new, sheet)
        _write_atomic(store[sheet], _store_file(path, sheet))
    if updates.get('Latest') is not None and not updates['Latest'].empty:
        store['Latest'] = updates['Latest'].copy()
        _write_atomic(store['Latest'], _store_file(path, 'Latest'))

    for f in folders:
        try:
            shutil.rmtree(f)
        except OSError as exc:
            # Harmless: these rows are now in the store, and if the folder is
            # drained again the append keys de-duplicate them.
            print(f'  merged stash {f.name} could not be removed ({exc}); left in place')

    try:
        write_workbook(store, updates, path)
    except Exception as exc:  # noqa: BLE001 - a failed view must never cost the record
        print(f'  workbook view not refreshed ({type(exc).__name__}: {exc}). The run is '
              f'saved in {store_dir(path).name}/ and the view will be rebuilt next run.')
    return store


def write_workbook(store: dict, updates: dict, path=TRACKER_PATH) -> None:
    """Regenerate the Excel view: replace-sheets from this run, append-sheets
    from the store cut to the recent window."""
    path = Path(path)
    companies = max(1, len(store.get('Latest', [])))
    days = max(30, min(WORKBOOK_HISTORY_DAYS, WORKBOOK_MAX_ROWS // companies))
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
    book = {}
    for sheet in SHEET_ORDER:
        if sheet in APPEND_KEYS:
            df = store.get(sheet)
            if df is not None and sheet != 'RunLog' and 'date' in df.columns:
                df = df[pd.to_datetime(df['date']) >= cutoff]
        else:
            df = updates.get(sheet) if updates.get(sheet) is not None else store.get(sheet)
        if df is not None and not df.empty:
            # `_ref_*`: the tracker's memory of what it last reported, not a figure.
            book[sheet] = df.drop(columns=[c for c in df.columns if str(c).startswith('_')])
    if not book:
        return          # nothing inside the window yet; the store holds the record

    tmp = path.with_suffix('.tmp.xlsx')
    try:
        with pd.ExcelWriter(tmp, engine='openpyxl') as xl:
            for sheet, df in book.items():
                _excel_safe(df).to_excel(xl, sheet_name=sheet[:31], index=False)
            _format(xl, book)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    last = None
    for attempt in range(3):
        try:
            os.replace(tmp, path)
            return
        except OSError as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    try:
        tmp.unlink()
    except OSError:
        pass
    raise TrackerLocked(f'{path.name} is held by another program ({last})')


def _excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Excel cannot hold tz-aware datetimes, infinities, or very long text."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            try:
                df[c] = df[c].dt.tz_localize(None)
            except (TypeError, AttributeError):
                pass
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].replace([np.inf, -np.inf], np.nan)
        elif df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].map(lambda v: (None if isinstance(v, float) and not np.isfinite(v)
                                         else v[:32000] if isinstance(v, str) else v))
    return df


PCT_COLS = {'Completeness', 'MarginOfSafety', 'MarginOfSafetyRaw',
            'ImpliedGrowth', 'HistoricalGrowth', 'DCFGrowthUsed',
            'EPSGrowth10y', 'ROEMedian', 'ROICMedian', 'GrossMarginMedian',
            'NetMarginMedian', 'FCFMarginMedian', 'ShareCountCAGR',
            'ModelDispersion'}


def _format(writer, book) -> None:
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    head_fill = PatternFill('solid', fgColor='1F3864')
    head_font = Font(color='FFFFFF', bold=True, size=10)
    pass_fill = PatternFill('solid', fgColor='D9EAD3')
    fail_fill = PatternFill('solid', fgColor='F4CCCC')
    na_fill = PatternFill('solid', fgColor='EEEEEE')

    for sheet, df in book.items():
        ws = writer.sheets[sheet[:31]]
        ws.freeze_panes = 'B2' if sheet in ('Latest', 'History') else 'A2'
        for cell in ws[1]:
            cell.fill = head_fill
            cell.font = head_font
            cell.alignment = Alignment(horizontal='center', vertical='center',
                                       wrap_text=True)
        n = len(df)
        for i, col in enumerate(df.columns, start=1):
            letter = get_column_letter(i)
            # Width from a sample, not a string copy of the whole column.
            sample = df[col].head(200).astype(str)
            longest = int(sample.str.len().max() or 10) if len(sample) else 10
            width = max(10, min(48 if col in ('detail', 'note', 'citation', 'Caveats')
                                else 26, longest + 2), len(str(col)) + 3)
            ws.column_dimensions[letter].width = width
            fmt = ('0.0%' if col in PCT_COLS
                   else '#,##0.00' if np.issubdtype(df[col].dtype, np.floating) else None)
            if fmt and n:
                for row in ws.iter_rows(min_row=2, max_row=n + 1, min_col=i, max_col=i):
                    row[0].number_format = fmt
        if 'verdict' in df.columns and n:
            letter = get_column_letter(list(df.columns).index('verdict') + 1)
            rng = f'{letter}2:{letter}{n + 1}'
            ws.conditional_formatting.add(rng, CellIsRule(operator='equal', formula=['"PASS"'], fill=pass_fill))
            ws.conditional_formatting.add(rng, CellIsRule(operator='equal', formula=['"FAIL"'], fill=fail_fill))
            ws.conditional_formatting.add(rng, CellIsRule(operator='equal', formula=['"N/A"'], fill=na_fill))
        if n and len(df.columns):
            ws.auto_filter.ref = ws.dimensions
