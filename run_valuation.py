"""Valuation driver: fetch filings, test them against Graham and Buffett, track.

    python run_valuation.py                         # the watchlist file
    python run_valuation.py --tickers KO,AAPL,JPM   # just these
    python run_valuation.py --universe sp500        # the whole index (~5 min first run)
    python run_valuation.py --open                  # open the dashboard when done
    python run_valuation.py --force                 # refetch filings and prices

Each run
  1. resolves tickers to SEC CIKs and fetches XBRL company facts - refetched
     only when the filing index shows a new 10-K or 10-Q,
  2. builds a split-adjusted annual panel per company - see fundamentals.py for
     the three ways raw XBRL silently produces wrong numbers - or reuses the
     one built from the same filing last time,
  3. fetches split-adjusted (not dividend-adjusted) prices, the AAA corporate
     yield and CPI from FRED,
  4. runs Graham's seven defensive tests, his 1976 ten-point screen, and the
     Buffett quality proxies; computes a range of intrinsic-value models and
     the growth today's price implies,
  5. appends the run to the tracker's history store, refreshes
     valuation_tracker.xlsx (a view of the last 400 days), flags threshold
     crossings since the last run, and writes valuation_dashboard.html.

Research tooling, as in the scanner. An intrinsic value is an opinion with
arithmetic attached; nothing here is a recommendation to buy or sell anything.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback
from pathlib import Path

# pythonw (the scheduled-task launcher) leaves stdout/stderr as None; the
# scanner learned this the hard way. Point them at a file before importing
# anything that prints on load.
TASK_LOG = Path(__file__).resolve().parent / 'valuation_task.out'
if sys.stdout is None or sys.stderr is None:
    _fh = open(TASK_LOG, 'a', encoding='utf-8', buffering=1)
    if sys.stdout is None:
        sys.stdout = _fh
    if sys.stderr is None:
        sys.stderr = _fh

import pandas as pd  # noqa: E402

import os  # noqa: E402

from valuation import classes, fundamentals, score, sources, tracker  # noqa: E402
from valuation import dashboard as dash  # noqa: E402
from valuation.config import (CONFIG_VERSION, DEFAULT, LOG_PATH, ROOT,  # noqa: E402
                              SHARE_CLASSES, TRACKER_PATH)

log = logging.getLogger('valuation')
DEFAULT_WATCHLIST = ROOT / 'valuation_watchlist.txt'


def setup_logging(verbose: bool = True) -> None:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fh = logging.FileHandler(LOG_PATH, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s  %(levelname)-7s %(message)s'))
    log.addHandler(fh)
    if verbose:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter('%(message)s'))
        log.addHandler(sh)


def read_watchlist(path: Path, with_lists: bool = False):
    """Tickers from a watchlist file.

    One ticker per line; `#` starts a comment; blank lines are ignored. A line
    `@sp500` or `@sp600` adds every constituent of that index. With
    `with_lists`, also returns ``{ticker: [lists it came from]}`` - "Watchlist"
    for a line of its own, the index's name for a constituent - which the
    dashboard filters on.
    """
    if not path.exists():
        raise FileNotFoundError(
            f'{path} not found. Create it with one ticker per line, or pass '
            '--tickers / --universe.')
    lists: dict = {}
    # utf-8-sig: a file saved with a byte-order mark (Notepad can) otherwise
    # turns its first line into a ticker made of an invisible character.
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        t = line.split('#', 1)[0].strip()
        if not t:
            continue
        if t.startswith('@'):
            name = t[1:].lower()
            if name not in sources.INDEX_NAMES:
                raise ValueError(f'{path.name}: unknown list {t!r} - use @sp500 or @sp600')
            for c in sources.index_constituents(name):
                lists.setdefault(c, []).append(sources.INDEX_NAMES[name])
        else:
            lists.setdefault(t.upper().replace('.', '-'), []).append('Watchlist')
    tickers = list(lists)
    return (tickers, lists) if with_lists else tickers


def universe_tickers(name: str) -> list:
    """Every constituent of an index (`sp500`, `sp600`)."""
    return sources.index_constituents(name)


def write_alerts(changes: pd.DataFrame | None, path: Path) -> int:
    """The run's changes that are news, as markdown for the workflow to post.

    News means a company moved: not a first evaluation, and not a change of
    rules. Returns how many were written; writes nothing when there are none.
    """
    if changes is None or changes.empty:
        return 0
    real = changes[(changes['change'] != 'new')
                   & ~changes['change'].str.contains('(rules', regex=False)]
    if real.empty:
        return 0

    def cell(v):
        return '' if v is None or (isinstance(v, float) and pd.isna(v)) else str(v).replace('|', '/')
    rows = ['| Company | Change | Before | After | Detail |', '|---|---|---|---|---|']
    rows += [f'| {cell(c.Ticker)} | {cell(c.change)} | {cell(c.before)} | {cell(c.after)} | '
             f'{cell(c.detail)} |' for c in real.itertuples()]
    day = pd.Timestamp(real['date'].iloc[0]).strftime('%Y-%m-%d')
    path.write_text(f'{len(real)} change(s) in the {day} run:\n\n' + '\n'.join(rows) + '\n',
                    encoding='utf-8')
    return len(real)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Graham / Buffett valuation tracker')
    src = ap.add_mutually_exclusive_group()
    src.add_argument('--tickers', help='comma-separated tickers, e.g. KO,AAPL,JPM')
    src.add_argument('--watchlist', type=Path, default=None,
                     help=f'ticker file (default {DEFAULT_WATCHLIST.name})')
    src.add_argument('--universe', choices=['sp500', 'sp600'],
                     help='every constituent of an index')
    ap.add_argument('--limit', type=int, default=None, help='evaluate only the first N')
    ap.add_argument('--force', action='store_true', help='refetch filings and prices')
    ap.add_argument('--no-tracker', action='store_true', help='do not write the workbook')
    ap.add_argument('--no-dashboard', action='store_true', help='do not write the HTML')
    ap.add_argument('--open', action='store_true', help='open the dashboard when done')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)

    setup_logging(verbose=not args.quiet)
    cfg = DEFAULT
    t0 = time.time()

    if args.tickers:
        tickers = [t.strip().upper().replace('.', '-') for t in args.tickers.split(',') if t.strip()]
        lists, label = {t: ['Ad hoc'] for t in tickers}, 'ad hoc'
    elif args.universe:
        tickers = universe_tickers(args.universe)
        lists, label = {t: [sources.INDEX_NAMES[args.universe]] for t in tickers}, args.universe
    else:
        path = args.watchlist or DEFAULT_WATCHLIST
        tickers, lists = read_watchlist(path, with_lists=True)
        label = path.stem
    if args.limit:
        tickers = tickers[:args.limit]
    if not tickers:
        log.error('no tickers to evaluate')
        return 2

    log.info(f'=== valuation run: {len(tickers)} tickers ({label}), config {CONFIG_VERSION} ===')
    verbose = not args.quiet

    # Load the history store before doing any work. If it cannot be read the
    # run stops here, in seconds, instead of after the fetch - and never by
    # silently treating an unreadable record as an empty one.
    store = {}
    if not args.no_tracker:
        try:
            store = tracker.load_store(TRACKER_PATH)
        except tracker.TrackerLocked as exc:
            log.error(str(exc))
            return 3
    previous_latest = store.get('Latest')

    log.info('[1/5] macro')
    macro = sources.get_macro(cfg, verbose=verbose)

    log.info('[2/5] filings')
    # Splits first, so each panel can be built the moment its filing is read
    # and the ~20 MB document released - holding them all measured 1.1 GB at
    # 49 companies and would not fit in memory for the S&P 500. A panel built
    # from the same document, splits and code last run is reused unparsed.
    splits = sources.get_splits(tickers, force=args.force, verbose=verbose)
    # Multi-class filers: their per-share figures come from each annual
    # report's own XBRL, read once and kept (see valuation/classes.py).
    with_classes = [t for t in tickers if t in SHARE_CLASSES]
    if with_classes:
        classes.refresh(sources.resolve_ciks(with_classes, verbose=False)[0],
                        pause=cfg.sec_pause, verbose=verbose)

    def key_for(t, sha1, meta):
        extra = classes.signature(t)
        return fundamentals.entry_key(f'{sha1}+{extra}' if sha1 and extra else sha1,
                                      splits.get(t), meta, cfg)

    built = sources.fetch_many(
        tickers, cfg, force=args.force, verbose=verbose,
        consume=lambda t, blob: fundamentals.build_entry(
            t, classes.inject(blob, t), cfg, splits.get(t),
            key=key_for(t, blob['sha1'], blob['meta'])),
        prebuilt=None if args.force else (lambda t, ref, meta: fundamentals.cached_entry(
            t, key_for(t, ref.sha1, meta), cfg, source=ref.source)))
    if not built:
        log.error('no filings could be fetched; nothing to evaluate')
        return 4
    panels, infos = fundamentals.collect_built(built, verbose=verbose)
    del built

    log.info('[3/5] prices')
    prices = sources.get_prices(list(panels), years=12, force=args.force, verbose=verbose)

    log.info('[4/5] evaluate')
    payments = sources.get_dividends(list(panels))
    results, latest = score.evaluate_all(panels, infos, prices, macro, cfg, verbose=verbose,
                                         payments=payments)
    if latest.empty:
        log.error('nothing could be evaluated')
        return 5

    runtime = time.time() - t0
    run_id = tracker.new_run_id()

    log.info('[5/5] write')
    if not args.no_tracker:
        updates = tracker.build_updates(results, latest, macro, previous_latest,
                                        run_id, runtime, label, len(tickers))
        try:
            store = tracker.write_tracker(updates, TRACKER_PATH, store=store)
            log.info(f'  tracker: {tracker.store_dir(TRACKER_PATH).name}/ '
                     f'({len(store.get("History", []))} history rows), '
                     f'view in {TRACKER_PATH.name}')
        except tracker.TrackerLocked as exc:
            stash = tracker.stash_pending(updates, TRACKER_PATH)
            log.error(f'  {exc}')
            log.error(f'  run stashed to {stash}; it merges on the next successful run')
        changes = updates.get('Changes')
        if changes is not None and not changes.empty:
            real = changes[changes['change'] != 'new']
            if len(real):
                log.info(f'  {len(real)} change(s) since the last run:')
                for _, c in real.head(15).iterrows():
                    log.info(f'    {c["Ticker"]:<6} {c["change"]:<24} '
                             f'{c["before"]} -> {c["after"]}')
        # On GitHub the workflow posts these to an issue, which notifies you.
        if os.environ.get('VALUATION_ALERTS'):
            n = write_alerts(changes, Path(os.environ['VALUATION_ALERTS']))
            if n:
                log.info(f'  {n} change(s) written for the alert')

    if not args.no_dashboard:
        payload = dash.build_payload(results, panels, macro, store.get('History'), cfg,
                                     changes=store.get('Changes'), lists=lists)
        path = dash.render(payload)
        log.info(f'  dashboard: {path.name}')
        if args.open:
            import webbrowser
            webbrowser.open(path.resolve().as_uri())

    _print_summary(latest)
    log.info(f'=== done in {time.time() - t0:.0f}s ===')
    return 0


def _print_summary(latest: pd.DataFrame) -> None:
    """The handful of lines worth reading in a log."""
    # Selected on the measurement, not on a list of verdict labels that a
    # renamed verdict would silently fall out of.
    measured = latest[latest['MoSVerdict'] == 'MEASURED']
    show = measured[measured['MarginOfSafety'] >= 0]
    if show.empty:
        log.info('  no company is priced at or below its estimated value on a measured margin')
    else:
        log.info(f'  {len(show)} compan{"y" if len(show) == 1 else "ies"} priced at or below estimated value:')
        for _, r in show.iterrows():
            log.info(f'    {r["Ticker"]:<6} {r["Verdict"]:<28} price {r["Price"]:>9,.2f}  '
                     f'value {r["IntrinsicValue"]:>9,.2f}  margin {r["MarginOfSafety"]:>6.0%}')
    if len(measured):
        # The median of measured companies only: a withheld margin's
        # price/value is exactly what the withholding keeps out of summaries.
        log.info(f'  margin of safety measurable for {len(measured)}/{len(latest)}; '
                 f'median price/value {measured["PriceToValue"].median():.2f}x')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        log.error('run failed:\n' + traceback.format_exc())
        sys.exit(1)
