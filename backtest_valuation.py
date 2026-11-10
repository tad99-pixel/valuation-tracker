"""Point-in-time backtest: did the tracker's verdicts predict anything?

    python backtest_valuation.py                      # year-ends 2014-2023, today's S&P 500
    python backtest_valuation.py --tickers KO,AAPL    # a few names
    python backtest_valuation.py --analyse-only       # redo the tables from the last run's rows
    python backtest_valuation.py --root PATH          # read the tracker's caches from another checkout
    python tests/test_backtest.py                     # its tests, offline

For each year-end from 2014 to 2023, every company is valued as the tracker
would have valued it that day:

* its panel is rebuilt from the XBRL facts filed on or before the day, and
  nothing filed later (`fundamentals.build_panel(as_of=...)`): later
  restatements, fiscal years and cover-page counts are dropped, and the
  history window, the filing's age and the share count are judged from that
  day. A company that has since moved to a new registration - Alphabet,
  Disney, Linde, Medtronic - is read from the registration that was filing
  then (`PREDECESSORS`);
* it is priced at that day's close, with the twelve years of closes before
  it that the live tracker downloads, and its dividend record is cut there;
* Graham's tests read the AAA yield of the last month completed before the
  day, and the CPI of the last month published by then;
* `score.evaluate` - the tracker's own code, at today's configuration -
  gives the verdict and the price-to-value ratio.

Each company's total return over the next one, three and five years is then
measured from Yahoo's split-adjusted closes and its dividends, reinvested at
the ex-date close (the tracker's own prices carry no dividends), and each
verdict group and each price-to-value quintile is set against the
equal-weighted universe.

**Survivorship.** The universe is today's S&P 500. Every company in it
survived to 2026, and many joined the index only after the year they are
tested in, because they had done well; the companies that went bankrupt,
were acquired or shrank out of the index are missing. Returns are
overstated, and relative results can be too. `survivorship` measures the
size of it against the point-in-time index membership and the S&P 500
Equal Weight ETF (RSP), which held the actual index.

Not point in time either: the configuration (r7, written in 2026, after the
data traps it handles had been found), the filer's SIC code and entity type
(today's), and the CPI's seasonal factors (revised since).

Downloads (prices, dividends, the membership history, filings the tracker
has not cached) and results go to `<root>/data_cache/backtest/`. The
tracker's filings, FRED series and CIK map are only read; its S&P 500 list
refreshes itself weekly, as in a tracker run.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from valuation import classes, fundamentals, score, sources
from valuation.config import CONFIG_VERSION, DEFAULT, ROOT

warnings.filterwarnings('ignore')

HORIZONS = (1, 3, 5)
FIRST_YEAR, LAST_YEAR = 2014, 2023
# Far enough back for every as-of day's tests: ten fiscal years of market
# value for the one-dollar premise, twenty years of dividends for Graham.
PRICES_SINCE = '1990-01-01'
# The live tracker downloads twelve years of closes (`run_valuation.py`).
PRICE_YEARS = 12
# A close older than this is not the day's price: the tracker's own limit.
MAX_STALE_DAYS = sources.MAX_PRICE_STALENESS_DAYS
# CPI for a month is published in the middle of the next one.
CPI_LAG_DAYS = 21
# A dividend this large a share of the price, on the day of a split, is a
# spin-off Yahoo has booked twice (see `total_return_index`).
SPIN_DIVIDEND_SHARE = 0.05
# RSP holds the S&P 500 Equal Weight Index - the actual index of each day,
# rebalanced quarterly, the companies that later left it included. SPY is the
# cap-weighted index, for reference.
BENCHMARKS = ('RSP', 'SPY')
PIT_URL = ('https://raw.githubusercontent.com/fja05680/sp500/master/'
           'S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv')
WIKI_URL = sources.INDEX_URLS['sp500']
UA = {'User-Agent': 'ValuationTracker-backtest/1.0'}

# Registrations a company filed under before its current one, newest first,
# beyond the two the tracker merges itself (`config.CIK_SUCCESSIONS`). A
# reorganisation into a new holding company starts a new CIK, and EDGAR does
# not link the two: as of 2014, Alphabet had filed nothing - Google Inc. had
# filed everything. A reader at the time read the predecessor, so the
# backtest does. Each was checked against SEC's filing index (name, former
# names, 10-K dates), and Yahoo's history for the ticker reaches back into
# the predecessor's years. Kept here, not in the tracker, whose panels they
# would change.
PREDECESSORS = {
    1652044: (1288776,),            # Alphabet (2015) <- Google Inc.
    1613103: (64670,),              # Medtronic plc (2015) <- Medtronic, Inc.
    1551182: (31277,),              # Eaton plc (2012) <- Eaton Corporation
    1571949: (1174746,),            # Intercontinental Exchange (2013) <- IntercontinentalExchange
    1707925: (884905,),             # Linde plc (2018) <- Praxair
    1730168: (1649338, 1441634),    # Broadcom Inc. (2018) <- Broadcom Ltd. (2016) <- Avago
    1711269: (54507,),              # Evergy (2018) <- Westar Energy
    1739940: (701221,),             # The Cigna Group (2018) <- Cigna Corporation
    1757898: (1624899, 815065),     # STERIS (Ireland, 2019) <- STERIS plc (2015) <- STERIS Corp.
    1744489: (1001039,),            # Walt Disney (2019) <- Disney before the Fox acquisition
    1792044: (1623613, 69499),      # Viatris (2020) <- Mylan N.V. (2015) <- Mylan Inc.
    1841666: (6769,),               # APA Corporation (2021) <- Apache
    1835632: (1058057,),            # Marvell Technology, Inc. (2021) <- Marvell Technology Group
    1858681: (1411494,),            # Apollo Global Management (2022) <- Apollo before Athene
    1973266: (1091907,),            # TKO Group (2023) <- World Wrestling Entertainment
    1701605: (808362,),             # Baker Hughes Company (2017) <- Baker Hughes Incorporated
    2012383: (1364742,),            # BlackRock (2024) <- BlackRock before GIP
    1996862: (1144519,),            # Bunge Global (2023) <- Bunge Limited
    2011641: (1832433,),            # Ferguson Enterprises (2024) <- Ferguson plc
    2041610: (813828,),             # Paramount Skydance (2025) <- Paramount Global (CBS)
}

# Rows with no tracker verdict: priced that day, but no annual report on
# file in XBRL yet; declined as a fund or trust; or the panel failed.
NO_PANEL, DECLINED, ERROR = 'NO ANNUAL REPORT YET', 'DECLINED (NOT OPERATING)', 'ERROR'
VERDICT_ORDER = [v[0] for v in score.VERDICTS] + [NO_PANEL, DECLINED, ERROR]
# The tracker's own buy list: the verdicts `score.VERDICTS` counts as candidates.
CANDIDATE_VERDICTS = tuple(v[0] for v in score.VERDICTS if v[3])
CHEAP_VERDICTS = CANDIDATE_VERDICTS + ('STATISTICALLY CHEAP ONLY',)


# ---------------------------------------------------------------------------
# Where things are
# ---------------------------------------------------------------------------

class Paths:
    """The tracker's caches (read) and the backtest's own folder (written)."""

    def __init__(self, root=ROOT, work=None):
        self.root = Path(root)
        self.facts = self.root / 'data_cache' / 'sec_facts'
        self.macro = self.root / 'data_cache' / 'macro'
        self.state = self.root / 'valuation_state'
        self.work = Path(work) if work else self.root / 'data_cache' / 'backtest'
        self.own_facts = self.work / 'sec_facts'
        self.prices = self.work / 'prices'
        self.membership = self.work / 'membership'
        for d in (self.work, self.own_facts, self.prices, self.membership):
            d.mkdir(parents=True, exist_ok=True)


def read_tracker_state(paths: Paths) -> None:
    """Read the index list, the CIK map and the share-class figures from the
    tracker checkout at `paths.root` (a no-op inside that checkout)."""
    sources.STATE_DIR = paths.state
    sources.CIK_MAP_PATH = paths.state / 'cik_map.json'
    sources.CIK_OVERRIDE_PATH = paths.state / 'cik_overrides.json'
    classes.STORE_PATH = paths.state / 'share_classes.json'


def year_ends(first: int, last: int) -> list:
    return [pd.Timestamp(f'{y}-12-31') for y in range(first, last + 1)]


# ---------------------------------------------------------------------------
# Companies and their filings
# ---------------------------------------------------------------------------

def index_table(paths: Paths) -> pd.DataFrame:
    """The S&P 500's constituents table - Symbol, CIK, DateAdded - cached for
    a week. The same table the universe comes from (`index_constituents`)."""
    text = _cached_text(paths.membership / 'sp500_wikipedia.html', WIKI_URL, max_age_days=7)
    for tbl in pd.read_html(StringIO(text)):
        cols = {str(c): c for c in tbl.columns}
        if {'Symbol', 'CIK', 'Date added'} <= set(cols):
            return pd.DataFrame({
                'Symbol': [sources._norm_ticker(s) for s in tbl[cols['Symbol']]],
                'CIK': pd.to_numeric(tbl[cols['CIK']], errors='coerce'),
                'DateAdded': pd.to_datetime(tbl[cols['Date added']], errors='coerce')})
    raise ValueError('no constituents table with Symbol, CIK and "Date added" columns')


def resolve(tickers: list, paths: Paths, verbose: bool = True) -> tuple:
    """``({ticker: cik}, unresolved)``.

    The index table's CIK column first: it is kept for the current
    constituents, where SEC's ticker file - the tracker's map - can lag a
    ticker that changed hands. In November 2026 that map sent BNY to a
    BlackRock municipal bond fund and DOC to Physicians Realty, whose ticker
    Healthpeak took in 2024. The tracker's map, read as it is, covers
    anything else.
    """
    try:
        table = index_table(paths).dropna(subset=['CIK'])
        found = {s: int(c) for s, c in zip(table['Symbol'], table['CIK']) if s in set(tickers)}
    except Exception as exc:  # noqa: BLE001 - offline: the tracker's map alone
        if verbose:
            print(f'  index table unavailable ({exc}); CIKs from the tracker map')
        found = {}
    cache = sources._read_cik_cache()
    known = sources._with_overrides(dict(cache[0]) if cache else {})
    found.update({t: int(known[t]) for t in tickers if t not in found and t in known})
    return found, [t for t in tickers if t not in found]


def family(cik: int) -> list:
    """Every registration of the company, current first: the tracker's own
    successions, else the backtest's `PREDECESSORS`."""
    tracked = sources.cik_family(cik)
    if len(tracked) > 1:
        return tracked
    for current, earlier in PREDECESSORS.items():
        if cik == current or cik in earlier:
            return [current, *earlier]
    return [cik]


def one_per_company(tickers: list, ciks: dict) -> dict:
    """``{ticker: cik}``, the first listed ticker of each company: two share
    classes of one company (Alphabet, Fox, News Corp) are one company's
    returns twice, and would weigh double in an equal-weighted universe."""
    out, seen = {}, set()
    for t in tickers:
        cik = ciks.get(t)
        if cik is None:
            continue
        fam = tuple(family(int(cik)))
        if fam in seen:
            continue
        seen.add(fam)
        out[t] = int(cik)
    return out


def first_annual_report(doc: dict):
    """The filing date of the first annual report in a companyfacts document."""
    first = None
    for concepts in (doc.get('facts') or {}).values():
        for node in concepts.values():
            for rows in (node.get('units') or {}).values():
                for r in rows:
                    f = r.get('filed')
                    if f and r.get('form') in fundamentals.ANNUAL_FORMS and (first is None or f < first):
                        first = f
    return first


def chain_facts(docs: list) -> dict:
    """One document from a company's registrations, current first.

    Each predecessor counts for what it filed before the registration that
    followed it filed its first annual report. After that it is a subsidiary
    or a shell - Apache, Baker Hughes Holdings LLC and Apollo Asset
    Management kept filing 10-Ks for years - and its figures are not the
    company's: merged in whole, a subsidiary's later filing would win
    latest-filed-wins over the parent's own.
    """
    kept = [docs[0]]
    for later, doc in zip(docs, docs[1:]):
        start = first_annual_report(later)
        if start:
            doc = fundamentals.facts_as_of(doc, pd.Timestamp(start) - pd.Timedelta(days=1))
        kept.append(doc)
    return kept[0] if len(kept) == 1 else sources.merge_facts(kept)


def _fetched_after(folder: Path, cik: int, day) -> bool:
    """Was the cached document fetched after `day`, so that it holds every
    fact filed by then?"""
    meta = sources._read_json(folder / f'CIK{cik:010d}.meta.json', {})
    try:
        fetched = pd.Timestamp(meta['fetched'])
    except (KeyError, ValueError, TypeError):
        return False
    if fetched.tzinfo is not None:
        fetched = fetched.tz_convert(None)
    return fetched.normalize() > pd.Timestamp(day)


def ensure_documents(companies: dict, paths: Paths, last_day, cfg=DEFAULT,
                     verbose: bool = True) -> dict:
    """Every company's filings on disk: ``{ticker: {'facts': [paths], 'sub': path}}``.

    The tracker's cached copies serve when they were fetched after the last
    as-of day; anything else is fetched from SEC by the tracker's own
    fetchers, into the backtest's folder. A company that changed
    registration has one document per registration (`family`), current
    first.
    """
    out, fetched, failed = {}, 0, []
    for i, (t, cik) in enumerate(companies.items(), 1):
        if verbose and i % 50 == 0:
            print(f'    filings {i}/{len(companies)}', end='\r')
        fam = family(cik)
        docs, sub = [], None
        for c in fam:
            p = paths.facts / f'CIK{c:010d}.json.gz'
            if p.exists() and _fetched_after(paths.facts, c, last_day):
                docs.append(p)
                if c == fam[0] and (paths.facts / f'SUB{c:010d}.json.gz').exists():
                    sub = paths.facts / f'SUB{c:010d}.json.gz'
                continue
            got = _fetch_into(paths.own_facts, c, cfg)
            if got is not None:
                docs.append(got)
                fetched += 1
        if sub is None:
            sub = _fetch_submissions_into(paths.own_facts, fam[0], cfg)
        if not docs or sub is None:
            failed.append(t)
            continue
        out[t] = {'facts': [str(d) for d in docs], 'sub': str(sub)}
    if verbose:
        print(' ' * 40, end='\r')
        print(f'  filings: {len(out)} companies ({fetched} documents fetched from SEC)'
              + (f'; none for {", ".join(failed)}' if failed else ''))
    return out


def _with_facts_dir(folder: Path, fn):
    real = sources.FACTS_DIR
    sources.FACTS_DIR = folder
    try:
        return fn()
    finally:
        sources.FACTS_DIR = real


def _fetch_submissions_into(folder: Path, cik: int, cfg):
    meta = _with_facts_dir(folder, lambda: sources.fetch_submissions(cik, cfg))
    p = folder / f'SUB{cik:010d}.json.gz'
    return p if meta is not None and p.exists() else None


def _fetch_into(folder: Path, cik: int, cfg):
    meta = _fetch_submissions_into(folder, cik, cfg)
    latest = None
    if meta is not None:
        with gzip.open(meta, 'rt', encoding='utf-8') as fh:
            latest = json.load(fh).get('latest_filing')
    ref, _ = _with_facts_dir(folder, lambda: sources.fetch_companyfacts(
        cik, cfg, latest_filing=latest))
    p = folder / f'CIK{cik:010d}.json.gz'
    return p if ref is not None and p.exists() else None


def _read_gz(path) -> dict:
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Prices, dividends and total returns
# ---------------------------------------------------------------------------

def _price_frame(raw, t: str):
    """Close (split-adjusted, as the tracker's), Adj Close, Dividends and
    Stock Splits for `t` from a download, or None.

    A dividend dated on a row with no close moves to the next close, where it
    can be reinvested, rather than being lost with the row.
    """
    sub = sources._ticker_frame(raw, t)
    if sub is None or 'Close' not in sub or sub['Close'].notna().sum() == 0:
        return None
    sub = sub.copy()
    idx = pd.DatetimeIndex(sub.index)
    sub.index = idx.tz_localize(None) if idx.tz is not None else idx
    sub = sub[~sub.index.duplicated(keep='last')].sort_index()
    traded = sub['Close'].notna() & (sub['Close'] > 0)
    df = sub.loc[traded, [c for c in ('Close', 'Adj Close') if c in sub]].astype(float)
    for col, how in (('Dividends', 'sum'), ('Stock Splits', 'max')):
        ev = (sub[col] if col in sub else pd.Series(0.0, index=sub.index)).fillna(0.0)
        ev = ev[ev > 0]
        pos = df.index.searchsorted(ev.index)
        keep = pos < len(df)
        moved = pd.Series(ev.to_numpy()[keep], index=df.index[pos[keep]])
        df[col] = moved.groupby(level=0).agg(how).reindex(df.index).fillna(0.0)
    df = sources._drop_live_bar(df)
    return df if len(df) else None


def download_prices(tickers: list, paths: Paths, refresh: bool = False,
                    verbose: bool = True) -> list:
    """Daily prices since 1990 for each ticker, one parquet each. Returns the
    tickers Yahoo has nothing for (recorded, so they are not asked for again
    until `refresh`)."""
    import yfinance as yf
    missing_path = paths.prices / '_missing.json'
    known_missing = set() if refresh else set(sources._read_json(missing_path, []))
    todo = [t for t in tickers if t not in known_missing
            and (refresh or not (paths.prices / f'{t}.parquet').exists())]
    if verbose and todo:
        print(f'  prices: downloading {len(todo)} tickers since {PRICES_SINCE[:4]}')
    got = {}
    for i in range(0, len(todo), 40):
        batch = todo[i:i + 40]
        raw = None
        for threads in (True, False):
            try:
                raw = yf.download(batch, start=PRICES_SINCE, auto_adjust=False, actions=True,
                                  progress=False, group_by='ticker', threads=threads,
                                  timeout=30 if threads else 60)
                break
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f'  price batch failed ({exc})' + ('; retrying slower' if threads else ''))
                time.sleep(10)
        for t in batch:
            df = _price_frame(raw, t)
            if df is None:
                try:
                    one = yf.download(t, start=PRICES_SINCE, auto_adjust=False, actions=True,
                                      progress=False, group_by='ticker', threads=False, timeout=60)
                    df = _price_frame(one, t)
                except Exception:  # noqa: BLE001
                    df = None
            if df is not None:
                df.to_parquet(paths.prices / f'{t}.parquet')
                got[t] = df
        if verbose:
            print(f'    prices {min(i + 40, len(todo))}/{len(todo)}', end='\r')
        time.sleep(0.6)
    missing = sorted((known_missing | set(todo)) - set(got)
                     - {t for t in tickers if (paths.prices / f'{t}.parquet').exists()})
    sources._write_json(missing_path, missing)
    if verbose and todo:
        print(' ' * 40, end='\r')
        print(f'  prices: {len(got)} downloaded' + (f'; none for {", ".join(missing)}' if missing else ''))
    return missing


def read_prices(path):
    try:
        df = pd.read_parquet(path)
    except Exception:  # noqa: BLE001 - missing or unreadable
        return None
    df.index = pd.DatetimeIndex(df.index)
    return df if len(df) else None


def total_return_index(px: pd.DataFrame) -> pd.Series:
    """Growth of one dollar, each dividend reinvested at its ex-date close:
    (close + dividend) / previous close, the CRSP convention.

    Yahoo's dividends are stated per share on today's split basis, the same
    basis as its split-adjusted closes, so the two add. Yahoo books a
    spin-off as a fractional split - GE 1.281 for GE HealthCare, AT&T 1.324
    for Warner Bros. Discovery - which keeps the adjusted close continuous,
    the spun-off shares' value staying in the return. Where it books the
    same spin-off as a dividend too (Danaher's Fortive, July 2016: a 1.319
    split and $24.56), the dividend would count it twice and is dropped: any
    dividend above `SPIN_DIVIDEND_SHARE` of the previous close, on a day with
    a split.
    """
    close = px['Close'].astype(float)
    div = (px['Dividends'].astype(float) if 'Dividends' in px
           else pd.Series(0.0, index=px.index))
    if 'Stock Splits' in px:
        twice = (px['Stock Splits'] > 0) & (div > SPIN_DIVIDEND_SHARE * close.shift(1))
        div = div.where(~twice, 0.0)
    gross = (close + div) / close.shift(1)
    gross.iloc[0] = 1.0
    return gross.cumprod().rename('TR')


def _at(series: pd.Series, day):
    """The last value on or before `day`, if it is no more than
    `MAX_STALE_DAYS` old; else None."""
    s = series.loc[:pd.Timestamp(day)]
    if s.empty or (pd.Timestamp(day) - s.index[-1]).days > MAX_STALE_DAYS:
        return None
    return float(s.iloc[-1])


def forward_returns(tr: pd.Series | None, day, data_end, horizons=HORIZONS,
                    prefix: str = 'TR') -> dict:
    """``{'TR1y': ..., 'TR3y': ...}``: the total return from the close of `day`
    to the close `h` years later. None where either close is missing, or the
    window ends after `data_end` (not yet observable)."""
    out = {f'{prefix}{h}y': None for h in horizons}
    if tr is None:
        return out
    start = _at(tr, day)
    if start is None or start <= 0:
        return out
    for h in horizons:
        end_day = pd.Timestamp(day) + pd.DateOffset(years=h)
        if end_day > pd.Timestamp(data_end):
            continue
        end = _at(tr, end_day)
        if end is not None:
            out[f'{prefix}{h}y'] = end / start - 1.0
    return out


def prices_as_of(px: pd.DataFrame | None, day):
    """The closes the tracker would have had on `day`: twelve years up to it,
    or None when the last of them is stale (not yet listed, or halted)."""
    if px is None:
        return None
    day = pd.Timestamp(day)
    close = px['Close'].loc[day - pd.DateOffset(years=PRICE_YEARS):day]
    if close.empty or (day - close.index[-1]).days > MAX_STALE_DAYS:
        return None
    return close.to_frame('Close')


def payments_as_of(px: pd.DataFrame | None, day):
    """The dividend record as of `day`, for Graham's twenty-year test:
    payments on or before it, `since` the first close on file."""
    if px is None or 'Dividends' not in px:
        return None
    d = px['Dividends']
    d = d[(d > 0) & (d.index <= pd.Timestamp(day))].astype(float)
    d.attrs['since'] = px.index[0]
    return d


def split_history(px: pd.DataFrame | None):
    """Splits since 2008, as `sources.get_splits` gives the tracker; None when
    there is no price history to read them from."""
    if px is None or 'Stock Splits' not in px:
        return None
    return sources._as_split_series(px['Stock Splits'])


# ---------------------------------------------------------------------------
# Macro inputs as of a day
# ---------------------------------------------------------------------------

def load_macro(paths: Paths) -> dict:
    """AAA, CPIAUCSL and DGS10 from the tracker's FRED cache, else from FRED."""
    out = {}
    for sid in ('AAA', 'CPIAUCSL', 'DGS10'):
        try:
            s = pd.read_parquet(paths.macro / f'{sid}.parquet')
            s = s[s.columns[0]]
        except Exception:  # noqa: BLE001 - not cached: fetch, keep in our folder
            s = sources._fred_download(sid)
            s.to_frame().to_parquet(paths.work / f'{sid}.parquet')
        s.index = pd.DatetimeIndex(s.index)
        out[sid] = s.dropna().sort_index()
    return out


def macro_as_of(series: dict, day, cfg=DEFAULT) -> dict:
    """What `sources.get_macro` would have returned on `day`.

    Both FRED series are monthly and dated the first of the month. The AAA
    yield is a monthly average, complete once the month has ended; CPI is
    published about three weeks after that (`CPI_LAG_DAYS`). At the end of
    December 2014, both are November's.
    """
    day = pd.Timestamp(day)
    aaa = series['AAA']
    aaa = aaa[aaa.index + pd.DateOffset(months=1) <= day]
    cpi_all = series['CPIAUCSL']
    cpi = cpi_all[cpi_all.index + pd.DateOffset(months=1) + pd.Timedelta(days=CPI_LAG_DAYS) <= day]
    t10 = series['DGS10'].loc[:day]
    base = cpi_all[cpi_all.index.year == cfg.graham.base_year]
    return {
        'aaa_yield': float(aaa.iloc[-1]) if len(aaa) else None,
        'aaa_as_of': aaa.index[-1] if len(aaa) else None,
        'treasury_10y': float(t10.iloc[-1]) if len(t10) else None,
        'cpi_ratio': float(cpi.iloc[-1] / base.mean()) if len(cpi) and len(base) else None,
        'cpi_as_of': cpi.index[-1] if len(cpi) else None,
        'cpi_base_year': cfg.graham.base_year,
    }


# ---------------------------------------------------------------------------
# Evaluation, one company at a time
# ---------------------------------------------------------------------------

def evaluate_company(job: dict) -> list:
    """Every as-of row for one company: what the tracker would have said
    that day. Runs in a worker process; returns are attached afterwards
    (`attach_returns`).

    `job`: ticker, cik, facts (paths), sub (path), prices (path), class_store
    (path), days, macro ({day: dict}).
    """
    t = job['ticker']
    classes.STORE_PATH = Path(job['class_store'])
    cfg = DEFAULT
    px = read_prices(job['prices'])
    splits = split_history(px)
    facts, meta, problem = None, {}, None
    if job['facts']:        # none: SEC has no XBRL filings for it at all
        try:
            facts = chain_facts([_read_gz(p) for p in job['facts']])
            meta = _read_gz(job['sub']) if job.get('sub') else {}
        except Exception as exc:  # noqa: BLE001
            facts, problem = None, f'filings unreadable: {exc}'
    etype = str(meta.get('entityType') or '').lower()
    if facts is not None:
        # Visa and Berkshire: per-class figures the tracker has read, if any.
        facts = classes.inject({'facts': facts}, t)['facts']

    rows = []
    for day in job['days']:
        day = pd.Timestamp(day)
        row = {'AsOf': day, 'Ticker': t, 'CIK': job['cik']}
        cut = prices_as_of(px, day)
        row['PricedAtDay'] = cut is not None
        if problem:
            row.update(Verdict=ERROR, Caveats=problem)
        elif facts is None:
            row.update(Verdict=NO_PANEL, Caveats='no XBRL filings found')
        elif etype and etype != 'operating':
            row.update(Verdict=DECLINED, Caveats=f'SEC entity type {etype}')
        else:
            try:
                panel, info = fundamentals.build_panel(t, facts, meta, cfg, splits=splits,
                                                       as_of=day)
                if panel is None or panel.empty:
                    row.update(Verdict=NO_PANEL,
                               Caveats=(info or {}).get('error') or 'no annual report filed')
                else:
                    info['facts_source'] = 'backtest'
                    r = score.evaluate(t, panel, info, cut, job['macro'][day], cfg,
                                       payments_as_of(px, day), as_of=day)
                    row.update(score.to_row(r))
                    row['Ticker'] = t
            except Exception as exc:  # noqa: BLE001
                row.update(Verdict=ERROR, Caveats=f'{type(exc).__name__}: {exc}')
        rows.append(row)
    return rows


def attach_returns(df: pd.DataFrame, paths: Paths, data_end) -> pd.DataFrame:
    """Each row's forward total returns (TR1y, TR3y, TR5y) and price returns
    (PR...), from the price files - kept apart from the evaluation, so a
    change to the return arithmetic needs no new evaluation."""
    found = {}
    for t, g in df.groupby('Ticker'):
        px = read_prices(paths.prices / f'{t}.parquet')
        tr = total_return_index(px) if px is not None else None
        close = px['Close'] if px is not None else None
        for i, day in g['AsOf'].items():
            found[i] = {**forward_returns(tr, day, data_end),
                        **forward_returns(close, day, data_end, prefix='PR')}
    ret = pd.DataFrame.from_dict(found, orient='index').astype(float)
    out = df.drop(columns=[c for c in ret.columns if c in df.columns]).join(ret)
    out.attrs['data_end'] = f'{pd.Timestamp(data_end):%Y-%m-%d}'
    return out


def data_end_of(paths: Paths):
    """The last close the benchmark has: no window ends later."""
    spy = read_prices(paths.prices / 'SPY.parquet')
    return spy.index[-1] if spy is not None else pd.Timestamp.today().normalize()


def run_evaluation(jobs: list, workers: int, verbose: bool = True) -> list:
    rows, t0 = [], time.time()
    if workers <= 1:
        results = map(evaluate_company, jobs)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=workers)
        results = pool.map(evaluate_company, jobs, chunksize=2)
    try:
        for i, out in enumerate(results, 1):
            rows.extend(out)
            if verbose and (i % 25 == 0 or i == len(jobs)):
                print(f'    evaluated {i}/{len(jobs)} companies ({time.time() - t0:.0f}s)', end='\r')
    finally:
        if pool is not None:
            pool.shutdown()
    if verbose:
        print()
    return rows


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _ann(r, h: int):
    """An `h`-year cumulative return, annualised."""
    return (1.0 + r) ** (1.0 / h) - 1.0


def portfolios(df: pd.DataFrame, by: str, h: int) -> pd.DataFrame:
    """One equal-weighted portfolio per as-of day and group, bought at the
    day's close and held `h` years, beside the equal-weighted universe.

    The universe is every company priced that day whose `h`-year return can
    be observed - whatever its verdict. Columns: AsOf, the group, n, ret,
    universe, n_universe, excess (annualised, so horizons compare).
    """
    col = f'TR{h}y'
    d = df[df[col].notna()]
    uni = d.groupby('AsOf')[col].agg(universe='mean', n_universe='count')
    g = (d.groupby(['AsOf', by], observed=True)[col].agg(ret='mean', n='count')
         .reset_index().merge(uni, left_on='AsOf', right_index=True))
    g['excess'] = _ann(g['ret'], h) - _ann(g['universe'], h)
    return g


def summarise_portfolios(p: pd.DataFrame, by: str, h: int, order=None) -> pd.DataFrame:
    """Per group: the years it was held, members per year, its annualised
    return and excess over the universe averaged across years, the share of
    years it beat the universe, and - for one-year windows, which do not
    overlap - the t-statistic of the yearly excess."""
    rows = []
    for key, g in p.groupby(by, observed=True):
        ex = g['excess']
        t = (ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)))
             if h == 1 and len(ex) >= 3 and ex.std(ddof=1) > 0 else None)
        rows.append({by: key, 'years': len(g), 'n_total': int(g['n'].sum()),
                     'n_per_year': g['n'].mean(), 'ret_ann': _ann(g['ret'], h).mean(),
                     'excess_ann': ex.mean(), 'beat_share': (ex > 0).mean(), 't': t})
    out = pd.DataFrame(rows)
    if order is not None and len(out):
        out[by] = pd.Categorical(out[by], [o for o in order if o in set(out[by])], ordered=True)
        out = out.sort_values(by).reset_index(drop=True)
    return out


def universe_summary(df: pd.DataFrame, h: int) -> dict:
    col = f'TR{h}y'
    u = df[df[col].notna()].groupby('AsOf')[col].agg(['mean', 'count'])
    return {'years': len(u), 'n_per_year': u['count'].mean(), 'ret_ann': _ann(u['mean'], h).mean()}


def quintiles(df: pd.DataFrame, value: str, ok: pd.Series, cheap_low: bool = True,
              col: str = 'PVQuintile') -> pd.DataFrame:
    """Quintile of `value` within each as-of day, among the rows `ok` marks:
    1 the cheapest fifth, 5 the dearest. `cheap_low`: a low value is cheap
    (price-to-value), or a high one is (earnings yield)."""
    df = df.copy()
    df[col] = np.nan
    for _, g in df[ok].groupby('AsOf'):
        if len(g) >= 5:
            r = g[value].rank(method='first', ascending=cheap_low)
            df.loc[g.index, col] = pd.qcut(r, 5, labels=[1, 2, 3, 4, 5]).astype(float)
    df.attrs.update(quintile_value=value, cheap_low=cheap_low)
    return df


def add_quintiles(df: pd.DataFrame, measured_only: bool = True, col: str = 'PVQuintile') -> pd.DataFrame:
    """Price-to-value quintiles. `measured_only` ranks only companies whose
    margin of safety the tracker measured (the models agree within 3x);
    otherwise every company with an estimate, withheld margins included."""
    ok = df['PriceToValue'].notna() & (df['PriceToValue'] > 0)
    if measured_only:
        ok &= df['MoSVerdict'].eq('MEASURED')
    return quintiles(df, 'PriceToValue', ok, cheap_low=True, col=col)


def spread_and_ic(df: pd.DataFrame, h: int, col: str = 'PVQuintile') -> dict:
    """Cheapest-minus-dearest quintile (annualised, averaged over years) and
    the rank correlation of cheapness with the forward return within each
    year - positive when the cheaper did better, whichever measure ranked
    them (`quintiles` records it in the frame's attrs)."""
    ret = f'TR{h}y'
    value = df.attrs.get('quintile_value', 'PriceToValue')
    sign = -1.0 if df.attrs.get('cheap_low', True) else 1.0
    d = df[df[col].notna() & df[ret].notna()]
    spreads, ics = [], []
    for _, g in d.groupby('AsOf'):
        q = g.groupby(col)[ret].mean()
        if 1 in q.index and 5 in q.index:
            spreads.append(_ann(q[1], h) - _ann(q[5], h))
        if len(g) >= 10:
            ics.append(sign * g[value].rank().corr(g[ret].rank()))
    out = {}
    for name, xs in (('spread', spreads), ('ic', ics)):
        xs = np.array(xs, dtype=float)
        out[name] = xs.mean() if len(xs) else None
        out[f'{name}_years'] = len(xs)
        out[f'{name}_t'] = (xs.mean() / (xs.std(ddof=1) / np.sqrt(len(xs)))
                            if h == 1 and len(xs) >= 3 and xs.std(ddof=1) > 0 else None)
        out[f'{name}_positive'] = (xs > 0).mean() if len(xs) else None
    return out


# ---------------------------------------------------------------------------
# Survivorship
# ---------------------------------------------------------------------------

def _cached_text(path: Path, url: str, max_age_days: int = 30) -> str:
    if path.exists() and (time.time() - path.stat().st_mtime) / 86400 <= max_age_days:
        return path.read_text(encoding='utf-8')
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    path.write_text(r.text, encoding='utf-8')
    return r.text


def pit_membership(paths: Paths) -> pd.DataFrame:
    """The S&P 500's members on each day it changed, since 1996, as the
    tickers of the day (fja05680/sp500, compiled from Andreas Clenow's list
    and the index's announced changes). Columns: date, tickers (a set)."""
    text = _cached_text(paths.membership / 'sp500_historical_components.csv', PIT_URL)
    df = pd.read_csv(StringIO(text))
    df['date'] = pd.to_datetime(df['date'])
    df['tickers'] = df['tickers'].map(lambda s: {sources._norm_ticker(x) for x in str(s).split(',') if x})
    return df.sort_values('date').reset_index(drop=True)


def members_on(pit: pd.DataFrame, day) -> set:
    rows = pit[pit['date'] <= pd.Timestamp(day)]
    return set() if rows.empty else rows['tickers'].iloc[-1]


def date_added(paths: Paths) -> dict:
    """``{ticker: date}`` each current constituent last joined the index, from
    the index table - under whatever ticker it had then, so a renamed company
    (Facebook, now Meta) keeps its date."""
    table = index_table(paths)
    return dict(zip(table['Symbol'], table['DateAdded']))


def survivorship(df: pd.DataFrame, pit: pd.DataFrame, added: dict, bench: dict,
                 data_end) -> tuple:
    """``(per-year table, {h: summary})``.

    Per as-of day: how many companies the index held (its own list), how
    many of today's constituents were priced, how many of those were in the
    index then and how many joined it later, and the one-year equal-weighted
    return of each - beside RSP, which held the index as it actually was.
    """
    df = df.copy()
    df['Added'] = df['Ticker'].map(added)
    df['InIndex'] = np.where(df['Added'].isna(), np.nan, (df['Added'] <= df['AsOf']).astype(float))
    rows = []
    for day, g in df[df['PricedAtDay']].groupby('AsOf'):
        members = members_on(pit, day)
        stay = g[g['InIndex'] == 1.0]
        later = g[g['InIndex'] == 0.0]
        row = {'AsOf': day, 'index_members': len(members), 'priced': len(g),
               'in_index_then': len(stay), 'joined_later': len(later),
               'ticker_in_index_then': int(g['Ticker'].isin(members).sum())}
        row['left_since'] = len(members) - row['in_index_then']
        for h in HORIZONS:
            col = f'TR{h}y'
            row[f'universe_{h}y'] = g[col].mean() if g[col].notna().any() else None
            row[f'stayers_{h}y'] = stay[col].mean() if stay[col].notna().any() else None
            row[f'joined_{h}y'] = later[col].mean() if later[col].notna().any() else None
            for b, tr in bench.items():
                row[f'{b}_{h}y'] = forward_returns(tr, day, data_end, (h,))[f'TR{h}y']
        rows.append(row)
    per_year = pd.DataFrame(rows)
    summary = {}
    for h in HORIZONS:
        d = per_year.dropna(subset=[f'universe_{h}y', f'RSP_{h}y'])
        if d.empty:
            continue
        s = {'years': len(d)}
        for k in ('universe', 'stayers', 'joined', 'RSP', 'SPY'):
            s[k] = _ann(d[f'{k}_{h}y'].astype(float), h).mean()
        s['gap_universe_rsp'] = s['universe'] - s['RSP']
        s['gap_stayers_rsp'] = s['stayers'] - s['RSP']
        summary[h] = s
    return per_year, summary


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _pct(x, digits=1, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ''
    return f'{x:+.{digits}%}' if sign else f'{x:.{digits}%}'


def _num(x, fmt='{:.1f}'):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ''
    return fmt.format(x)


def md_table(header: list, rows: list) -> str:
    out = ['| ' + ' | '.join(header) + ' |',
           '|' + '|'.join('---' if i == 0 else '---:' for i in range(len(header))) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def _horizon_cells(sums: dict, k) -> list:
    """For group `k`: the 1y return, then each horizon's annualised excess and
    years ahead, with the 1y t-statistic."""
    cells = []
    for h in HORIZONS:
        s = sums[h].loc[k] if k in sums[h].index else None
        if s is None:
            cells += ([''] if h == 1 else []) + ['', ''] + ([''] if h == 1 else [])
            continue
        if h == 1:
            cells.append(_pct(s['ret_ann']))
        cells += [_pct(s['excess_ann'], sign=True),
                  f"{int(round(s['beat_share'] * s['years']))}/{int(s['years'])}"]
        if h == 1:
            cells.append(_num(s['t'], '{:+.1f}'))
    return cells


def _horizon_header(first: list) -> list:
    header = list(first)
    for h in HORIZONS:
        header += (['1y return'] if h == 1 else []) + [f'{h}y excess', f'{h}y ahead'] + (
            ['t (1y)'] if h == 1 else [])
    return header


def _universe_note(df: pd.DataFrame) -> str:
    u = {h: universe_summary(df, h) for h in HORIZONS}
    return ('Equal-weighted universe: {:.0f} companies a year; '.format(u[1]['n_per_year'])
            + ', '.join(f"{_pct(u[h]['ret_ann'])} a year over {h}-year windows ({u[h]['years']})"
                        for h in HORIZONS) + '.')


def horizon_table(df: pd.DataFrame, by: str, order: list, label=str) -> str:
    """One row per group: members, its one-year return, and for each horizon
    the annualised excess over the equal-weighted universe and the share of
    years it was ahead; for one year, the t-statistic too."""
    sums = {h: summarise_portfolios(portfolios(df, by, h), by, h, order).set_index(by)
            for h in HORIZONS}
    keys = [k for k in order if any(k in s.index for s in sums.values())]
    rows = []
    for k in keys:
        s1 = sums[1].loc[k] if k in sums[1].index else None
        rows.append([label(k), _num(s1['n_total'] if s1 is not None else None, '{:.0f}'),
                     _num(s1['n_per_year'] if s1 is not None else None)] + _horizon_cells(sums, k))
    return (md_table(_horizon_header([by, 'Company-years', 'Per year']), rows)
            + '\n\n' + _universe_note(df))


def quintile_table(df: pd.DataFrame) -> str:
    qs = [1.0, 2.0, 3.0, 4.0, 5.0]
    sums = {h: summarise_portfolios(portfolios(df, 'PVQuintile', h), 'PVQuintile', h,
                                    qs).set_index('PVQuintile') for h in HORIZONS}
    med = df.groupby('PVQuintile')['PriceToValue'].median()
    rows = []
    for q in qs:
        if q not in sums[1].index:
            continue
        name = {1.0: '1 (cheapest)', 5.0: '5 (dearest)'}.get(q, str(int(q)))
        rows.append([name, f'{med.get(q, np.nan):.2f}x', _num(sums[1].loc[q, 'n_per_year'])]
                    + _horizon_cells(sums, q))
    for name, key in (('Cheapest minus dearest', 'spread'),
                      ('Rank correlation, cheapness vs return', 'ic')):
        r = [f'**{name}**', '', '']
        for h in HORIZONS:
            si = spread_and_ic(df, h)
            v = si[key]
            r += ([''] if h == 1 else []) + [
                (_pct(v, sign=True) if key == 'spread' else _num(v, '{:+.3f}')),
                f"{int(round((si[key + '_positive'] or 0) * si[key + '_years']))}/{si[key + '_years']} > 0"]
            if h == 1:
                r.append(_num(si[key + '_t'], '{:+.1f}'))
        rows.append(r)
    return (md_table(_horizon_header(['Price-to-value quintile', 'Median P/V', 'Per year']), rows)
            + '\n\n' + _universe_note(df[df['PVQuintile'].notna()]).replace(
                'Equal-weighted universe', 'The ranked companies, equal-weighted') + ' Excess is '
            'measured against the whole universe.')


def signal_table(df: pd.DataFrame) -> str:
    """The tracker's price-to-value against the plainest value measure,
    earnings yield on three-year average EPS, ranking the same companies:
    does the modelling add anything to a P/E sort?"""
    df = df.assign(EarningsYield=pd.to_numeric(df.get('EPS3yAvg'), errors='coerce')
                   / pd.to_numeric(df.get('Price'), errors='coerce'))
    measured = df['MoSVerdict'].eq('MEASURED') & (df['PriceToValue'] > 0)
    has_ey = df['EarningsYield'].notna()
    signals = (
        ('Price-to-value, margin measured', quintiles(df, 'PriceToValue', measured, True)),
        ('Price-to-value, every estimate', quintiles(df, 'PriceToValue', df['PriceToValue'] > 0, True)),
        ('Earnings yield, the companies of the first row',
         quintiles(df, 'EarningsYield', measured & has_ey, False)),
        ('Earnings yield, every company with three years of EPS',
         quintiles(df, 'EarningsYield', has_ey, False)),
    )
    header = ['Ranking', 'Ranked per year']
    for h in HORIZONS:
        header += [f'{h}y cheapest - dearest'] + (['t'] if h == 1 else []) + [f'{h}y rank corr.'] + (
            ['t'] if h == 1 else [])
    rows = []
    for name, q in signals:
        per_year = q[q['PVQuintile'].notna()].groupby('AsOf').size().mean()
        r = [name, _num(per_year, '{:.0f}')]
        for h in HORIZONS:
            si = spread_and_ic(q, h)
            r += [_pct(si['spread'], sign=True)] + ([_num(si['spread_t'], '{:+.1f}')] if h == 1 else [])
            r += [_num(si['ic'], '{:+.3f}')] + ([_num(si['ic_t'], '{:+.1f}')] if h == 1 else [])
        rows.append(r)
    return md_table(header, rows)


def coverage_table(df: pd.DataFrame) -> str:
    rows = []
    for day, g in df[df['PricedAtDay']].groupby('AsOf'):
        measured = g[g['MoSVerdict'].eq('MEASURED')]
        vc = g['Verdict'].value_counts()
        rows.append([f'{day:%Y}', len(g), int(g['Verdict'].isin(VERDICT_ORDER[:8]).sum()),
                     _num(pd.to_numeric(g.get('FiscalYears'), errors='coerce').median(), '{:.0f}'),
                     len(measured), _num(measured['PriceToValue'].median(), '{:.2f}x'),
                     int(vc.reindex(list(CANDIDATE_VERDICTS)).fillna(0).sum()),
                     int(vc.get('GOOD BUSINESS, FAIR PRICE', 0)),
                     int(vc.get('STATISTICALLY CHEAP ONLY', 0)),
                     int(vc.get('GOOD BUSINESS, DEAR', 0)),
                     int(vc.get('GOOD BUSINESS, VALUE UNCLEAR', 0)),
                     int(vc.get('NO', 0) + vc.get('WATCH', 0)),
                     int(vc.get('INSUFFICIENT DATA', 0) + vc.get(NO_PANEL, 0)
                         + vc.get(DECLINED, 0) + vc.get(ERROR, 0))])
    return md_table(['Year-end', 'Priced', 'Verdict given', 'Median years on file',
                     'Margin measured', 'Median P/V',
                     'Candidate or cheap+quality', 'Fair price', 'Statistically cheap', 'Dear',
                     'Value unclear', 'No / watch', 'No verdict'], rows)


def survivorship_tables(per_year: pd.DataFrame, summary: dict) -> str:
    rows = []
    for _, r in per_year.iterrows():
        rows.append([f"{r['AsOf']:%Y}", r['index_members'], r['priced'], r['in_index_then'],
                     r['joined_later'], r['left_since'],
                     _pct(r['universe_1y']), _pct(r['stayers_1y']), _pct(r['joined_1y']),
                     _pct(r['RSP_1y']), _pct(r['SPY_1y'])])
    a = md_table(['Year-end', 'Index members then', "Today's constituents priced",
                  '... in the index then', '... joined later', 'Members since gone',
                  'Next year: universe', 'in index then', 'joined later', 'RSP', 'SPY'], rows)
    rows = []
    for h, s in summary.items():
        rows.append([f'{h}y ({s["years"]} windows)', _pct(s['universe']), _pct(s['stayers']),
                     _pct(s['joined']), _pct(s['RSP']), _pct(s['SPY']),
                     _pct(s['gap_universe_rsp'], sign=True), _pct(s['gap_stayers_rsp'], sign=True)])
    b = md_table(['Horizon', 'Universe', 'In index then', 'Joined later', 'RSP', 'SPY',
                  'Universe - RSP', 'In index then - RSP'], rows)
    return a, b


def check_total_returns(paths: Paths, tickers, days, tolerance: float = 0.005) -> str:
    """How far the total returns built here are from Yahoo's own adjusted
    closes, over every one-year window: a check on the dividend arithmetic."""
    worst, off = [], []
    for t in tickers:
        px = read_prices(paths.prices / f'{t}.parquet')
        if px is None or 'Adj Close' not in px:
            continue
        tr, adj = total_return_index(px), px['Adj Close']
        gaps = []
        for d in days:
            a = forward_returns(tr, d, tr.index[-1], (1,))['TR1y']
            b = forward_returns(adj, d, adj.index[-1], (1,))['TR1y']
            if a is not None and b is not None:
                gaps.append(abs(a - b))
        if gaps:
            worst.append(max(gaps))
            if max(gaps) > tolerance:
                off.append(f'{t} ({max(gaps):.1%})')
    if not worst:
        return 'Total returns: no adjusted closes to check against.'
    return (f'Total returns (closes plus dividends) against Yahoo\'s adjusted closes, every one-year '
            f'window: median largest gap {np.median(worst):.2%}; {len(off)} of {len(worst)} tickers '
            f'beyond {tolerance:.1%}' + (f': {", ".join(off[:12])}' if off else '') + '.')


def analyse(df: pd.DataFrame, paths: Paths, verbose: bool = True) -> str:
    """The tables, as markdown."""
    df = df[df['PricedAtDay']].copy()
    df['AsOf'] = pd.to_datetime(df['AsOf'])
    for c in [f'TR{h}y' for h in HORIZONS] + ['PriceToValue']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Verdict'] = df['Verdict'].fillna(ERROR)
    df['Buy list'] = np.where(df['Verdict'].isin(CANDIDATE_VERDICTS), 'Candidate or cheap, quality passes',
                              np.where(df['Verdict'].isin(CHEAP_VERDICTS), 'Statistically cheap only',
                                       'Everything else'))
    measured = add_quintiles(df, measured_only=True)
    everything = add_quintiles(df, measured_only=False)

    parts = [f'Config {CONFIG_VERSION}; as-of days {df["AsOf"].min():%Y-%m-%d} to '
             f'{df["AsOf"].max():%Y-%m-%d}; {df["Ticker"].nunique()} companies; '
             f'returns to {df.attrs.get("data_end", "")}.',
             check_total_returns(paths, sorted(df['Ticker'].unique()), sorted(df['AsOf'].unique()))]
    parts += ['## Verdicts per year', coverage_table(df)]
    parts += ['## Verdict groups against the equal-weighted universe',
              horizon_table(df, 'Verdict', VERDICT_ORDER)]
    parts += ["## The tracker's buy list", horizon_table(
        df, 'Buy list', ['Candidate or cheap, quality passes', 'Statistically cheap only',
                         'Everything else'])]
    parts += ['## Price-to-value quintiles, margin measured', quintile_table(measured)]
    parts += ['## Price-to-value quintiles, every estimate (withheld margins included)',
              quintile_table(everything)]
    parts += ['## Price-to-value against a plain earnings-yield sort', signal_table(df)]
    try:
        pit, added = pit_membership(paths), date_added(paths)
        bench = {b: total_return_index(read_prices(paths.prices / f'{b}.parquet'))
                 for b in BENCHMARKS}
        per_year, summary = survivorship(df, pit, added, bench, df.attrs.get('data_end')
                                         or max(tr.index[-1] for tr in bench.values()))
        a, b = survivorship_tables(per_year, summary)
        parts += ['## Survivorship: the universe against the index as it was', a, b]
        df['InIndex'] = df['Ticker'].map(added) <= df['AsOf']
        stay = df[df['InIndex']]
        parts += ['## Verdict groups, companies in the index at the time only',
                  horizon_table(stay, 'Verdict', VERDICT_ORDER)]
        parts += ['## Price-to-value quintiles, margin measured, in the index at the time only',
                  quintile_table(add_quintiles(stay, measured_only=True))]
    except Exception as exc:  # noqa: BLE001 - network: the main tables stand alone
        parts += ['## Survivorship', f'not measured: {type(exc).__name__}: {exc}']
    return '\n\n'.join(parts) + '\n'


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Point-in-time backtest of the valuation tracker')
    ap.add_argument('--start', type=int, default=FIRST_YEAR, help='first as-of year-end')
    ap.add_argument('--end', type=int, default=LAST_YEAR, help='last as-of year-end')
    ap.add_argument('--tickers', help="comma-separated tickers instead of today's S&P 500")
    ap.add_argument('--limit', type=int, help='only the first N companies')
    ap.add_argument('--root', type=Path, default=ROOT,
                    help='the tracker checkout whose caches are read (default: this one)')
    ap.add_argument('--work', type=Path, default=None,
                    help='downloads and results (default: <root>/data_cache/backtest)')
    ap.add_argument('--workers', type=int, default=4, help='processes evaluating companies')
    ap.add_argument('--refresh-prices', action='store_true', help='download prices again')
    ap.add_argument('--analyse-only', action='store_true',
                    help="redo the tables from the last run's rows")
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args(argv)
    verbose = not args.quiet
    paths = Paths(args.root, args.work)
    read_tracker_state(paths)
    rows_path = paths.work / 'backtest_rows.parquet'
    days = year_ends(args.start, args.end)
    t0 = time.time()

    if args.analyse_only:
        # The verdicts as evaluated; the returns recomputed from the prices.
        df = attach_returns(pd.read_parquet(rows_path), paths, data_end_of(paths))
    else:
        if args.tickers:
            tickers = [sources._norm_ticker(t) for t in args.tickers.split(',') if t.strip()]
        else:
            tickers = sources.index_constituents('sp500', verbose=verbose)
        print(f'=== backtest: {len(tickers)} tickers, as of '
              f'{days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}, config {CONFIG_VERSION} ===')
        ciks, unresolved = resolve(tickers, paths, verbose=verbose)
        if unresolved:
            print(f'  no CIK for {", ".join(unresolved)}')
        companies = one_per_company(tickers, ciks)
        if args.limit:
            companies = dict(list(companies.items())[:args.limit])
        print(f'  {len(companies)} companies ({len(tickers) - len(unresolved)} tickers resolved)')
        docs = ensure_documents(companies, paths, days[-1], verbose=verbose)
        download_prices(list(companies) + list(BENCHMARKS), paths,
                        refresh=args.refresh_prices, verbose=verbose)
        data_end = data_end_of(paths)
        series = load_macro(paths)
        macro = {d: macro_as_of(series, d) for d in days}
        if verbose:
            for d in (days[0], days[-1]):
                m = macro[d]
                print(f'  macro at {d:%Y-%m-%d}: AAA {m["aaa_yield"]:.2f}% ({m["aaa_as_of"]:%Y-%m}), '
                      f'CPI x{m["cpi_ratio"]:.2f} since 1973 ({m["cpi_as_of"]:%Y-%m})')
        # Every company, filings or not: one priced in a year it had no
        # annual report on file still belongs to that year's universe.
        jobs = [{'ticker': t, 'cik': cik, 'facts': docs.get(t, {}).get('facts', []),
                 'sub': docs.get(t, {}).get('sub'), 'prices': str(paths.prices / f'{t}.parquet'),
                 'class_store': str(classes.STORE_PATH), 'days': days, 'macro': macro}
                for t, cik in companies.items()]
        rows = run_evaluation(jobs, args.workers, verbose=verbose)
        df = pd.DataFrame(rows)
        for c in ('PriceDate', 'LastFY'):
            if c in df:
                df[c] = pd.to_datetime(df[c], errors='coerce')
        df = attach_returns(df, paths, data_end)
        df.to_parquet(rows_path)
        df.to_csv(paths.work / 'backtest_rows.csv', index=False)
        sources._write_json(paths.work / 'run.json', {
            'config': CONFIG_VERSION, 'run': pd.Timestamp.now().isoformat(),
            'data_end': f'{data_end:%Y-%m-%d}', 'days': [f'{d:%Y-%m-%d}' for d in days],
            'tickers': tickers, 'companies': companies})
        vc = df['Verdict'].value_counts()
        print('  verdicts: ' + ', '.join(f'{k} {v}' for k, v in vc.items()))

    tables = analyse(df, paths, verbose=verbose)
    out = paths.work / 'backtest_tables.md'
    out.write_text(tables, encoding='utf-8')
    print(f'  rows: {rows_path}\n  tables: {out}')
    print(f'=== done in {time.time() - t0:.0f}s ===')
    return 0


if __name__ == '__main__':
    sys.exit(main())
