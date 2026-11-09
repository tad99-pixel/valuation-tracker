"""Data sources: SEC XBRL fundamentals, FRED macro, and prices.

Fundamentals come from the SEC's own XBRL `companyfacts` API rather than from
yfinance. The reason is Graham: his earnings-stability test needs ten years of
annual figures and his dividend test twenty, while yfinance serves four years
of annual statements. EDGAR carries everything the company has filed in XBRL,
which is roughly 2009 onward, and it is the primary document rather than a
vendor's copy of it.

Prices are this package's own cache, deliberately not the scanner's: the
scanner's closes are dividend-adjusted, which is right for a return model and
wrong for a historical market value (see `get_prices`).

What is fetched, and how often:

* the filing index (`submissions`) every `filings_check_days`. SEC serves no
  Last-Modified or ETag, so the index is the only way to learn that a company
  has reported, and it is a twentieth the size of the facts;
* the companyfacts document only when that index lists a 10-K or 10-Q newer
  than the cached copy, or once the copy is `facts_max_age_days` old;
* prices incrementally: the last few sessions, appended to the cached history
  unless a split has restated it since;
* split histories weekly, forty tickers to a request.

Access etiquette: SEC asks automated clients to identify themselves and to
stay under ten requests a second. The User-Agent names this tool, and every
SEC request goes through one throttle.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from io import StringIO

import numpy as np
import pandas as pd
import requests

from .config import FACTS_DIR, MACRO_DIR, STATE_DIR, ValuationConfig

warnings.filterwarnings('ignore')

# SEC asks automated clients to identify themselves, ideally with a contact.
# If SEC starts refusing requests (403), set VALUATION_USER_AGENT - on GitHub,
# a repository secret - to something like "ValuationTracker/1.0 you@example.com".
UA = {'User-Agent': os.environ.get('VALUATION_USER_AGENT')
      or 'ValuationTracker/1.0 (non-commercial research script)'}

SEC_TICKER_TXT = 'https://www.sec.gov/include/ticker.txt'
SEC_FACTS = 'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json'
SEC_SUBMISSIONS = 'https://data.sec.gov/submissions/CIK{cik:010d}.json'
FRED_CSV = 'https://fred.stlouisfed.org/graph/fredgraph.csv'

WIKI_CIK_URLS = [
    'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
    'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies',
]

CIK_MAP_PATH = STATE_DIR / 'cik_map.json'
# Hand-maintained corrections, read last and allowed to win. A ticker that
# EDGAR maps to a holding company with no XBRL, or a dual-listing whose
# filings sit under the other line, is fixed here rather than in code.
CIK_OVERRIDE_PATH = STATE_DIR / 'cik_overrides.json'

_last_sec_call = [0.0]


def _sec_get(url: str, timeout: float, pause: float = 0.12):
    """Every SEC request, spaced at least `pause` seconds from the last."""
    wait = pause - (time.monotonic() - _last_sec_call[0])
    if wait > 0:
        time.sleep(wait)
    try:
        return requests.get(url, headers=UA, timeout=timeout)
    finally:
        _last_sec_call[0] = time.monotonic()


def _read_json(path, default):
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001 - missing or unreadable is the same here
        return default


def _write_json(path, obj) -> None:
    try:
        path.write_text(json.dumps(obj))
    except Exception:  # noqa: BLE001
        pass


def _write_json_gz(path, obj) -> str:
    """Write `obj` as gzip, atomically and deterministically; return its sha1.

    No timestamp in the gzip header (``mtime=0``): the same document must
    produce the same bytes, because the hash is what the panel cache keys on.
    """
    data = gzip.compress(json.dumps(obj, separators=(',', ':')).encode('utf-8'),
                         compresslevel=6, mtime=0)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return hashlib.sha1(data).hexdigest()


# ---------------------------------------------------------------------------
# ticker -> CIK
# ---------------------------------------------------------------------------

def _norm_ticker(t: str) -> str:
    return str(t).strip().upper().replace('.', '-')


def _read_cik_cache():
    """``(map, fetched, complete)`` from disk, or None."""
    blob = _read_json(CIK_MAP_PATH, None)
    if not blob:
        return None
    try:
        m = {k: int(v) for k, v in blob['map'].items()}
        fetched = datetime.fromisoformat(blob['fetched'])
    except Exception:  # noqa: BLE001
        return None
    # A map written before the flag existed came from ticker.txt if it is that
    # large; the index-table fallback tops out near 1,100 names.
    return m, fetched, bool(blob.get('complete', len(m) > 5000))


def _sec_ticker_map(verbose: bool) -> dict:
    out = {}
    try:
        r = _sec_get(SEC_TICKER_TXT, timeout=30)
        r.raise_for_status()
        for line in r.text.splitlines():
            parts = line.split('\t')
            if len(parts) == 2 and parts[1].strip().isdigit():
                out[_norm_ticker(parts[0])] = int(parts[1])
        if verbose:
            print(f'  cik map: {len(out)} tickers from SEC ticker.txt')
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f'  cik map: SEC ticker.txt unavailable ({exc}); trying index tables')
    return out


def _index_table_ciks(verbose: bool) -> dict:
    out = {}
    for url in WIKI_CIK_URLS:
        try:
            resp = requests.get(url, headers={'User-Agent': 'ValuationTracker/1.0'}, timeout=30)
            resp.raise_for_status()
            for tbl in pd.read_html(StringIO(resp.text)):
                cols = {str(c): c for c in tbl.columns}
                sym = cols.get('Symbol') or cols.get('Ticker')
                cik = next((v for k, v in cols.items() if k.upper() == 'CIK'), None)
                if sym is None or cik is None:
                    continue
                for s, c in zip(tbl[sym], tbl[cik]):
                    try:
                        out.setdefault(_norm_ticker(s), int(c))
                    except (TypeError, ValueError):
                        continue
                break
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f'  cik map: {url.rsplit("/", 1)[-1]} failed ({exc})')
    return out


def load_cik_map(refresh_days: int = 14, verbose: bool = True, force: bool = False) -> dict:
    """Return {TICKER: cik}, cached on disk.

    `include/ticker.txt` is the full official map and the only route here that
    covers the whole market. `files/company_tickers.json` returns 403 from some
    networks, so the Wikipedia index tables - which carry a CIK column for the
    S&P 500 and S&P 600 - are kept as a partial fallback.

    The cache records whether it is complete (the SEC file) or partial (the
    index tables), and a partial map never replaces a complete one. It used
    to: one afternoon with ticker.txt unreachable overwrote ten thousand
    tickers with eleven hundred, and every company outside the two indexes
    read "no CIK" for the next fortnight.
    """
    cache = _read_cik_cache()
    now = datetime.now(timezone.utc)
    if cache and not force and cache[2] and now - cache[1] < timedelta(days=refresh_days):
        return _with_overrides(cache[0])

    out = _sec_ticker_map(verbose)
    complete = len(out) >= 1000
    if not complete:
        for k, v in _index_table_ciks(verbose).items():
            out.setdefault(k, v)

    if not complete and cache and cache[2]:
        if verbose:
            print('  cik map: keeping the complete cached map; the partial one only adds to it')
        return _with_overrides({**out, **cache[0]})
    if not out:
        if cache:
            if verbose:
                print('  cik map: all sources failed; using the stale cached map')
            return _with_overrides(cache[0])
        raise RuntimeError(
            'Could not build a ticker -> CIK map from SEC ticker.txt or the '
            'index tables. Fundamentals cannot be fetched without it. Check '
            f'connectivity, or write a map by hand to {CIK_OVERRIDE_PATH}.')
    _write_json(CIK_MAP_PATH, {'fetched': now.isoformat(), 'complete': complete, 'map': out})
    return _with_overrides(out)


def _with_overrides(m: dict) -> dict:
    if CIK_OVERRIDE_PATH.exists():
        try:
            ov = json.loads(CIK_OVERRIDE_PATH.read_text())
            m = {**m, **{_norm_ticker(k): int(v) for k, v in ov.items()}}
        except Exception as exc:  # noqa: BLE001
            print(f'  cik overrides unreadable ({exc}); ignored')
    return m


def resolve_ciks(tickers, verbose: bool = True) -> tuple:
    """Split `tickers` into {ticker: cik} and a list of unresolved ones.

    A ticker missing from a cached map is usually a listing or a ticker
    change newer than the map, so a miss refreshes it - at most once a day,
    so a ticker that genuinely has no CIK does not cost a download per run.
    """
    want = list(dict.fromkeys(_norm_ticker(t) for t in tickers))
    m = load_cik_map(verbose=verbose)
    if any(k not in m for k in want):
        cache = _read_cik_cache()
        if cache is None or datetime.now(timezone.utc) - cache[1] > timedelta(days=1):
            m = load_cik_map(verbose=verbose, force=True)
    found = {k: m[k] for k in want if k in m}
    return found, [k for k in want if k not in m]


# ---------------------------------------------------------------------------
# SEC filing index and companyfacts
# ---------------------------------------------------------------------------

# The reports whose arrival means the companyfacts document has new facts.
PERIODIC_FORMS = frozenset({'10-K', '10-K/A', '10-Q', '10-Q/A', '10-KT', '10-KT/A',
                            '20-F', '20-F/A', '40-F', '40-F/A'})
# Facts are trusted this long when there is no filing date to compare them to.
FACTS_BLIND_DAYS = 7


def _facts_path(cik: int):
    return FACTS_DIR / f'CIK{cik:010d}.json.gz'


def _facts_meta_path(cik: int):
    return FACTS_DIR / f'CIK{cik:010d}.meta.json'


def _sub_path(cik: int):
    return FACTS_DIR / f'SUB{cik:010d}.json.gz'


def _latest_filed(blob: dict):
    """The newest `filed` date anywhere in a companyfacts document."""
    best = ''
    for taxonomy in (blob.get('facts') or {}).values():
        for node in taxonomy.values():
            for rows in (node.get('units') or {}).values():
                for r in rows:
                    f = r.get('filed')
                    if f and f > best:
                        best = f
    return best or None


def _facts_meta(cik: int) -> dict:
    """The sidecar of a cached document: when fetched, newest filing, hash.

    A document cached before sidecars existed gets one built from the file:
    its hash, its file time as the fetch time, and an unknown newest filing,
    which the freshness rule treats conservatively.
    """
    p = _facts_path(cik)
    if not p.exists():
        return {}
    meta = _read_json(_facts_meta_path(cik), {})
    if meta.get('sha1') and meta.get('fetched'):
        return meta
    try:
        meta = {'fetched': datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
                'latest_filed': None, 'sha1': hashlib.sha1(p.read_bytes()).hexdigest()}
    except OSError:
        return {}
    _write_json(_facts_meta_path(cik), meta)
    return meta


def facts_current(meta: dict, latest_filing, cfg: ValuationConfig) -> tuple:
    """Is a cached companyfacts document up to date? ``(bool, reason)``."""
    if not meta:
        return False, 'not cached'
    fetched = datetime.fromisoformat(meta['fetched'])
    age = (datetime.now(timezone.utc) - fetched).total_seconds() / 86400.0
    if age > cfg.facts_max_age_days:
        return False, f'{age:.0f} days old'
    if not latest_filing:
        return age <= FACTS_BLIND_DAYS, 'no filing date to compare against'
    if meta.get('latest_filed') and meta['latest_filed'] >= latest_filing:
        return True, 'includes the latest report'
    # Fetched comfortably after the latest report: if that report added no
    # facts (an amendment to the cover page, say), waiting for a fact dated
    # on it would refetch the document on every run, forever.
    if fetched.date() >= (pd.Timestamp(latest_filing) + pd.Timedelta(days=2)).date():
        return True, 'fetched after the latest report'
    return False, f'the {latest_filing} report is newer than the cached facts'


class FactsRef:
    """A companyfacts document on disk, parsed only if a panel must be built.

    The parse is most of the cost of a cached run - a large filer's document
    is ~20 MB of JSON - and a panel built from an unchanged document can be
    reused instead (`fundamentals.cached_entry`), keyed on `sha1`.
    """

    __slots__ = ('cik', 'source', 'sha1', '_doc')

    def __init__(self, cik: int, source: str, sha1, doc=None):
        self.cik, self.source, self.sha1, self._doc = cik, source, sha1, doc

    def load(self) -> dict:
        if self._doc is not None:
            return self._doc
        with gzip.open(_facts_path(self.cik), 'rt', encoding='utf-8') as fh:
            return json.load(fh)


def fetch_companyfacts(cik: int, cfg: ValuationConfig, force: bool = False,
                       latest_filing=None) -> tuple:
    """Make the cached companyfacts document current: ``(FactsRef | None, source)``.

    `source` is 'cache' (current), 'network' (just fetched), 'stale-cache' (a
    refetch was due and failed, so the copy predates the filer's latest
    report), 'not-filed' (no XBRL at all), or 'failed: ...'. The caller
    records it, so a run that fell back to an out-of-date copy says so.
    """
    meta = _facts_meta(cik)
    if not force and facts_current(meta, latest_filing, cfg)[0]:
        return FactsRef(cik, 'cache', meta['sha1']), 'cache'
    try:
        r = _sec_get(SEC_FACTS.format(cik=cik), timeout=60, pause=cfg.sec_pause)
        if r.status_code == 404:
            return None, 'not-filed'
        r.raise_for_status()
        blob = r.json()
    except Exception as exc:  # noqa: BLE001
        if meta:
            print(f'  CIK{cik}: network failed ({exc}); using the copy fetched '
                  f'{meta["fetched"][:10]}')
            return FactsRef(cik, 'stale-cache', meta['sha1']), 'stale-cache'
        return None, f'failed: {exc}'
    sha1 = None
    try:
        sha1 = _write_json_gz(_facts_path(cik), blob)
        _write_json(_facts_meta_path(cik), {
            'fetched': datetime.now(timezone.utc).isoformat(),
            'latest_filed': _latest_filed(blob), 'sha1': sha1})
    except Exception as exc:  # noqa: BLE001
        print(f'  facts cache write failed for CIK{cik}: {exc}')
    return FactsRef(cik, 'network', sha1, blob), 'network'


def fetch_submissions(cik: int, cfg: ValuationConfig, force: bool = False):
    """The filer's index entry: SIC code, entity type, fiscal year end, and the
    date of its latest periodic report.

    The SIC code drives the bank / insurer / REIT / utility carve-outs, and
    taking it from the filer's own submission is better than inferring an
    industry from the ticker. The latest 10-K or 10-Q date decides whether
    the companyfacts document needs refetching.
    """
    p = _sub_path(cik)
    cached = None
    if p.exists():
        try:
            with gzip.open(p, 'rt', encoding='utf-8') as fh:
                cached = json.load(fh)
        except Exception:  # noqa: BLE001
            cached = None
    # A record from before `latest_filing` was kept cannot answer the refetch
    # question, so it is refreshed whatever its age.
    if (cached is not None and not force and 'latest_filing' in cached
            and (time.time() - p.stat().st_mtime) / 86400.0 <= cfg.filings_check_days):
        return cached
    try:
        r = _sec_get(SEC_SUBMISSIONS.format(cik=cik), timeout=45, pause=cfg.sec_pause)
        r.raise_for_status()
        j = r.json()
    except Exception:  # noqa: BLE001
        return cached
    recent = (j.get('filings') or {}).get('recent') or {}
    periodic = [d for f, d in zip(recent.get('form') or [], recent.get('filingDate') or [])
                if f in PERIODIC_FORMS and d]
    slim = {
        'cik': cik,
        'name': j.get('name'),
        'sic': j.get('sic'),
        'sicDescription': j.get('sicDescription'),
        'fiscalYearEnd': j.get('fiscalYearEnd'),
        'tickers': j.get('tickers', []),
        'exchanges': j.get('exchanges', []),
        'entityType': j.get('entityType'),
        'latest_filing': max(periodic) if periodic else None,
    }
    try:
        _write_json_gz(p, slim)
    except Exception:  # noqa: BLE001
        pass
    return slim


def fetch_many(tickers, cfg: ValuationConfig, force: bool = False,
               verbose: bool = True, consume=None, prebuilt=None) -> dict:
    """Fetch the filing index and facts for several tickers, throttled for the SEC.

    Returns ``{ticker: result}``, omitting anything that could not be resolved
    or fetched. Unresolved tickers are printed, not swallowed - a watchlist
    silently losing a name is worse than a slow run.

    ``prebuilt(ticker, ref, meta)`` is asked first and may return a finished
    result (a cached panel built from the same document), in which case the
    document is never parsed. Otherwise the document is read and handed to
    ``consume(ticker, blob)``, and only the callback's return value is kept: a
    companyfacts document expands to about 20 MB in memory, and holding all of
    them measured 1.1 GB at 49 companies - a MemoryError for the S&P 500.
    """
    ciks, missing = resolve_ciks(tickers, verbose=verbose)
    if missing and verbose:
        print(f'  no CIK for {len(missing)} ticker(s): {", ".join(missing[:12])}'
              + (' ...' if len(missing) > 12 else ''))

    out, failed, declined = {}, [], []
    seen = {'network': 0, 'cache': 0, 'stale-cache': 0, 'reused': 0}
    for i, (t, cik) in enumerate(ciks.items(), 1):
        if verbose and i % 25 == 0:
            print(f'    fundamentals {i}/{len(ciks)}', end='\r')
        meta = fetch_submissions(cik, cfg, force=force)
        if meta is None:
            # The SIC code decides the bank / insurer / REIT carve-outs.
            # Evaluating a bank as an operating company would fail it on
            # tests Graham never meant for it, so it is skipped this run.
            failed.append(f'{t} (filing index unavailable - SIC unknown)')
            continue
        etype = str(meta.get('entityType') or '').lower()
        if etype and etype != 'operating':
            # A fund or trust holds other securities: there is no income
            # statement to test, and a look-through needs holdings data this
            # tracker does not fetch. SEC's own entity type says so; matching
            # "ETF" or "TRUST" in a name both missed funds and caught REITs.
            declined.append(f'{t} ({etype})')
            continue
        ref, source = fetch_companyfacts(cik, cfg, force=force,
                                         latest_filing=meta.get('latest_filing'))
        if ref is None:
            failed.append(f'{t} ({source})')
            continue
        seen[source] += 1
        if prebuilt is not None:
            hit = prebuilt(t, ref, meta)
            if hit is not None:
                out[t] = hit
                seen['reused'] += 1
                continue
        try:
            facts = ref.load()
        except Exception as exc:  # noqa: BLE001
            # Corrupt on disk: drop the sidecar so the next run refetches it.
            _facts_meta_path(cik).unlink(missing_ok=True)
            failed.append(f'{t} (cached facts unreadable: {exc})')
            continue
        blob = {'cik': cik, 'facts': facts, 'meta': meta, 'source': source,
                'sha1': ref.sha1}
        out[t] = consume(t, blob) if consume is not None else blob
        del facts, blob, ref
    if verbose:
        print(' ' * 60, end='\r')
        stale = seen['stale-cache']
        print(f'  fundamentals: {len(out)} companies ({seen["network"]} fetched, '
              f'{seen["cache"] + stale} cached, {seen["reused"]} panels reused'
              + (f', {stale} STALE' if stale else '') + ')')
        if declined:
            print(f'  declined {len(declined)} non-operating filer(s): {", ".join(declined[:8])}')
        if failed:
            print(f'  no XBRL facts for {len(failed)}: {", ".join(failed[:8])}'
                  + (' ...' if len(failed) > 8 else ''))
    return out


# ---------------------------------------------------------------------------
# Macro: AAA yield for Graham's relative tests, CPI for his dollar thresholds
# ---------------------------------------------------------------------------

def _fred_series(series_id: str, max_age_days: int = 3):
    """One FRED series as a Series indexed by date, cached to parquet."""
    p = MACRO_DIR / f'{series_id}.parquet'
    if p.exists():
        try:
            if (time.time() - p.stat().st_mtime) / 86400.0 <= max_age_days:
                s = pd.read_parquet(p)
                return s[s.columns[0]]
        except Exception:  # noqa: BLE001
            pass
    try:
        r = requests.get(FRED_CSV, params={'id': series_id}, headers=UA, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        date_col, val_col = df.columns[0], df.columns[1]
        df[date_col] = pd.to_datetime(df[date_col])
        df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
        s = df.dropna().set_index(date_col)[val_col]
        s.name = series_id
        try:
            s.to_frame().to_parquet(p)
        except Exception:  # noqa: BLE001
            pass
        return s
    except Exception:  # noqa: BLE001
        if p.exists():
            try:
                s = pd.read_parquet(p)
                return s[s.columns[0]]
            except Exception:  # noqa: BLE001
                pass
        return None


def get_macro(cfg: ValuationConfig, verbose: bool = True) -> dict:
    """AAA corporate yield, 10y Treasury, and the CPI ratio since Graham's edition.

    The AAA yield is not decoration: three of Graham's 1976 criteria are
    expressed relative to it, and his 1974 revision of the growth formula
    divides by it. Hard-coding 4.4% - the yield when he wrote - would price
    every stock as though it were 1962.

    No silent defaults. An unavailable AAA yield is None and Graham's relative
    tests report themselves unmeasurable; an unavailable CPI is None too. An
    earlier version substituted 1.0 for the CPI ratio, which is truthy, so the
    size test silently ran in 1973 dollars and passed every company over $100m
    of revenue.
    """
    aaa = _fred_series('AAA')
    dgs10 = _fred_series('DGS10')
    cpi = _fred_series('CPIAUCSL', max_age_days=20)

    out = {
        'aaa_yield': float(aaa.iloc[-1]) if aaa is not None and len(aaa) else None,
        'treasury_10y': float(dgs10.iloc[-1]) if dgs10 is not None and len(dgs10) else None,
        'cpi_ratio': None,
        'cpi_base_year': cfg.graham.base_year,
    }
    if cpi is not None and len(cpi):
        base = cpi[cpi.index.year == cfg.graham.base_year]
        if len(base):
            out['cpi_ratio'] = float(cpi.iloc[-1] / base.mean())

    if verbose:
        aaa_txt = f'{out["aaa_yield"]:.2f}%' if out['aaa_yield'] is not None else 'unavailable'
        cpi_txt = (f'CPI x{out["cpi_ratio"]:.1f} since {cfg.graham.base_year}'
                   if out['cpi_ratio'] is not None else 'CPI unavailable')
        print(f'  macro: AAA {aaa_txt}, 10y '
              f'{out["treasury_10y"] if out["treasury_10y"] else "n/a"}, {cpi_txt}')
    return out


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

# Deliberately NOT the scanner's cache. The scanner downloads with
# auto_adjust=True, which back-adjusts history for dividends as well as
# splits - correct for a return-based model, wrong for valuation. Measured on
# November 2016 closes: Coca-Cola reads 30.89 against an actual 42.12 (-27%),
# Exxon 54.31 against 83.65 (-35%). Any test that needs a *historical market
# value* - Buffett's one-dollar premise, Graham's five-year P/E high - was
# being fed a price a third too low, flattering both. Valuation needs prices
# adjusted for splits (so they share a basis with the restated EPS) and for
# nothing else.
VAL_PRICE_DIR = MACRO_DIR.parent / 'prices_split_only'
VAL_PRICE_DIR.mkdir(parents=True, exist_ok=True)
# Tickers whose full download began later than requested: the listing starts
# there, so a shorter history is complete rather than truncated.
PRICE_STARTS_PATH = VAL_PRICE_DIR / '_history_start.json'

# A cached close older than this is not a current price. Walgreens, taken
# private in 2025, kept being valued on its last cached quote because the
# fallback had no age limit; past this, the company gets NO PRICE.
MAX_PRICE_STALENESS_DAYS = 10
# Sessions re-downloaded on an incremental update, to confirm the cached
# history is still on the provider's current split basis.
PRICE_OVERLAP_DAYS = 10


def _val_cache_path(t: str):
    return VAL_PRICE_DIR / f'{t.replace("/", "_")}.parquet'


def _read_price_cache(t: str):
    try:
        df = pd.read_parquet(_val_cache_path(t))
        df.index = pd.DatetimeIndex(df.index)
        return df if len(df) else None
    except Exception:  # noqa: BLE001 - missing or unreadable
        return None


def _save_price_cache(t: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_val_cache_path(t))
    except Exception:  # noqa: BLE001
        pass


def _last_close_date() -> pd.Timestamp:
    """The most recent US session whose close has passed, by the New York clock.

    Freshness used to be a count of calendar days from the local date, which
    only works for a run made before the US close: run after it - as a
    scheduled cloud job is - yesterday's cached close was "current" and
    today's was never fetched. Weekends are skipped; holidays are not known,
    and cost only a small download that finds no new bar.
    """
    now = pd.Timestamp.now(tz='America/New_York')
    d = now.normalize().tz_localize(None)
    if (now.hour, now.minute) < (16, 30):
        d -= pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def _drop_live_bar(df: pd.DataFrame) -> pd.DataFrame:
    """Drop today's bar while the US session is still open.

    Run during US trading hours (evening in Singapore), the last bar is a live
    quote. Cached, it would be served the next day as that day's close.
    """
    if df.empty:
        return df
    try:
        now_ny = pd.Timestamp.now(tz='America/New_York')
    except Exception:  # noqa: BLE001
        return df
    if df.index[-1].normalize() == now_ny.tz_localize(None).normalize() and (
            now_ny.hour, now_ny.minute) < (16, 30):
        return df.iloc[:-1]
    return df


def _ticker_frame(raw, t: str):
    """One ticker's columns from a `yf.download` result, or None."""
    if raw is None or getattr(raw, 'empty', True):
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        if t not in raw.columns.get_level_values(0):
            return None
        return raw[t]
    return raw


def _extract_close(raw, t: str):
    sub = _ticker_frame(raw, t)
    if sub is None or 'Close' not in sub:
        return None
    sub = sub[['Close']].dropna()
    sub = sub[sub['Close'] > 0]
    if sub.empty:
        return None
    sub.index = pd.DatetimeIndex(sub.index)
    if sub.index.tz is not None:
        sub.index = sub.index.tz_localize(None)
    sub = _drop_live_bar(sub[~sub.index.duplicated(keep='last')].sort_index())
    return None if sub.empty else sub


def merge_incremental(old: pd.DataFrame, new: pd.DataFrame):
    """`old` extended by `new`, or None if the provider has restated history.

    yfinance restates every earlier close when a stock splits. Bars fetched
    after a split appended to a history cached before it would read as a
    crash, or a surge, of exactly the split ratio - so the overlapping
    sessions are compared first, and any disagreement beyond rounding sends
    the ticker back for a full download.
    """
    common = old.index.intersection(new.index)
    if len(common) == 0:
        return None
    drift = (new.loc[common, 'Close'] / old.loc[common, 'Close'] - 1).abs()
    if float(drift.max()) > 0.005:
        return None
    merged = pd.concat([old[old.index < new.index[0]], new])
    return merged[~merged.index.duplicated(keep='last')].sort_index()


def get_prices(tickers, years: int = 12, force: bool = False,
               verbose: bool = True) -> dict:
    """Split-adjusted, dividend-unadjusted daily closes, cached separately.

    Returns ``{ticker: DataFrame}`` with at least a ``Close`` column. Ten-plus
    years are requested because the one-dollar premise compares market value
    across a decade; a shorter history makes that test unmeasurable rather
    than approximate.

    A cached history is extended, not replaced: only the last few sessions are
    downloaded and appended, unless they show the history was restated.
    """
    import yfinance as yf
    tickers = list(dict.fromkeys(_norm_ticker(t) for t in tickers))
    today = pd.Timestamp.today().normalize()
    start = today - pd.DateOffset(years=years)
    last_close = _last_close_date()
    starts = _read_json(PRICE_STARTS_PATH, {})
    out, full, incr, cached = {}, [], {}, {}

    for t in tickers:
        df = None if force else _read_price_cache(t)
        if df is None:
            full.append(t)
            continue
        cached[t] = df
        # A 2019 listing has no 2014 prices to fetch; it used to be
        # re-downloaded in full on every run looking for them.
        long_enough = (df.index[0] <= start + pd.Timedelta(days=10)
                       or starts.get(t) == df.index[0].strftime('%Y-%m-%d'))
        if not long_enough:
            full.append(t)
        elif df.index[-1] >= last_close:
            out[t] = df.loc[df.index >= start]
        else:
            incr[t] = df.index[-1]

    def fetch(names, since_of) -> dict:
        got = {}
        for i in range(0, len(names), 40):
            batch = names[i:i + 40]
            raw = None
            # One retry, slower: a transient failure would otherwise drop up
            # to forty companies to NO PRICE in a single run.
            for threads in (True, False):
                try:
                    raw = yf.download(batch, start=since_of(batch).strftime('%Y-%m-%d'),
                                      auto_adjust=False, actions=False, progress=False,
                                      group_by='ticker', threads=threads,
                                      timeout=30 if threads else 60)
                    break
                except Exception as exc:  # noqa: BLE001
                    if verbose:
                        print(f'  price batch failed ({exc})'
                              + ('; retrying slower' if threads else ''))
                    if threads:
                        time.sleep(10)
            for t in batch:
                sub = _extract_close(raw, t)
                if sub is not None:
                    got[t] = sub
            time.sleep(0.6)
        return got

    if verbose and (full or incr):
        print(f'  prices: {len(incr)} extended, {len(full)} downloaded in full, '
              f'{len(tickers) - len(full) - len(incr)} current - split-adjusted only')

    if incr:
        names, restated = sorted(incr), []
        got = fetch(names, lambda b: min(incr[t] for t in b)
                    - pd.Timedelta(days=PRICE_OVERLAP_DAYS))
        for t in names:
            new = got.get(t)
            if new is None:
                continue                    # network: the stale fallback decides
            merged = merge_incremental(cached[t], new)
            if merged is None:
                restated.append(t)          # restated by a split: refetch in full
                continue
            _save_price_cache(t, merged)
            out[t] = merged.loc[merged.index >= start]
        if restated:
            full.extend(restated)
            if verbose:
                print(f'  prices: history restated since cached for {", ".join(restated[:8])} '
                      '(a split?) - downloading in full')

    if full:
        got = fetch(full, lambda b: start)
        for t in full:
            sub = got.get(t)
            if sub is None:
                continue
            _save_price_cache(t, sub)
            if sub.index[0] > start + pd.Timedelta(days=10):
                starts[t] = sub.index[0].strftime('%Y-%m-%d')
            else:
                starts.pop(t, None)
            out[t] = sub.loc[sub.index >= start]
        _write_json(PRICE_STARTS_PATH, starts)

    # A recently cached close beats no price when the network fails; the
    # evaluation stamps the price date on every row so a degraded run is
    # visible. Past MAX_PRICE_STALENESS_DAYS it is not a price at all.
    for t in tickers:
        if t in out:
            continue
        df = cached.get(t)
        if df is None:
            df = _read_price_cache(t)
        if df is not None and (today - df.index[-1]).days <= MAX_PRICE_STALENESS_DAYS:
            out[t] = df.loc[df.index >= start]
    missing = [t for t in tickers if t not in out]
    if verbose and missing:
        print(f'  prices: none for {len(missing)} ({", ".join(missing[:8])}) '
              '- delisted, acquired, or renamed')
    return out


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

SPLITS_PATH = MACRO_DIR / 'splits.parquet'
SPLITS_FETCHED_PATH = MACRO_DIR / 'splits_fetched.json'
# Splits before this date never touch a fact: XBRL filings begin in 2009, and
# a fact is restated only for the splits after its own filing date.
SPLITS_SINCE = '2008-01-01'


def _as_split_series(s) -> pd.Series:
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(s).astype(float)
    idx = pd.DatetimeIndex(s.index)
    s.index = idx.tz_localize(None) if idx.tz is not None else idx
    s = s[s > 0]
    return s[s.index >= pd.Timestamp(SPLITS_SINCE)].sort_index()


def _splits_from_download(raw, t: str):
    """The split events for `t` in a batch download, or None if `t` has no rows.

    A ticker that failed inside a batch comes back as an all-NaN column, and
    reading that as "never split" is the error this module exists to avoid.
    Only a ticker with prices in the result has a trustworthy splits column.
    """
    sub = _ticker_frame(raw, t)
    if (sub is None or 'Close' not in sub or 'Stock Splits' not in sub
            or sub['Close'].notna().sum() == 0):
        return None
    s = sub['Stock Splits'].fillna(0.0)
    return _as_split_series(s[s != 0])


def _download_splits(tickers: list) -> tuple:
    """``({ticker: Series}, failed)`` - forty tickers to a request, with one
    direct retry for any ticker missing from its batch."""
    import yfinance as yf
    ok, failed = {}, []
    for i in range(0, len(tickers), 40):
        batch = tickers[i:i + 40]
        try:
            raw = yf.download(batch, start=SPLITS_SINCE, auto_adjust=False, actions=True,
                              progress=False, group_by='ticker', threads=True, timeout=30)
        except Exception:  # noqa: BLE001 - network, rate limit, parse
            raw = None
        for t in batch:
            s = _splits_from_download(raw, t)
            if s is None:
                try:
                    s = _as_split_series(yf.Ticker(t).splits)
                except Exception:  # noqa: BLE001
                    failed.append(t)
                    continue
            ok[t] = s
        if i + 40 < len(tickers):
            time.sleep(0.6)
    return ok, failed


def get_splits(tickers, max_age_days: int = 7, force: bool = False,
               verbose: bool = True) -> dict:
    """Split history per ticker: ``{ticker: Series}``, or ``None`` if unknown.

    This is not a convenience. A per-share figure filed before a split is
    stated on the old share basis, and EDGAR only restates the two or three
    comparative years that appear in a later annual report - so Apple's EPS
    history arrives on three different bases at once (pre-2014, post-7:1, and
    post-4:1 from 2020). The correction is applied per fact, using the splits
    after the date that fact was *filed*.

    Three cache rules, each the fix for a way an earlier version lied:

    * Freshness is tracked **per ticker** (``splits_fetched.json``), not by
      the shared file's mtime - any rewrite used to make every other ticker's
      rows look new, so a split could go unseen indefinitely.
    * A **failed fetch is never cached**. It was stored exactly like "never
      split", overwrote good rows, and was trusted for a week. Now a failure
      keeps the last verified history; a ticker that has never been fetched
      successfully comes back as ``None``, which `score` turns into "per-share
      figures withheld" rather than computing on mixed share bases.
    * Only tickers that are missing or stale are fetched, and ``force``
      refetches them all.
    """
    cached = None
    if SPLITS_PATH.exists():
        try:
            cached = pd.read_parquet(SPLITS_PATH)
        except Exception:  # noqa: BLE001
            cached = None
    fetched_at = _read_json(SPLITS_FETCHED_PATH, {}) if SPLITS_FETCHED_PATH.exists() else {}

    wanted = {_norm_ticker(t) for t in tickers}
    now = time.time()
    due = sorted(t for t in wanted
                 if force or now - float(fetched_at.get(t, 0)) > max_age_days * 86400)

    failed = []
    if due:
        ok, failed = _download_splits(due)
        rows = []
        for t, s in ok.items():
            fetched_at[t] = now
            if len(s) == 0:
                rows.append({'Ticker': t, 'Date': pd.NaT, 'Ratio': np.nan})
            else:
                rows.extend({'Ticker': t, 'Date': d, 'Ratio': float(v)} for d, v in s.items())
        new = pd.DataFrame(rows, columns=['Ticker', 'Date', 'Ratio'])
        keep = cached[~cached['Ticker'].isin(set(ok))] if cached is not None else None
        df = pd.concat([keep, new], ignore_index=True) if keep is not None else new
        try:
            df.to_parquet(SPLITS_PATH)
            SPLITS_FETCHED_PATH.write_text(json.dumps(fetched_at))
        except Exception:  # noqa: BLE001
            pass
        cached = df

    out = _splits_to_dict(cached if cached is not None else
                          pd.DataFrame(columns=['Ticker', 'Date', 'Ratio']), wanted)
    unknown = [t for t in wanted if t not in fetched_at]
    for t in unknown:
        out[t] = None
    if verbose:
        n_split = sum(1 for t in wanted if out.get(t) is not None and len(out[t]))
        msg = (f'  splits: {len(due) - len(failed)} fetched, {len(wanted) - len(due)} cached; '
               f'{n_split} have split since {SPLITS_SINCE[:4]}')
        if failed:
            msg += (f'; {len(failed)} fetch(es) failed ({", ".join(failed[:6])})'
                    + (f', {len(unknown)} with no verified history' if unknown else ''))
        print(msg)
    return out


def _splits_to_dict(df: pd.DataFrame, wanted: set) -> dict:
    out = {}
    since = pd.Timestamp(SPLITS_SINCE)
    for t in wanted:
        sub = df[(df['Ticker'] == t) & df['Ratio'].notna() & df['Date'].notna()]
        s = sub.set_index('Date')['Ratio'].sort_index() if len(sub) else pd.Series(dtype=float)
        s.index = pd.DatetimeIndex(s.index)
        # Rows cached before the cut-off was applied, trimmed the same way so
        # the panel cache sees one history whichever route fetched it.
        out[t] = s[s.index >= since].astype(float)
    return out
