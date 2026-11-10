"""Tests for the point-in-time backtest (backtest_valuation.py).

    python tests/test_backtest.py

No network access: synthetic filings, prices and FRED series. One test also
reads a real filing - Apple's companyfacts document - from the tracker's
cache when it is there (`data_cache/sec_facts`, or the checkout named by
BACKTEST_ROOT), and says so when it is skipped.

The central guarantee, that no fact filed after the as-of day reaches a
panel, is pinned in tests/test_valuation.py, which runs on every tracker
run; here it is checked again through the backtest's own worker, and on a
real document.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import backtest_valuation as BT  # noqa: E402
from test_valuation import _point_in_time_doc, _rewrite_after  # noqa: E402
from valuation import fundamentals as F  # noqa: E402
from valuation.config import DEFAULT as CFG  # noqa: E402


def _prices(start='1999-01-04', end='2020-12-31', dividend=0.5):
    """Daily closes rising 5% a year, a quarterly dividend, a 2:1 split in 2016."""
    idx = pd.bdate_range(start, end)
    years = (idx - idx[0]).days / 365.25
    px = pd.DataFrame({'Close': 20.0 * 1.05 ** years}, index=idx)
    px['Adj Close'] = px['Close']
    px['Dividends'] = 0.0
    px.loc[px.index[::63], 'Dividends'] = dividend
    px['Stock Splits'] = 0.0
    px.loc[pd.Timestamp('2016-06-01'), 'Stock Splits'] = 2.0
    return px


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------

def test_total_return_reinvests_dividends():
    idx = pd.bdate_range('2020-01-06', periods=5)
    px = pd.DataFrame({'Close': [100.0, 100.0, 99.0, 99.0, 110.0],
                       'Dividends': [0.0, 0.0, 1.0, 0.0, 0.0]}, index=idx)
    tr = BT.total_return_index(px)
    # The price fell by the dividend on the ex-date: the holder lost nothing.
    assert list(tr.round(12)) == [1.0, 1.0, 1.0, 1.0, round(110 / 99, 12)], list(tr)
    # A spin-off booked as a 1.319 split (the adjusted close does not drop)
    # and again as a $24.56 dividend - Danaher, 2016 - counts once; a regular
    # dividend on a split day still counts.
    spin = pd.DataFrame({'Close': [76.6, 76.0], 'Dividends': [0.0, 24.56],
                         'Stock Splits': [0.0, 1.319]}, index=idx[:2])
    assert abs(BT.total_return_index(spin).iloc[-1] - 76.0 / 76.6) < 1e-12
    regular = spin.assign(Dividends=[0.0, 0.5])
    assert abs(BT.total_return_index(regular).iloc[-1] - 76.5 / 76.6) < 1e-12
    print('  total return: a dividend reinvested at the ex-date close; a spin-off once')


def test_forward_returns_need_both_closes_and_a_finished_window():
    idx = pd.bdate_range('2014-01-01', '2016-06-30')
    tr = pd.Series(np.linspace(1.0, 2.0, len(idx)), index=idx)
    r = BT.forward_returns(tr, '2014-12-31', data_end=idx[-1], horizons=(1, 3))
    want = tr.loc[:'2015-12-31'].iloc[-1] / tr.loc[:'2014-12-31'].iloc[-1] - 1
    assert abs(r['TR1y'] - want) < 1e-12 and r['TR3y'] is None, r
    # A year-end on a weekend takes the Friday close.
    r = BT.forward_returns(tr, '2015-01-03', data_end=idx[-1], horizons=(1,))
    assert abs(r['TR1y'] - (tr.loc[:'2016-01-03'].iloc[-1] / tr.loc['2015-01-02'] - 1)) < 1e-12
    # No close near the start (not yet listed), or near the end (halted).
    assert BT.forward_returns(tr, '2013-12-31', data_end=idx[-1], horizons=(1,))['TR1y'] is None
    gap = tr.drop(tr.loc['2015-12-01':'2016-01-31'].index)
    assert BT.forward_returns(gap, '2014-12-31', data_end=idx[-1], horizons=(1,))['TR1y'] is None
    print('  forward returns: both closes, and a window that has ended')


# ---------------------------------------------------------------------------
# What the tracker could have seen that day
# ---------------------------------------------------------------------------

def test_macro_uses_only_months_published_by_the_day():
    months = pd.date_range('1973-01-01', '2016-12-01', freq='MS')
    aaa = pd.Series(np.arange(len(months), dtype=float) / 10, index=months)
    cpi = pd.Series(np.arange(len(months), dtype=float) + 40.0, index=months)
    t10 = pd.Series(2.0, index=pd.bdate_range('2010-01-01', '2016-12-30'))
    series = {'AAA': aaa, 'CPIAUCSL': cpi, 'DGS10': t10}
    m = BT.macro_as_of(series, '2014-12-31')
    assert m['aaa_as_of'] == pd.Timestamp('2014-11-01'), m      # December's average not complete
    assert m['cpi_as_of'] == pd.Timestamp('2014-11-01'), m      # published mid-December
    assert m['aaa_yield'] == aaa['2014-11-01']
    assert abs(m['cpi_ratio'] - cpi['2014-11-01'] / cpi[cpi.index.year == 1973].mean()) < 1e-12
    m = BT.macro_as_of(series, '2015-01-15')
    assert m['aaa_as_of'] == pd.Timestamp('2014-12-01'), m      # December complete
    assert m['cpi_as_of'] == pd.Timestamp('2014-11-01'), m      # December's CPI not out yet
    print('  macro: the AAA yield and CPI published by the day')


def test_prices_dividends_and_splits_are_cut_at_the_day():
    px = _prices()
    day = pd.Timestamp('2014-12-31')
    cut = BT.prices_as_of(px, day)
    assert cut.index[-1] <= day and cut.index[0] >= day - pd.DateOffset(years=BT.PRICE_YEARS)
    assert list(cut.columns) == ['Close']
    assert BT.prices_as_of(px, '1998-12-31') is None, 'priced before it listed'
    halted = px.drop(px.loc['2014-12-10':'2014-12-31'].index)
    assert BT.prices_as_of(halted, day) is None, 'a three-week-old close used as the price'
    pay = BT.payments_as_of(px, day)
    assert pay.index.max() <= day and (pay > 0).all() and pay.attrs['since'] == px.index[0]
    splits = BT.split_history(px)
    assert list(splits.index) == [pd.Timestamp('2016-06-01')] and splits.iloc[0] == 2.0
    print('  as of the day: closes, dividends cut there; splits for the share basis')


def test_price_frame_keeps_a_dividend_dated_on_a_day_without_a_close():
    idx = pd.bdate_range('2020-01-06', periods=4)
    raw = pd.DataFrame({'Close': [10.0, np.nan, 10.5, 11.0], 'Adj Close': [9.0, np.nan, 9.5, 10.0],
                        'Dividends': [0.0, 0.2, 0.0, 0.0], 'Stock Splits': [0.0] * 4}, index=idx)
    df = BT._price_frame(raw, 'X')
    assert len(df) == 3 and df['Dividends'].sum() == 0.2 and df.loc[idx[2], 'Dividends'] == 0.2
    print('  prices: a dividend on a row with no close moves to the next close')


# ---------------------------------------------------------------------------
# Portfolios
# ---------------------------------------------------------------------------

def test_portfolio_excess_is_per_year_then_averaged():
    rows = []
    for day, returns in (('2014-12-31', {'A': [0.1, 0.3], 'B': [0.0]}),
                         ('2015-12-31', {'A': [0.0], 'B': [0.2, 0.4]})):
        for verdict, rs in returns.items():
            rows += [{'AsOf': pd.Timestamp(day), 'Verdict': verdict, 'TR1y': r} for r in rs]
    df = pd.DataFrame(rows)
    p = BT.portfolios(df, 'Verdict', 1)
    y1 = p[p['AsOf'] == '2014-12-31'].set_index('Verdict')
    assert abs(y1.loc['A', 'excess'] - (0.2 - 0.4 / 3)) < 1e-12, y1
    s = BT.summarise_portfolios(p, 'Verdict', 1).set_index('Verdict')
    want_a = ((0.2 - 0.4 / 3) + (0.0 - 0.2)) / 2
    assert abs(s.loc['A', 'excess_ann'] - want_a) < 1e-12 and s.loc['A', 'beat_share'] == 0.5, s
    assert s.loc['A', 'years'] == 2 and s.loc['A', 'n_total'] == 3
    # Three years cumulative is annualised before it is compared.
    df3 = df.rename(columns={'TR1y': 'TR3y'})
    p3 = BT.portfolios(df3, 'Verdict', 3)
    a = p3[(p3['AsOf'] == '2014-12-31') & (p3['Verdict'] == 'A')].iloc[0]
    assert abs(a['excess'] - (1.2 ** (1 / 3) - (1 + 0.4 / 3) ** (1 / 3))) < 1e-12
    print('  portfolios: equal weights within a year, years averaged, horizons annualised')


def test_quintiles_rank_within_each_year_and_only_measured_margins():
    rows = []
    for day in ('2014-12-31', '2015-12-31'):
        for i in range(1, 11):
            rows.append({'AsOf': pd.Timestamp(day), 'PriceToValue': i * (2 if day > '2015' else 1),
                         'MoSVerdict': 'MEASURED', 'TR1y': 0.30 - 0.03 * i})
        rows.append({'AsOf': pd.Timestamp(day), 'PriceToValue': 0.1,
                     'MoSVerdict': 'UNRELIABLE', 'TR1y': -0.5})
    df = pd.DataFrame(rows)
    q = BT.add_quintiles(df)
    assert q['PVQuintile'].isna().sum() == 2, 'a withheld margin was ranked'
    first = q[q['AsOf'] == '2014-12-31'].set_index('PriceToValue')['PVQuintile']
    assert first[1] == first[2] == 1 and first[9] == first[10] == 5, first
    si = BT.spread_and_ic(q, 1)
    assert abs(si['ic'] - 1) < 1e-12 and si['spread'] > 0, si     # cheaper did better: +1
    every = BT.add_quintiles(df, measured_only=False)
    assert every['PVQuintile'].notna().all()
    # A measure where high is cheap (earnings yield) ranks the other way round
    # and reports the same sign.
    df['EY'] = 1 / df['PriceToValue']
    ey = BT.quintiles(df, 'EY', df['MoSVerdict'].eq('MEASURED'), cheap_low=False)
    assert (ey['PVQuintile'] == q['PVQuintile']).sum() == 20
    assert abs(BT.spread_and_ic(ey, 1)['ic'] - 1) < 1e-12
    print('  quintiles: within each year; withheld margins only when asked')


# ---------------------------------------------------------------------------
# Which filings are the company's
# ---------------------------------------------------------------------------

def test_a_predecessor_counts_until_its_successor_reports():
    """Alphabet filed its first 10-K the day Google Inc. filed its last, as a
    subsidiary. The predecessor is the record until then; after, its filings
    are not the company's, whichever wins a same-day tie."""
    from test_valuation import dur, facts_doc
    old = facts_doc({'NetIncomeLoss': {'USD': [
        dur('2013-01-01', '2013-12-31', 12.9e9, '2014-02-10'),
        dur('2014-01-01', '2014-12-31', 14.4e9, '2015-02-06'),
        dur('2015-01-01', '2015-12-31', 1.0, '2016-02-11')]}})
    new = facts_doc({'NetIncomeLoss': {'USD': [
        dur('2014-01-01', '2014-12-31', 14.1e9, '2016-02-11'),
        dur('2015-01-01', '2015-12-31', 16.3e9, '2016-02-11')]}})
    merged = BT.chain_facts([new, old])
    meta = {'sic': '7370'}
    p, _ = F.build_panel('GOOGL', merged, meta, CFG, splits=pd.Series(dtype=float))
    assert list(p['net_income']) == [12.9e9, 14.1e9, 16.3e9], list(p['net_income'])
    p, _ = F.build_panel('GOOGL', merged, meta, CFG, splits=pd.Series(dtype=float),
                         as_of='2015-12-31')
    assert list(p['net_income']) == [12.9e9, 14.4e9], list(p['net_income'])
    assert BT.chain_facts([new]) is new
    assert BT.family(1288776) == BT.family(1652044) == [1652044, 1288776]
    assert BT.family(1441634) == [1730168, 1649338, 1441634]
    assert BT.family(34088) == [2115436, 34088], 'the tracker\'s own successions first'
    assert BT.family(320193) == [320193]
    print('  predecessors: the record until the successor reports, not after')


def test_constituents_resolve_through_the_index_table():
    """SEC's ticker file lagged two tickers that changed hands: BNY pointed at
    a BlackRock municipal bond fund, DOC at Physicians Realty."""
    from valuation import sources as SRC
    table = ('<table><tr><th>Symbol</th><th>Security</th><th>CIK</th><th>Date added</th></tr>'
             '<tr><td>BNY</td><td>BNY Mellon</td><td>0001390777</td><td>1995-03-31</td></tr>'
             '<tr><td>KO</td><td>Coca-Cola</td><td>0000021344</td><td>1957-03-04</td></tr></table>')
    real = (SRC.STATE_DIR, SRC.CIK_MAP_PATH, SRC.CIK_OVERRIDE_PATH, BT.classes.STORE_PATH)
    try:
        with tempfile.TemporaryDirectory() as d:
            paths = BT.Paths(root=d, work=Path(d) / 'work')
            paths.state.mkdir(parents=True, exist_ok=True)
            (paths.state / 'cik_map.json').write_text(json.dumps({
                'fetched': '2026-11-09T00:00:00+00:00', 'complete': True,
                'map': {'BNY': 1137390, 'KO': 21344, 'ZZZ': 99}}))
            (paths.membership / 'sp500_wikipedia.html').write_text(table, encoding='utf-8')
            BT.read_tracker_state(paths)
            found, missing = BT.resolve(['BNY', 'KO', 'ZZZ', 'NONE'], paths, verbose=False)
            added = BT.date_added(paths)
    finally:
        SRC.STATE_DIR, SRC.CIK_MAP_PATH, SRC.CIK_OVERRIDE_PATH, BT.classes.STORE_PATH = real
    assert found == {'BNY': 1390777, 'KO': 21344, 'ZZZ': 99} and missing == ['NONE'], found
    assert added['KO'] == pd.Timestamp('1957-03-04')
    print('  CIKs: the index table first, the tracker map for the rest')


# ---------------------------------------------------------------------------
# The worker, end to end
# ---------------------------------------------------------------------------

def _write_gz(path: Path, obj) -> None:
    with gzip.open(path, 'wt', encoding='utf-8') as fh:
        json.dump(obj, fh)


def _job(folder: Path, doc: dict, days: list) -> dict:
    _write_gz(folder / 'facts.json.gz', doc)
    _write_gz(folder / 'sub.json.gz', {'sic': '2000', 'name': 'PIT CO', 'entityType': 'operating'})
    _prices(start='2005-01-03', end='2018-12-31', dividend=0.1).to_parquet(folder / 'px.parquet')
    months = pd.date_range('1973-01-01', '2018-12-01', freq='MS')
    series = {'AAA': pd.Series(4.0, index=months), 'CPIAUCSL': pd.Series(
        np.linspace(40, 250, len(months)), index=months),
        'DGS10': pd.Series(2.5, index=pd.bdate_range('2005-01-03', '2018-12-31'))}
    return {'ticker': 'PIT', 'cik': 1, 'facts': [str(folder / 'facts.json.gz')],
            'sub': str(folder / 'sub.json.gz'), 'prices': str(folder / 'px.parquet'),
            'class_store': str(folder / 'none.json'), 'days': days,
            'macro': {d: BT.macro_as_of(series, d) for d in days}}


def test_worker_rows_see_nothing_filed_after_their_day():
    """Through the backtest's own worker: each day's row is unchanged when
    every fact filed after that day is rewritten, and reads only the fiscal
    years filed by then."""
    doc = _point_in_time_doc()
    for day in (pd.Timestamp('2013-12-31'), pd.Timestamp('2014-12-31')):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            a = BT.evaluate_company(_job(Path(d1), doc, [day]))
            b = BT.evaluate_company(_job(Path(d2), _rewrite_after(doc, day), [day]))
        assert pd.DataFrame(a).equals(pd.DataFrame(b)), (day, a, b)
        row = a[0]
        assert row['LastFY'] == pd.Timestamp(f'{day.year - 1}-12-31'), row['LastFY']
        assert row['Verdict'] in BT.VERDICT_ORDER and row['PricedAtDay'], row
        assert row['PriceDate'] <= day and 'TR1y' not in row, 'the worker saw the future'
    print('  worker: each row is the panel as filed by its day, and no return')


def test_returns_are_attached_from_the_price_files():
    with tempfile.TemporaryDirectory() as d:
        paths = BT.Paths(root=d, work=d)
        px = _prices(start='2005-01-03', end='2018-12-31')
        px.to_parquet(paths.prices / 'PIT.parquet')
        rows = pd.DataFrame({'AsOf': pd.to_datetime(['2014-12-31', '2016-12-31']),
                             'Ticker': 'PIT', 'TR1y': [9.0, 9.0]})
        out = BT.attach_returns(rows, paths, pd.Timestamp('2018-12-31'))
    tr = BT.total_return_index(px)
    want = tr.loc[:'2015-12-31'].iloc[-1] / tr.loc[:'2014-12-31'].iloc[-1] - 1
    assert abs(out['TR1y'].iloc[0] - want) < 1e-12, 'a stale return survived'
    assert np.isnan(out['TR3y'].iloc[1]) and out['PR1y'].notna().all()
    assert out.attrs['data_end'] == '2018-12-31'
    print('  returns: attached afterwards, from the prices, up to the last close')


def test_a_real_filing_is_point_in_time():
    """Apple's own companyfacts document, from the tracker's cache: rewriting
    everything filed after each day changes nothing in that day's panel, and
    the panel ends with the fiscal year last reported before it."""
    root = Path(os.environ.get('BACKTEST_ROOT') or BT.ROOT)
    path = root / 'data_cache' / 'sec_facts' / 'CIK0000320193.json.gz'
    if not path.exists():
        print(f'  real filing: skipped - no cached Apple document under {root}')
        return
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        doc = json.load(fh)
    splits = pd.Series([7.0, 4.0], index=pd.DatetimeIndex(['2014-06-09', '2020-08-31']))
    for day, last in (('2013-12-31', '2013-09-28'), ('2015-12-31', '2015-09-26'),
                      ('2019-12-31', '2019-09-28')):
        a, ia = F.build_panel('AAPL', doc, {'sic': '3571'}, CFG, splits=splits, as_of=day)
        b, ib = F.build_panel('AAPL', _rewrite_after(_with_cover(doc), day), {'sic': '3571'},
                              CFG, splits=splits, as_of=day)
        pd.testing.assert_frame_equal(a, b, check_freq=False)
        assert ia['cover_shares'] == ib['cover_shares'] and ia['last_fy'] == ib['last_fy']
        assert a.index[-1] == pd.Timestamp(last), (day, a.index[-1])
    # FY2012 EPS: 44.15 as filed in 2012-13, on today's basis 44.15 / 28.
    a, _ = F.build_panel('AAPL', doc, {'sic': '3571'}, CFG, splits=splits, as_of='2013-12-31')
    assert abs(a.loc['2012-09-29', 'eps_diluted'] - 44.15 / 28) < 1e-9, a['eps_diluted']
    print(f'  real filing: Apple as of 2013, 2015 and 2019 - nothing filed later used')


def _with_cover(doc: dict) -> dict:
    """`_rewrite_after` appends to the cover-page count; make sure there is one."""
    dei = doc['facts'].setdefault('dei', {})
    dei.setdefault('EntityCommonStockSharesOutstanding', {'units': {'shares': []}})
    return doc


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in tests:
        print(f'{fn.__name__}:')
        fn()
    print(f'\nall {len(tests)} tests passed')
