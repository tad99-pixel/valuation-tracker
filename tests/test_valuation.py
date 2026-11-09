"""Regression tests for the valuation tracker's easy-to-get-silently-wrong parts.

    python tests/test_valuation.py

No network access: everything runs on synthetic filings, panels and prices.
Each test pins one trap that real data has already sprung - the docstring of
each names the company where it was first seen.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from valuation import buffett as B  # noqa: E402
from valuation import dashboard as D  # noqa: E402
from valuation import fundamentals as F  # noqa: E402
from valuation import graham as G  # noqa: E402
from valuation import metrics as M  # noqa: E402
from valuation import score as S  # noqa: E402
from valuation import tracker as T  # noqa: E402
from valuation.config import DEFAULT as CFG  # noqa: E402


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def dur(start, end, val, filed, form='10-K'):
    return {'start': start, 'end': end, 'val': val, 'filed': filed, 'form': form,
            'fp': 'FY', 'accn': f'acc-{filed}'}


def facts_doc(gaap: dict, dei: dict | None = None) -> dict:
    """A minimal companyfacts document: {tag: {unit: [rows]}}."""
    return {'cik': 1, 'entityName': 'TEST CO',
            'facts': {'us-gaap': {t: {'units': u} for t, u in gaap.items()},
                      'dei': {t: {'units': u} for t, u in (dei or {}).items()}}}


def years(n, start=2011):
    return pd.DatetimeIndex([pd.Timestamp(f'{start + i}-12-31') for i in range(n)])


def synthetic_company(n=15, shares=1e9, growth=0.06, final_pe=15.0):
    """A clean, steadily growing operating company - every test column filled."""
    idx = years(n)
    g = (1 + growth) ** np.arange(n)
    rev = 10e9 * g
    ni = rev * 0.12
    sh = np.full(n, shares)
    equity = 30e9 * g
    p = pd.DataFrame(index=idx)
    p.index.name = 'FYEnd'
    p['revenue'] = rev
    p['net_income'] = ni
    p['eps'] = ni / sh
    p['eps_diluted'] = p['eps']
    p['shares_used'] = sh
    p['equity'] = equity
    p['assets'] = equity * 1.6
    p['bvps'] = equity / sh
    p['tangible_equity'] = equity * 0.9
    p['tangible_bvps'] = p['tangible_equity'] / sh
    p['assets_current'] = rev * 0.8
    p['liabilities_current'] = rev * 0.3
    p['working_capital'] = p['assets_current'] - p['liabilities_current']
    p['current_ratio'] = p['assets_current'] / p['liabilities_current']
    p['lt_debt'] = rev * 0.2
    p['total_debt'] = rev * 0.25
    p['liabilities'] = p['assets'] - equity
    p['ncav'] = p['assets_current'] - p['liabilities']
    p['ncav_per_share'] = p['ncav'] / sh
    p['cash'] = rev * 0.1
    p['short_term_investments'] = 0.0
    p['net_debt'] = p['total_debt'] - p['cash']
    p['dividends_per_share'] = p['eps'] * 0.4
    p['dividends_paid'] = ni * 0.4
    p['cfo'] = ni * 1.3
    p['capex'] = rev * 0.04
    p['depreciation_amortisation'] = rev * 0.035
    p['ppe_net'] = rev * 0.5
    p['fcf'] = p['cfo'] - p['capex']
    p['operating_income'] = ni / 0.79
    p['interest_expense'] = p['total_debt'] * 0.04
    p['gross_margin'] = 0.45
    p['net_margin'] = 0.12
    p['fcf_margin'] = p['fcf'] / rev
    p['roe'] = ni / equity
    p['roic'] = 0.15
    p['debt_to_equity'] = p['total_debt'] / equity
    p['revenue_per_share'] = rev / sh
    price = final_pe * float(p['eps'].iloc[-3:].mean())
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=252 * 12)
    close = price * np.exp(np.linspace(np.log(0.4), 0, len(dates)))
    prices = pd.DataFrame({'Close': close}, index=dates)
    info = {'group': 'operating', 'shares_outstanding_now': shares,
            'per_share_reliable': True, 'name': 'TEST CO', 'n_years': n,
            'last_fy': idx[-1], 'shares_now_source': 'test'}
    return p, info, prices


MACRO = {'aaa_yield': 6.0, 'treasury_10y': 4.5, 'cpi_ratio': 7.6, 'cpi_base_year': 1973}


def crit(result, key, number=None):
    rows = result['defensive'] + result['enterprising'] + result['quality']
    for c in rows:
        if c.key == key and (number is None or c.number == number):
            return c
    raise KeyError(key)


# ---------------------------------------------------------------------------
# XBRL parsing
# ---------------------------------------------------------------------------

def test_restatement_period_length_and_split_by_filing_date():
    """Apple: FY2012 EPS appears as 44.15 (filed 2012) and 6.31 (restated, filed
    2014). The latest filing must win, quarterly rows inside a 10-K must not
    enter the annual series, and each value must be divided by the splits that
    happened AFTER it was filed - which is different for every year."""
    gaap = {
        'EarningsPerShareDiluted': {'USD/shares': [
            dur('2010-09-26', '2011-09-24', 27.68, '2011-10-26'),
            dur('2011-09-25', '2012-09-29', 44.15, '2012-10-31'),
            dur('2011-09-25', '2012-09-29', 6.31, '2014-10-27'),
            dur('2012-07-01', '2012-09-29', 99.0, '2015-10-28'),       # a quarter, in a 10-K
            dur('2014-09-28', '2015-09-26', 9.22, '2017-11-03'),
            dur('2017-10-01', '2018-09-29', 2.98, '2020-10-30'),
        ]},
        'NetIncomeLoss': {'USD': [
            dur('2010-09-26', '2011-09-24', 25.9e9, '2011-10-26'),
            dur('2011-09-25', '2012-09-29', 41.7e9, '2014-10-27'),
            dur('2014-09-28', '2015-09-26', 53.4e9, '2017-11-03'),
            dur('2017-10-01', '2018-09-29', 59.5e9, '2020-10-30'),
        ]},
    }
    splits = pd.Series([7.0, 4.0], index=pd.DatetimeIndex(['2014-06-09', '2020-08-31']))
    panel, info = F.build_panel('TEST', facts_doc(gaap), {'sic': '3571'}, CFG, splits=splits)
    eps = panel['eps_diluted']
    expect = {'2011-09-24': 27.68 / 28, '2012-09-29': 6.31 / 4,
              '2015-09-26': 9.22 / 4, '2018-09-29': 2.98}
    for d, v in expect.items():
        got = float(eps.loc[d])
        assert abs(got - v) < 1e-9, f'{d}: {got} != {v}'
    assert len(panel) == 4, f'a quarterly fact leaked in: {list(panel.index.date)}'
    print(f'  restatement + filing-date splits: {[round(x, 4) for x in eps]}')


def test_unit_scale_repair_needs_both_references():
    """Berkshire filed 2008-09 share counts a millionfold too large. A local
    median alone was outvoted by those two years and "repaired" the correct
    2007 figure; the repair must also leave genuine 9x growth alone."""
    idx = years(8, 2007)
    brk = pd.Series([1.5e6, 1.549e12, 1.551e12, 1.6e6, 1.6e6, 1.6e6, 1.6e6, 1.6e6], index=idx)
    fixed, notes = F._repair_unit_scale(brk)
    assert fixed.iloc[0] == 1.5e6, 'the correct 2007 value was altered'
    assert abs(fixed.iloc[1] - 1.549e6) < 1 and abs(fixed.iloc[2] - 1.551e6) < 1, notes
    assert notes == ['2008: /1,000,000', '2009: /1,000,000'], notes

    nvda = pd.Series([24.3e6, 24.0e6, 21.98e9, 23.5e9, 24.6e9, 25.0e9, 23.8e9], index=years(7, 2008))
    fixed, notes = F._repair_unit_scale(nvda)
    assert abs(fixed.iloc[0] - 24.3e9) < 1 and abs(fixed.iloc[1] - 24.0e9) < 1, notes

    grower = pd.Series([104e6, 107e6, 130e6, 200e6, 300e6, 415e6, 612e6, 900e6], index=idx)
    fixed, notes = F._repair_unit_scale(grower)
    assert notes == [] and fixed.equals(grower), f'real growth was "repaired": {notes}'
    print('  unit repair: Berkshire 2008-09 fixed, 2007 untouched, real growth untouched')


def test_share_basis_diagnostic_tells_split_from_merger():
    idx = years(6)
    split = pd.DataFrame({'shares_used': [100e6, 101e6, 202e6, 203e6, 204e6, 205e6]}, index=idx)
    out = F._share_basis_diagnostic(split)
    assert not out['per_share_reliable'] and out['share_basis_break'] == idx[2], out
    merger = pd.DataFrame({'shares_used': [219e6, 372e6, 380e6, 390e6, 395e6, 400e6]}, index=idx)
    out = F._share_basis_diagnostic(merger)
    assert out['per_share_reliable'] and out['share_events'], out
    print('  share basis: 2.0x read as an uncorrected split, 1.69x as a merger')


def test_debt_composition_never_double_counts():
    """Prologis files LongTermDebt, which already includes current maturities;
    Realty Income files only NotesPayable + LoansPayable."""
    idx = years(3)
    p = pd.DataFrame(index=idx)
    p['lt_debt_incl_current'] = [100.0, 100.0, 100.0]
    p['debt_current_maturities'] = [10.0, 10.0, 10.0]
    p['st_borrowings'] = [5.0, np.nan, 5.0]
    out = F._compose_debt(p.copy())
    assert list(out['total_debt']) == [105.0, 100.0, 105.0], list(out['total_debt'])

    q = pd.DataFrame(index=idx)
    q['notes_payable'] = [25.0, 25.0, np.nan]
    q['loans_payable'] = [1.7, np.nan, np.nan]
    out = F._compose_debt(q.copy())
    assert out['total_debt'].iloc[0] == 26.7 and out['total_debt'].iloc[1] == 25.0
    assert np.isnan(out['total_debt'].iloc[2]), 'unknown debt became zero debt'
    print('  debt composition: no double count, unknown stays unknown')


# ---------------------------------------------------------------------------
# Windows and growth
# ---------------------------------------------------------------------------

def test_windows_never_reach_back_across_gaps():
    """Berkshire's "3-year average EPS" was its 2012-14 figures, a decade stale,
    because missing years were dropped before the window was cut."""
    s = pd.Series([1.0, 2.0, 3.0, np.nan], index=years(4))
    assert M.trailing_average(s, 3)[0] is None
    assert M.latest(s) is None
    assert M.trailing_average(s, 3, min_valid=2)[0] == 2.5
    assert M.cagr(s)[0] is None, 'a growth rate ending in an older year was reported'
    print('  windows: positional, the latest year must exist')


def test_cagr_uses_elapsed_time_not_observations():
    s = pd.Series([100.0] + [np.nan] * 9 + [200.0], index=years(11, 2010))
    rate, note = M.cagr(s)
    assert abs(rate - (2 ** 0.1 - 1)) < 1e-3, (rate, note)
    print(f'  cagr over a gap: {rate:.4f} over ~10 elapsed years ({note})')


def test_eps_declines_window_and_completeness():
    vals = [1.0, 0.5, 1.0] + [1.0 + 0.1 * i for i in range(10)]   # old drop outside 10y
    n, note, complete = M.count_declines(pd.Series(vals, index=years(13)), 0.05, 10)
    assert n == 0 and complete, (n, note)
    gappy = pd.Series([1.0, 1.1, np.nan, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9], index=years(11))
    n, note, complete = M.count_declines(gappy, 0.05, 10)
    assert n == 0 and not complete, note
    print('  declines: ten-year window, gaps reported as incomplete')


def test_dividend_record_semantics():
    """Intel suspended its dividend in 2024: an interruption inside the record
    refutes "20 years uninterrupted" however short the record. A sparse first
    row with no statements is missing data, not a skipped dividend."""
    idx = years(15)
    p = pd.DataFrame({'dividends_per_share': 1.0, 'cfo': 1.0}, index=idx)
    r = M.dividend_record(p, 20)
    assert not r['verifiable'] and r['streak'] == 15
    p2 = p.copy(); p2.loc[idx[-1], 'dividends_per_share'] = np.nan
    r = M.dividend_record(p2, 20)
    assert r['verifiable'] and r['interrupted'] and r['streak'] == 0
    p3 = p.copy(); p3.loc[idx[0], ['dividends_per_share', 'cfo']] = np.nan
    r = M.dividend_record(p3, 20)
    assert not r['interrupted'] and r['streak'] == 14, r
    # Colgate: early years tagged outside the chain are unknown, not unpaid.
    p4 = p.copy(); p4.loc[idx[:6], 'dividends_per_share'] = np.nan
    r = M.dividend_record(p4, 20)
    assert not r['interrupted'] and not r['verifiable'] and r['streak'] == 9, r
    print('  dividends: unverifiable vs refuted vs missing kept apart')


# ---------------------------------------------------------------------------
# Criteria and verdicts
# ---------------------------------------------------------------------------

def test_missing_book_value_is_na_not_fail():
    p, info, prices = synthetic_company()
    p.loc[p.index[-1], 'bvps'] = np.nan
    r = S.evaluate('TEST', p, info, prices, MACRO, CFG)
    c = crit(r, 'price_to_book')
    assert c.verdict == 'N/A', c.to_row()
    print('  criterion 7: missing book value is N/A')


def test_non_payer_fails_dividend_yield():
    p, info, prices = synthetic_company()
    p['dividends_per_share'] = np.nan
    p['dividends_paid'] = np.nan
    r = S.evaluate('TEST', p, info, prices, MACRO, CFG)
    c = crit(r, 'enterprising', 'R3')
    assert c.verdict == 'FAIL' and c.value == 0.0, c.to_row()
    print('  1976 R3: a non-payer fails on a zero yield')


def test_share_class_mismatch_blocks_price_comparisons():
    """Berkshire's filings are per Class A share; the B ticker trades at
    1/1,500th. Price x the filed count must describe a plausible company or
    no P/E, P/B or per-share model may be computed."""
    p, info, prices = synthetic_company(shares=1e9)
    p['shares_used'] = 1e9 / 1500
    info['shares_outstanding_now'] = 1e9 / 1500
    r = S.evaluate('TEST', p, info, prices, MACRO, CFG)
    assert r['info']['per_share_basis_note'], 'basis mismatch not detected'
    assert crit(r, 'max_pe_3yr_avg').verdict == 'N/A'
    assert not r['models']['earnings_multiple']['applicable']
    assert not r['models']['dcf']['applicable'], 'DCF divided by a rejected share count'
    assert r['valuation_component'] == 0.0
    print('  share class mismatch: price comparisons and models withheld')


def test_clean_company_is_measured():
    p, info, prices = synthetic_company()
    r = S.evaluate('TEST', p, info, prices, MACRO, CFG)
    assert r['info']['per_share_basis_note'] is None
    assert r['completeness'] > 0.8, r['completeness']
    assert r['consensus']['n_models'] == 3, r['consensus']
    assert crit(r, 'earnings_stability').verdict == 'PASS'
    print(f'  clean company: {r["verdict"]}, completeness {r["completeness"]:.0%}')


def test_dispersion_is_one_measure_for_any_model_count():
    def models(*vals):
        keys = ('dcf', 'epv', 'earnings_multiple')
        return {k: {'value': v, 'applicable': True} for k, v in zip(keys, vals)}
    c = B.consensus_value(models(100.0, 300.0), CFG)
    assert abs(c['dispersion'] - 0.5) < 1e-12 and c['reliable']
    c = B.consensus_value(models(100.0, 320.0), CFG)
    assert not c['reliable']
    c = B.consensus_value(models(100.0, 200.0, 300.0), CFG)
    assert abs(c['dispersion'] - 0.5) < 1e-12
    assert not B.consensus_value(models(100.0), CFG)['reliable'], 'one model is not a range'
    print('  dispersion: half-range / median, 3x spread is the limit')


def test_verdict_bands():
    q_ok, g_ok = {'rate': 0.9}, {'rate': 0.8}
    m = lambda band: {'verdict': 'MEASURED', 'band': band}
    assert S._verdict_band(0.7, m('STRONG'), g_ok, q_ok) == 'CANDIDATE'
    assert S._verdict_band(0.5, m('THIN'), g_ok, q_ok) == 'GOOD BUSINESS, FAIR PRICE'
    assert S._verdict_band(0.5, m('PREMIUM'), g_ok, q_ok) == 'GOOD BUSINESS, DEAR'
    assert S._verdict_band(0.5, {'verdict': 'UNRELIABLE'}, g_ok, q_ok) == 'GOOD BUSINESS, VALUE UNCLEAR'
    assert S._verdict_band(0.3, m('ADEQUATE'), g_ok, {'rate': 0.2}) == 'STATISTICALLY CHEAP ONLY'
    print('  verdicts: wonderful-company-at-a-fair-price has a label')


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def test_dashboard_json_cannot_break_out_of_its_script_tag():
    evil = 'ACME </script><img src=x onerror=alert(1)> & CO'
    payload = {'generated': 'now', 'config_version': 'x', 'watchlist': '', 'requested': 1,
               'macro': {}, 'params': {}, 'tiers': {}, 'provenance': [],
               'companies': [{'t': 'ACME', 'name': evil}]}
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / 'dash.html'
        D.render(payload, path=out, archive=False)
        html = out.read_text(encoding='utf-8')
    start = html.index('<script type="application/json" id="data">') + len(
        '<script type="application/json" id="data">')
    block = html[start:html.index('</script>', start)]
    assert '<' not in block, 'raw "<" inside the data block'
    assert json.loads(block)['companies'][0]['name'] == evil
    print('  dashboard: embedded JSON escaped, round-trips intact')


def test_tracker_lock_never_loses_history():
    """A workbook that exists but cannot be read (open in Excel) while the
    store is being seeded from it must stop the run, never be treated as
    empty - an empty store beside a full workbook would lose its history."""
    T.time.sleep = lambda s: None
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'tracker.xlsx'
        path.write_bytes(b'not a workbook')
        try:
            T.load_store(path)
        except T.TrackerLocked:
            pass
        else:
            raise AssertionError('unreadable workbook read as empty')
        assert path.read_bytes() == b'not a workbook'
        assert not T.store_dir(path).exists() or not any(T.store_dir(path).iterdir())
    print('  tracker: an unreadable workbook raises, history untouched')


def test_change_detection_reports_crossings():
    prev = pd.DataFrame([{'Ticker': 'A', 'Price': 100.0, 'GrahamNumber': 90.0,
                          'IntrinsicValue': 120.0, 'Verdict': 'WATCH', 'MoSBand': None}])
    now = pd.DataFrame([{'Ticker': 'A', 'Price': 80.0, 'GrahamNumber': 90.0,
                         'IntrinsicValue': None, 'Verdict': 'WATCH', 'MoSBand': None}])
    ch = T.detect_changes(prev, now, pd.Timestamp('2026-11-06'))
    kinds = set(ch['change'])
    assert 'below Graham Number' in kinds, kinds
    assert 'intrinsic value lost' in kinds and not any('above' in k for k in kinds), kinds
    print(f'  changes: {sorted(kinds)}')


# ---------------------------------------------------------------------------
# Review fixes (code-review max, 2026-11-07)
# ---------------------------------------------------------------------------

def test_year_end_moved_by_a_day_is_one_fiscal_year():
    """Deere re-tagged FY2015 as ending 2015-11-01 instead of 10-31; the panel
    had 21 rows for 19 years. The latest filing must win, as one row."""
    gaap = {'NetIncomeLoss': {'USD': [
        dur('2013-11-01', '2014-10-31', 3.1e9, '2014-12-15'),
        dur('2014-11-01', '2015-10-31', 1.9e9, '2015-12-15'),
        dur('2014-11-02', '2015-11-01', 2.0e9, '2017-12-15'),     # re-tagged, later
        dur('2015-11-02', '2016-10-30', 1.5e9, '2017-12-15'),
    ]}}
    panel, _ = F.build_panel('DE', facts_doc(gaap), {'sic': '3523'}, CFG,
                             splits=pd.Series(dtype=float))
    assert len(panel) == 3, list(panel.index.date)
    assert panel['net_income'].iloc[1] == 2.0e9, 'the older filing won'
    print('  year-ends a day apart: one fiscal year, latest filing kept')


def test_lapsed_debt_tag_is_unknown_and_combined_not_doubled():
    idx = years(4)
    p = pd.DataFrame(index=idx)
    p['lt_debt_noncurrent'] = [60.0, np.nan, np.nan, np.nan]
    p['debt_current'] = [5.0, 6.0, 7.0, 8.0]
    out = F._compose_debt(p.copy())
    assert out['total_debt'].iloc[0] == 65.0
    assert out['total_debt'].iloc[1:].isna().all(), 'a lapsed tag became current-only debt'
    q = pd.DataFrame({'debt_total_combined': [50.0], 'st_borrowings': [10.0]}, index=years(1))
    assert F._compose_debt(q.copy())['total_debt'].iloc[0] == 50.0
    print('  debt: a lapsed long-term tag is unknown; combined totals not doubled')


def test_basis_diagnostic_keeps_last_break_and_knows_seven_for_one():
    idx = years(8)
    two = pd.DataFrame({'shares_used': [1e9, 1e9, 4e9, 4e9, 4e9, 40e9, 40e9, 40e9]}, index=idx)
    out = F._share_basis_diagnostic(two)
    assert out['share_basis_break'] == idx[5], out
    seven = pd.DataFrame({'shares_used': [0.9e9] * 3 + [6.2e9] * 5}, index=idx)
    assert not F._share_basis_diagnostic(seven)['per_share_reliable'], '7-for-1 read as a merger'
    print('  basis diagnostic: last break kept, 7-for-1 recognised')


def test_banks_and_reits_get_their_carve_outs():
    p, info, prices = synthetic_company()
    bank = dict(info, group='bank')
    r = S.evaluate('BANK', p, bank, prices, MACRO, CFG)
    assert not r['models']['dcf']['applicable'], 'DCF ran for a bank'
    assert not r['models'].get('reverse_dcf', {}).get('applicable'), 'reverse DCF ran for a bank'
    reit = dict(info, group='reit')
    r = S.evaluate('REIT', p, reit, prices, MACRO, CFG)
    for key, num in (('enterprising', 'R1'), ('enterprising', 'R2'),
                     ('eps_growth', '5'), ('enterprising', 'K4'), ('enterprising', 'K5')):
        c = crit(r, key, num)
        assert c.verdict == 'N/A' and c.structural, (num, c.to_row())
    print('  carve-outs: no DCF for banks; no GAAP-EPS test for REITs')


def test_missing_cpi_is_not_1973_dollars():
    p, info, prices = synthetic_company()
    r = S.evaluate('TEST', p, info, prices, dict(MACRO, cpi_ratio=None), CFG)
    assert crit(r, 'min_revenue').verdict == 'N/A'
    print('  size test: N/A without CPI')


def test_epv_needs_filed_interest_to_assume_no_debt():
    p, info, prices = synthetic_company()
    p['net_debt'] = np.nan
    p['interest_expense'] = np.nan
    r = S.evaluate('TEST', p, info, prices, MACRO, CFG)
    assert not r['models']['epv']['applicable'], r['models']['epv']['note']
    print('  EPV: unknown debt and unknown interest is not "no debt"')


def test_negative_equity_fails_the_utility_leverage_test():
    p, info, prices = synthetic_company()
    p.loc[p.index[-1], 'equity'] = -1e9
    p['debt_to_equity'] = p['total_debt'] / p['equity'].where(p['equity'] > 0)
    r = S.evaluate('UTIL', p, dict(info, group='utility'), prices, MACRO, CFG)
    assert crit(r, 'financial_condition').verdict == 'FAIL'
    print('  utility leverage: negative equity fails')


def test_dispersion_caps_three_models_at_three_x():
    def models(*vals):
        keys = ('dcf', 'epv', 'earnings_multiple')
        return {k: {'value': v, 'applicable': True} for k, v in zip(keys, vals)}
    assert not B.consensus_value(models(60.0, 250.0, 260.0), CFG)['reliable'], '4.3x spread measured'
    assert B.consensus_value(models(100.0, 150.0, 290.0), CFG)['reliable']
    print('  dispersion: three models are held to 3x too')


def test_split_fetch_failure_is_never_cached():
    """A rate limit was stored as "never split", overwrote good history and
    was trusted for a week."""
    import types
    from valuation import sources as SRC
    with tempfile.TemporaryDirectory() as d:
        old = (SRC.SPLITS_PATH, SRC.SPLITS_FETCHED_PATH, sys.modules.get('yfinance'))
        SRC.SPLITS_PATH, SRC.SPLITS_FETCHED_PATH = Path(d) / 's.parquet', Path(d) / 's.json'
        good = pd.Series([4.0], index=pd.DatetimeIndex(['2021-07-20']))
        mode = {'fail': False}

        class FakeTicker:
            def __init__(self, t):
                pass

            @property
            def splits(self):
                if mode['fail']:
                    raise RuntimeError('rate limited')
                return good
        sys.modules['yfinance'] = types.SimpleNamespace(Ticker=FakeTicker)
        try:
            out = SRC.get_splits(['NVDA'], verbose=False)
            assert len(out['NVDA']) == 1
            mode['fail'] = True
            out = SRC.get_splits(['NVDA'], force=True, verbose=False)
            assert out['NVDA'] is not None and len(out['NVDA']) == 1, 'good history overwritten'
            out = SRC.get_splits(['NEWCO'], verbose=False)
            assert out['NEWCO'] is None, 'a failed first fetch read as "never split"'
        finally:
            SRC.SPLITS_PATH, SRC.SPLITS_FETCHED_PATH = old[0], old[1]
            if old[2] is not None:
                sys.modules['yfinance'] = old[2]
            else:
                sys.modules.pop('yfinance', None)
    print('  splits: failures keep the last good history; unknown stays unknown')


def test_tracker_survives_a_cleanup_failure():
    """A locked file in the pending stash used to make the retry loop delete
    the workbook it had just written."""
    import shutil as _sh
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'tracker.xlsx'
        stash = Path(d) / 'valuation_pending' / '20260101_000000'
        stash.mkdir(parents=True)
        pd.DataFrame([{'run_id': 'old', 'date': pd.Timestamp('2026-01-01')}]).to_parquet(
            stash / 'RunLog.parquet')
        real = T.shutil.rmtree
        T.shutil.rmtree = lambda p, *a, **k: (_ for _ in ()).throw(PermissionError('locked'))
        try:
            T.write_tracker({'RunLog': pd.DataFrame([{'run_id': 'new',
                                                       'date': pd.Timestamp('2026-01-02')}])},
                            path, store={})
        finally:
            T.shutil.rmtree = real
        assert path.exists(), 'the workbook was deleted'
        runs = set(pd.read_excel(path, sheet_name='RunLog')['run_id'])
        assert runs == {'old', 'new'}, runs
        stored = set(pd.read_parquet(T._store_file(path, 'RunLog'))['run_id'])
        assert stored == {'old', 'new'}, stored
    print('  tracker: cleanup failure leaves the new workbook, stash merged')


# ---------------------------------------------------------------------------
# Lower-severity review items (2026-11-08)
# ---------------------------------------------------------------------------

def test_open_workbook_never_blocks_the_record():
    """The store is the record; the workbook is a view. A workbook held open
    in Excel must not lose the run or stash it - only the view waits."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'tracker.xlsx'
        real = T.write_workbook
        T.write_workbook = lambda *a, **k: (_ for _ in ()).throw(T.TrackerLocked('open in Excel'))
        try:
            T.write_tracker({'RunLog': pd.DataFrame([{'run_id': 'r1',
                                                       'date': pd.Timestamp('2026-11-01')}])},
                            path, store={})
        finally:
            T.write_workbook = real
        assert list(pd.read_parquet(T._store_file(path, 'RunLog'))['run_id']) == ['r1']
    print('  tracker: a locked workbook leaves the run in the store')


def test_store_keeps_everything_and_the_view_is_bounded():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / 'tracker.xlsx'
        today = pd.Timestamp.today().normalize()
        old = pd.DataFrame([{'date': today - pd.Timedelta(days=600), 'Ticker': 'A', 'Price': 1.0}])
        new = pd.DataFrame([{'date': today, 'Ticker': 'A', 'Price': 2.0}])
        store = T.write_tracker({'History': old}, path, store={})
        T.write_tracker({'History': new}, path, store=store)
        assert len(T.load_store(path)['History']) == 2, 'the store lost a run'
        assert len(pd.read_excel(path, sheet_name='History')) == 1, 'the view is not bounded'
    print('  tracker: store keeps every run; the workbook shows the recent window')


def _apple_like_facts():
    return facts_doc({
        'EarningsPerShareDiluted': {'USD/shares': [
            dur('2010-09-26', '2011-09-24', 27.68, '2011-10-26'),
            dur('2011-09-25', '2012-09-29', 6.31, '2014-10-27'),
            dur('2014-09-28', '2015-09-26', 9.22, '2017-11-03'),
        ]},
        'NetIncomeLoss': {'USD': [
            dur('2010-09-26', '2011-09-24', 25.9e9, '2011-10-26'),
            dur('2011-09-25', '2012-09-29', 41.7e9, '2014-10-27'),
            dur('2014-09-28', '2015-09-26', 53.4e9, '2017-11-03'),
        ]},
        'StockholdersEquity': {'USD': [
            {'end': '2012-09-29', 'val': 118e9, 'filed': '2012-10-31', 'form': '10-K'},
            {'end': '2015-09-26', 'val': 119e9, 'filed': '2015-10-28', 'form': '10-K'},
        ]},
    })


def test_panel_cache_round_trips_and_keys_on_its_inputs():
    """A panel reused from the cache must be the panel a rebuild would give,
    and any change to the document, splits or code must miss the cache."""
    splits = pd.Series([7.0, 4.0], index=pd.DatetimeIndex(['2014-06-09', '2020-08-31']))
    meta = {'sic': '3571', 'name': 'APPLE INC'}
    with tempfile.TemporaryDirectory() as d:
        real = F.FUND_DIR
        F.FUND_DIR = Path(d)
        try:
            key = F.entry_key('sha-1', splits, meta, CFG)
            built = F.build_entry('AAPL', {'facts': _apple_like_facts(), 'meta': meta,
                                           'source': 'network'}, CFG, splits, key=key)
            hit = F.cached_entry('AAPL', key, CFG, source='cache')
            assert hit is not None, 'cache missed its own key'
            pd.testing.assert_frame_equal(built['panel'], hit['panel'], check_dtype=False,
                                          check_index_type=False, check_freq=False)
            for k in ('n_years', 'group', 'splits_known', 'last_fy', 'shares_outstanding_now'):
                assert built['info'][k] == hit['info'][k], (k, built['info'][k], hit['info'][k])
            assert hit['info']['facts_source'] == 'cache'
            assert F.cached_entry('AAPL', F.entry_key('sha-2', splits, meta, CFG), CFG) is None
            assert F.cached_entry('AAPL', F.entry_key('sha-1', None, meta, CFG), CFG) is None
        finally:
            F.FUND_DIR = real
    print('  panel cache: identical on reuse, missed on any changed input')


def test_split_factors_match_their_definition():
    splits = pd.Series([2.0, 3.0, 4.0],
                       index=pd.DatetimeIndex(['2012-05-01', '2016-01-04', '2020-08-31']))
    filed = pd.Series(pd.to_datetime(['2010-01-01', '2012-05-01', '2012-05-02', '2016-01-04',
                                      '2021-01-01', None]))
    got = list(F._split_factors(filed, splits))
    assert got == [24.0, 12.0, 12.0, 4.0, 1.0, 24.0], got
    print('  split factors: product of the splits after each filing date')


def test_instants_align_within_ten_days_only():
    gaap = {'NetIncomeLoss': {'USD': [dur('2020-01-01', '2020-12-31', 1e9, '2021-02-01'),
                                      dur('2021-01-01', '2021-12-31', 1e9, '2022-02-01')]},
            'Assets': {'USD': [{'end': '2020-12-26', 'val': 5e9, 'filed': '2021-02-01'},
                               {'end': '2021-11-15', 'val': 6e9, 'filed': '2022-02-01'}]}}
    panel, _ = F.build_panel('X', facts_doc(gaap), {'sic': '2000'}, CFG,
                             splits=pd.Series(dtype=float))
    assert panel['assets'].iloc[0] == 5e9 and np.isnan(panel['assets'].iloc[1]), panel['assets']
    print('  instants: 5 days off aligns; a quarter-end 46 days off does not')


def test_cover_count_is_checked_against_share_history():
    """With no current fiscal-year count to agree with, a cover-page count was
    accepted unchecked - a count in thousands would have been used as is."""
    today = pd.Timestamp.today().normalize()
    idx = pd.DatetimeIndex([today - pd.DateOffset(years=k) for k in (5, 4, 3)])
    stale = pd.DataFrame({'shares_used': [1.00e9, 1.01e9, 1.02e9]}, index=idx)
    filed = today - pd.Timedelta(days=30)
    n, why = F._vet_current_shares(stale, 1.05e9, filed, CFG)
    assert n == 1.05e9, why
    n, why = F._vet_current_shares(stale, 1.05e6, filed, CFG)
    assert n is None and 'could not be confirmed' in why, why
    blank = pd.DataFrame({'shares_used': [np.nan] * 3}, index=idx)
    n, why = F._vet_current_shares(blank, 1.05e9, filed, CFG)
    assert n is None and 'no recent share count' in why, why
    print('  cover count: checked against the last filed count, or not used')


def test_facts_refetch_only_on_a_new_report():
    from datetime import datetime, timedelta, timezone
    from valuation import sources as SRC
    now = datetime.now(timezone.utc)
    day = lambda n: (now - timedelta(days=n)).date().isoformat()   # noqa: E731
    meta = {'fetched': (now - timedelta(days=10)).isoformat(), 'latest_filed': day(30), 'sha1': 'x'}
    assert SRC.facts_current(meta, day(30), CFG)[0], 'refetched with nothing new filed'
    assert not SRC.facts_current(meta, day(3), CFG)[0], 'a new 10-Q went unnoticed'
    # A report that added no facts: fetched well after it, so do not loop.
    assert SRC.facts_current(dict(meta, latest_filed=day(40)), day(20), CFG)[0]
    assert not SRC.facts_current(dict(meta, fetched=(now - timedelta(days=100)).isoformat()),
                                 day(30), CFG)[0], 'the age backstop did not fire'
    assert not SRC.facts_current(meta, None, CFG)[0]
    assert SRC.facts_current(dict(meta, fetched=(now - timedelta(days=3)).isoformat()), None, CFG)[0]
    print('  facts: refetched for a new report or the backstop, not by the calendar')


def test_incremental_prices_detect_a_restated_history():
    from valuation import sources as SRC
    idx = pd.bdate_range('2026-01-05', periods=30)
    old = pd.DataFrame({'Close': np.linspace(100.0, 110.0, 30)}, index=idx)
    more = pd.bdate_range(idx[-1] + pd.Timedelta(days=1), periods=2)
    new = pd.DataFrame({'Close': list(old['Close'].iloc[-5:]) + [111.0, 112.0]},
                       index=idx[-5:].append(more))
    merged = SRC.merge_incremental(old, new)
    assert len(merged) == 32 and merged['Close'].iloc[-1] == 112.0
    split = new.copy()
    split['Close'] /= 4
    assert SRC.merge_incremental(old, split) is None, 'a 4:1 split was appended to old prices'
    print('  prices: extended in place; a restated history forces a full download')


def test_partial_cik_map_never_replaces_a_complete_one():
    from datetime import datetime, timedelta, timezone
    from valuation import sources as SRC
    with tempfile.TemporaryDirectory() as d:
        old = (SRC.CIK_MAP_PATH, SRC.CIK_OVERRIDE_PATH, SRC._sec_ticker_map, SRC._index_table_ciks)
        SRC.CIK_MAP_PATH, SRC.CIK_OVERRIDE_PATH = Path(d) / 'cik.json', Path(d) / 'none.json'
        full = {f'T{i}': i for i in range(6000)}
        SRC.CIK_MAP_PATH.write_text(json.dumps({
            'fetched': (datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
            'complete': True, 'map': full}))
        SRC._sec_ticker_map = lambda verbose: {}
        SRC._index_table_ciks = lambda verbose: {'NEWCO': 99999, 'T1': 5}
        try:
            m = SRC.load_cik_map(verbose=False)
            assert m['T5999'] == 5999 and m['NEWCO'] == 99999 and m['T1'] == 1
            kept = json.loads(SRC.CIK_MAP_PATH.read_text())
            assert kept['complete'] and len(kept['map']) == 6000, 'partial map overwrote it'
        finally:
            (SRC.CIK_MAP_PATH, SRC.CIK_OVERRIDE_PATH,
             SRC._sec_ticker_map, SRC._index_table_ciks) = old
    print('  cik map: a partial fallback adds to the complete map, never replaces it')


def test_funds_are_declined_by_entity_type():
    from valuation import sources as SRC
    old = (SRC.resolve_ciks, SRC.fetch_submissions, SRC.fetch_companyfacts)
    SRC.resolve_ciks = lambda tickers, verbose=True: ({'SPY': 884394}, [])
    SRC.fetch_submissions = lambda cik, cfg, force=False: {'entityType': 'other', 'name': 'SPDR'}
    SRC.fetch_companyfacts = lambda *a, **k: (_ for _ in ()).throw(AssertionError('fetched'))
    try:
        assert SRC.fetch_many(['SPY'], CFG, verbose=False) == {}
    finally:
        SRC.resolve_ciks, SRC.fetch_submissions, SRC.fetch_companyfacts = old
    print('  funds: declined on SEC entity type, before any facts are fetched')


def test_criterion_names_do_not_drift_between_branches():
    """The same test reported three names depending on which branch ran,
    which split one test into several rows in the tracker's history."""
    base, info, prices = synthetic_company()
    for group in ('operating', 'bank', 'insurance', 'reit', 'utility', 'financial'):
        names = {}
        gaps = base.copy()
        gaps.iloc[-1] = np.nan
        for p, px, macro in ((base, prices, MACRO), (base, None, MACRO),
                             (base, prices, {}), (gaps, prices, MACRO)):
            r = S.evaluate('X', p, dict(info, group=group), px, macro, CFG)
            for c in r['defensive'] + r['enterprising'] + r['quality']:
                names.setdefault((c.key, c.number), set()).add(c.name)
        drift = {k: v for k, v in names.items() if len(v) > 1}
        assert not drift, (group, drift)
    print('  criteria: one name per test in every branch and group')


def test_r2_fails_on_losses_and_7_fails_on_negative_book():
    p, info, prices = synthetic_company()
    p.loc[p.index[-1], ['eps', 'eps_diluted']] = -1.0
    r = S.evaluate('X', p, info, prices, MACRO, CFG)
    assert crit(r, 'enterprising', 'R2').verdict == 'FAIL', crit(r, 'enterprising', 'R2').to_row()
    p, info, prices = synthetic_company()
    p.loc[p.index[-1], 'bvps'] = -5.0
    r = S.evaluate('X', p, info, prices, MACRO, CFG)
    assert crit(r, 'price_to_book').verdict == 'FAIL', crit(r, 'price_to_book').to_row()
    print('  R2 and 7: a loss or negative book fails')


# ---------------------------------------------------------------------------
# Logic review (2026-11-08, config r5)
# ---------------------------------------------------------------------------

def test_buybacks_are_not_credited_as_growth_twice():
    """The DCF counts every dollar of owner earnings as value to today's
    shares, buyback dollars included; crediting per-share growth on top
    counted each buyback again. Dilution is still charged."""
    p, info, prices = synthetic_company(n=15, growth=0.06)
    t = np.arange(len(p))
    for rate, basis in ((-0.04, 'total, buybacks not credited'), (0.05, 'per share, net of dilution')):
        q = p.copy()
        q['shares_used'] = 1e9 * (1 + rate) ** t
        s = M.summarise(q, CFG)
        oe = M.owner_earnings(q, 'greenwald')[0]
        total = M.smoothed_cagr(oe, 10, 3)[0]
        per_share = M.smoothed_cagr(oe / q['shares_used'], 10, 3)[0]
        assert abs(s['owner_earnings_cagr'] - min(total, per_share)) < 1e-12, (rate, s, total, per_share)
        assert s['owner_earnings_cagr_basis'] == basis and s['revenue_cagr_basis'] == basis, s
    print('  growth: buybacks never credited twice; dilution still charged')


def test_one_dollar_premise_counts_buybacks_as_paid_out():
    """Kimberly-Clark, Altria, Kroger, 3M, Nike and Wells Fargo failed this test
    on earnings they had paid out as buybacks."""
    p, info, prices = synthetic_company()
    p['buybacks'] = p['net_income'] * 0.6              # dividends take the other 40%
    r = S.evaluate('X', p, info, prices, MACRO, CFG)
    c = crit(r, 'one_dollar_premise')
    assert c.verdict == 'N/A' and c.structural and 'nothing retained' in c.detail, c.to_row()
    p['buybacks'] = p['net_income'] * 0.3
    kept, _ = M.retained_earnings_test(p, 10)
    assert abs(kept - float(p['net_income'].iloc[-10:].sum() * 0.3)) < 1, kept
    # Home Depot kept 0.6% of what it earned: a ratio over a sliver is noise.
    p['buybacks'] = p['net_income'] * 0.55
    c = crit(S.evaluate('X', p, info, prices, MACRO, CFG), 'one_dollar_premise')
    assert c.verdict == 'N/A' and c.structural and 'too little retained' in c.detail, c.to_row()
    print('  one-dollar premise: retained means kept - dividends and buybacks both paid out')


def test_each_graham_criterion_counts_once():
    """Criterion 7 is met by P/B <= 1.5 *or* P/E x P/B <= 22.5; criterion 2
    needs both its conditions. Each is one test, numbered as Graham did."""
    p, info, prices = synthetic_company()
    r = S.evaluate('X', p, info, prices, MACRO, CFG)
    assert [c.number for c in r['defensive']] == ['1', '2', '3', '4', '5', '6', '7']
    for pe, want in ((10.0, 'PASS'), (14.0, 'FAIL')):           # at P/B 2.0: 20 vs 28
        p, info, prices = synthetic_company(final_pe=pe)
        p.loc[p.index[-1], 'bvps'] = float(prices['Close'].iloc[-1]) / 2.0
        c = crit(S.evaluate('X', p, info, prices, MACRO, CFG), 'price_to_book')
        assert c.verdict == want, (pe, c.to_row())
    p, info, prices = synthetic_company()
    p.loc[p.index[-1], 'current_ratio'] = 1.2               # debt test still passes
    c = crit(S.evaluate('X', p, info, prices, MACRO, CFG), 'financial_condition')
    assert c.verdict == 'FAIL' and 'current ratio 1.20x' in c.detail, c.to_row()
    print('  Graham: seven criteria, each counted once')


def test_utilities_take_grahams_utility_test_only():
    p, info, prices = synthetic_company()
    p['current_ratio'] = 0.6
    c = crit(S.evaluate('U', p, dict(info, group='utility'), prices, MACRO, CFG),
             'financial_condition')
    assert c.verdict == 'PASS' and 'utility' in c.name, c.to_row()
    print('  utilities: debt vs equity, not the industrial current ratio')


def test_preferred_stock_is_deducted_as_a_prior_claim():
    p = pd.DataFrame({'equity': [100.0] * 3, 'preferred_equity': [20.0] * 3,
                      'assets_current': [80.0] * 3, 'liabilities': [50.0] * 3,
                      'shares_diluted': [10.0] * 3, 'net_income': [10.0] * 3}, index=years(3))
    out = F._derive_columns(p)
    assert out['bvps'].iloc[-1] == 8.0 and out['ncav'].iloc[-1] == 10.0, out[['bvps', 'ncav']]
    assert out['tangible_equity'].iloc[-1] == 80.0
    print('  preferred stock: deducted from book value and NCAV')


def test_rule_changes_are_labelled_in_the_change_log():
    prev = pd.DataFrame([{'Ticker': 'A', 'Price': 100.0, 'Verdict': 'NO', 'MoSBand': None,
                          'ConfigVersion': 'r4'}])
    now = prev.assign(Verdict='WATCH', ConfigVersion='r5')
    ch = T.detect_changes(prev, now, pd.Timestamp('2026-11-08'))
    assert list(ch['change']) == ['verdict (rules r4 -> r5)'], list(ch['change'])
    ch = T.detect_changes(prev, now.assign(ConfigVersion='r4'), pd.Timestamp('2026-11-08'))
    assert list(ch['change']) == ['verdict'], list(ch['change'])
    print('  changes: a rule change is labelled, not reported as the company moving')


def test_reverse_dcf_reports_its_discount_rate_range():
    p, info, prices = synthetic_company()
    rev = S.evaluate('X', p, info, prices, MACRO, CFG)['models']['reverse_dcf']
    lo, hi = rev['detail']['range']
    assert lo < rev['value'] < hi, (lo, rev['value'], hi)
    print(f'  reverse DCF: {rev["value"]:.1%} at 9%, {lo:.1%} to {hi:.1%} at 8-10%')


def test_na_table_has_no_dead_keys():
    from valuation.config import NA_CRITERIA_BY_GROUP
    src = Path(G.__file__).read_text(encoding='utf-8') + Path(B.__file__).read_text(encoding='utf-8')
    dead = sorted({k for keys in NA_CRITERIA_BY_GROUP.values() for k in keys if f"'{k}'" not in src})
    assert not dead, f'carve-out keys no test reads: {dead}'
    print('  N/A table: every key is read by a test or model')


def test_dashboard_payload_carries_only_what_the_page_reads():
    p, info, prices = synthetic_company()
    r = S.evaluate('TEST', p, info, prices, MACRO, CFG)
    today = pd.Timestamp.today().normalize()
    hist = pd.DataFrame({'date': pd.date_range(end=today, periods=1000), 'Ticker': 'TEST',
                         'Price': 1.0, 'IntrinsicValue': 2.0, 'PriceToValue': 0.5,
                         'Verdict': 'NO'})
    pay = D.build_payload({'TEST': r}, {'TEST': p}, MACRO, hist, CFG)
    c = pay['companies'][0]
    for dead in ('composite', 'mos', 'dispersion', 'models_used', 'discount_rate',
                 'reverse_note', 'dcf_note'):
        assert dead not in c, dead
    assert set(c['fund']) == {'years', 'eps'}
    row = c['criteria']['defensive'][0]
    assert 'value' not in row and 'citation' not in row and row['key'], row
    assert all(len(v) == 3 for v in c['tally'].values())
    assert len(c['history']) <= D.HISTORY_POINTS
    assert c['history'][-1]['d'] == today.strftime('%Y-%m-%d'), 'thinning dropped the latest run'
    assert [v['label'] for v in pay['verdicts']] == [v[0] for v in S.VERDICTS]
    json.dumps(pay, allow_nan=False)
    print(f'  dashboard payload: {len(json.dumps(pay)) // 1024} KB for one company, '
          f'{len(c["history"])} history points')


def test_archive_keeps_a_month_daily_then_one_a_month():
    from datetime import date, timedelta
    with tempfile.TemporaryDirectory() as d:
        folder, today = Path(d), date(2026, 11, 8)
        days = [today - timedelta(days=i) for i in range(120)]
        for dd in days:
            (folder / f'valuation_dashboard_{dd:%Y%m%d}.html').write_text('x')
        D.prune_archive(folder, 30, today)
        left = {p.name[20:28] for p in folder.glob('*.html')}
        firsts = {}
        for dd in sorted(days):
            firsts.setdefault((dd.year, dd.month), dd)
        want = ({f'{dd:%Y%m%d}' for dd in days if (today - dd).days < 30}
                | {f'{dd:%Y%m%d}' for dd in firsts.values()})
        assert left == want, sorted(left ^ want)
    print(f'  archive: {len(left)} of 120 daily copies kept')


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in tests:
        print(f'{fn.__name__}:')
        fn()
    print(f'\nall {len(tests)} tests passed')
