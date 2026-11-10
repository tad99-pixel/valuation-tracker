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


def test_unit_scale_repair_takes_a_vote_from_the_years_own_figures():
    """McDonald's filed 2021-25 share counts in millions: each bad year's
    neighbours were bad too, so only 2021 was repaired and the stock went to
    INSUFFICIENT DATA. Net income / EPS breaks the tie - but cannot decide
    alone, since net income can be mis-scaled as well."""
    idx = years(10, 2016)
    good = np.array([8.61e8, 8.16e8, 7.86e8, 7.65e8, 7.50e8, 7.52e8, 7.41e8, 7.32e8, 7.22e8, 7.16e8])
    filed = pd.Series(good.copy(), index=idx)
    filed.iloc[5:] = good[5:] / 1e6
    implied = pd.Series(good * 1.001, index=idx)
    fixed, notes = F._repair_unit_scale(filed)
    assert len(notes) < 5, 'expected the neighbour test alone to miss part of the run'
    fixed, notes = F._repair_unit_scale(filed, implied)
    assert np.allclose(fixed, good) and len(notes) == 5, notes
    # Net income filed a millionfold too small one year: the count stays.
    odd = implied.copy()
    odd.iloc[3] /= 1e6
    fixed, notes = F._repair_unit_scale(pd.Series(good, index=idx), odd)
    assert notes == [] and np.allclose(fixed, good), notes

    # ConocoPhillips filed 2010-19 in thousands: ten of nineteen years, enough
    # to carry both medians. The vote alone left most of the run and divided
    # the correct 2009 by a thousand; the smooth net income / EPS series fixes
    # the run and leaves 2009 alone.
    idx = years(19, 2007)
    cop = np.array([2.16, 2.00, 1.97, 1.49, 1.39, 1.25, 1.24, 1.25, 1.24, 1.25, 1.22, 1.18,
                    1.12, 1.08, 1.33, 1.28, 1.21, 1.18, 1.25]) * 1e9
    filed = pd.Series(cop.copy(), index=idx)
    filed.iloc[3:13] = cop[3:13] / 1e3
    fixed, notes = F._repair_unit_scale(filed, pd.Series(cop * 1.002, index=idx))
    assert np.allclose(fixed, cop) and len(notes) == 10, notes
    # The mirror image - net income filed in thousands for a run of years, the
    # share count sound - changes nothing.
    nin = pd.Series(cop * 1.002, index=idx)
    nin.iloc[3:13] = nin.iloc[3:13] / 1e3
    fixed, notes = F._repair_unit_scale(pd.Series(cop.copy(), index=idx), nin)
    assert notes == [] and np.allclose(fixed, cop), notes

    # Coherent's FY2023 D&A total, a thousandfold its own parts, is repaired;
    # parts filed in thousands do not overrule a sound total.
    da = pd.Series([2.2e8, 2.7e8, 2.87e8, 6.817e11, 5.598e8, 5.536e8, 5.219e8], index=years(7, 2020))
    parts = pd.Series([2.2e8, 2.7e8, 2.87e8, 6.817e8, 5.598e8, 5.536e8, 5.219e8], index=da.index)
    fixed, notes = F._repair_unit_scale(da, parts)
    assert abs(fixed.iloc[3] - 6.817e8) < 1 and notes == ['2023: /1,000'], notes
    sound = da.copy()
    sound.iloc[3] = 6.817e8
    small_parts = parts.copy()
    small_parts.iloc[3] /= 1e3
    fixed, notes = F._repair_unit_scale(sound, small_parts)
    assert notes == [] and fixed.equals(sound), notes
    print('  unit repair: a run of bad years fixed with net income / EPS; D&A against its parts')


def test_a_split_shaped_jump_needs_a_split_to_cut_the_history():
    """Cigna doubled its share count buying Express Scripts, and the jump was
    read as a missed 2-for-1 split: ten years of per-share history discarded.
    A split leaves a gap in the price record; a merger leaves a bigger
    company, and a listing leaves no prices before it."""
    idx = years(10, 2015)
    shares = pd.Series([1e8] * 4 + [2e8] * 6, index=idx)
    flat = pd.Series([10e9] * 10, index=idx)
    days = pd.bdate_range('2014-06-01', '2025-12-31')
    calm = pd.DataFrame({'Close': np.linspace(50, 80, len(days))}, index=days)

    def run(revenue, prices, split_dates=()):
        p = pd.DataFrame({'shares_used': shares, 'revenue': revenue})
        info = dict(F._share_basis_diagnostic(p), split_dates=list(split_dates))
        assert info['share_breaks'][0]['kind'] == 'split', info
        return S._confirm_breaks(p, info, prices)

    merger = run(flat * np.r_[[1.0] * 4, [2.2] * 6], calm)
    assert merger['per_share_reliable'] and merger['share_events'], merger
    gapped = calm.copy()
    gapped.loc[gapped.index >= '2020-03-02', 'Close'] /= 2          # a 2-for-1 nobody recorded
    assert not run(flat, gapped)['per_share_reliable']
    assert not run(flat * np.r_[[1.0] * 4, [2.2] * 6], calm, ['2020-06-01'])['per_share_reliable']
    listed = calm[calm.index >= '2019-03-01']
    assert run(flat, listed)['per_share_reliable']
    assert not run(flat, calm)['per_share_reliable'], 'a jump nothing explains still cuts'
    print('  share basis: a split-shaped jump is a merger or a listing unless a split explains it')


def test_a_run_is_dated_by_the_close_it_priced():
    """Monday's cloud runs, made before the US open, priced Friday's close and
    were filed under Monday; a catch-up run after midnight UTC would file
    Monday's close under Tuesday. A run is now dated by its newest close, and
    the catch-up schedule does nothing when that close is already recorded."""
    latest = pd.DataFrame({'Ticker': ['A', 'B', 'C'],
                           'PriceDate': pd.to_datetime(['2026-11-06', '2026-11-06', '2026-10-30'])})
    assert T.run_date_of(latest) == pd.Timestamp('2026-11-06')
    assert T.run_date_of(None) == pd.Timestamp.today().normalize()
    store = {'Latest': latest}
    assert T.close_recorded(store, pd.Timestamp('2026-11-06'))
    assert not T.close_recorded(store, pd.Timestamp('2026-11-09'))
    assert not T.close_recorded({}, pd.Timestamp('2026-11-09'))
    print('  run dating: by the close a run priced; catch-up skips a recorded close')


def test_revenue_is_the_total_not_its_largest_tagged_part():
    """ASC 606 contract revenue leaves out leases, premiums and interest:
    American Tower files $0.77bn of it beside $10.13bn of Revenues, and the
    first tag in the chain used to win. Revenue including excise taxes never
    displaces revenue excluding them; a bank's revenue includes its interest."""
    tags = F.DURATION_CONCEPTS['revenue']
    amt = facts_doc({
        'RevenueFromContractWithCustomerExcludingAssessedTax': {'USD': [dur('2024-01-01', '2024-12-31', 0.77e9, '2025-02-25')]},
        'Revenues': {'USD': [dur('2024-01-01', '2024-12-31', 10.13e9, '2025-02-25')]},
        'OperatingLeaseLeaseIncome': {'USD': [dur('2024-01-01', '2024-12-31', 9.35e9, '2025-02-25')]}})
    vals, used, _ = F._annual_duration_series(amt, tags, 'revenue')
    assert vals.iloc[0] == 10.13e9 and used == ['Revenues'], (vals, used)
    bf = facts_doc({
        'RevenueFromContractWithCustomerExcludingAssessedTax': {'USD': [dur('2023-05-01', '2024-04-30', 4.18e9, '2024-06-13')]},
        'RevenueFromContractWithCustomerIncludingAssessedTax': {'USD': [dur('2023-05-01', '2024-04-30', 5.31e9, '2024-06-13')]}})
    vals, _, _ = F._annual_duration_series(bf, tags, 'revenue')
    assert vals.iloc[0] == 4.18e9, vals
    # A larger figure that is not twice the first is not evidence of a part:
    # Hasbro tags $4.75bn as Revenues beside the $4.14bn it reports.
    has = facts_doc({
        'RevenueFromContractWithCustomerExcludingAssessedTax': {'USD': [dur('2024-01-01', '2024-12-29', 4.14e9, '2025-02-27')]},
        'Revenues': {'USD': [dur('2024-01-01', '2024-12-29', 4.75e9, '2025-02-27')]}})
    vals, _, _ = F._annual_duration_series(has, tags, 'revenue')
    assert vals.iloc[0] == 4.14e9, vals
    bank = pd.DataFrame({'revenue': [2.04e9], 'net_interest_income': [5.34e9],
                         'noninterest_income': [2.04e9], 'net_income': [1.94e9]}, index=years(1, 2024))
    assert F._derive_columns(bank)['revenue'].iloc[0] == 7.38e9
    print('  revenue: the total, not ASC 606 contract revenue; net of excise; banks with interest')


def test_a_zero_share_count_is_no_count():
    """Constellation Energy's carve-out years before its 2022 spin-off file
    zero shares and an EPS of 0.00 against $589m of income. Kept, the zero
    made 2022 a jump of "infx" and cut the per-share history there."""
    raw = pd.DataFrame({
        'net_income': [5.89e8, -2.05e8, -1.6e8, 1.623e9, 3.749e9, 2.319e9],
        'eps_diluted': [0.0, 0.0, -0.49, 5.01, 11.89, 7.40],
        'shares_diluted': [0.0, 0.0, 3.29e8, 3.24e8, 3.15e8, 3.14e8],
    }, index=years(6, 2020))
    p = F._derive_columns(raw)
    assert p['shares_used'].iloc[:2].isna().all() and p['eps'].iloc[:2].isna().all(), p
    assert p['eps'].iloc[2] == -0.49
    out = F._share_basis_diagnostic(p)
    assert out['per_share_reliable'], out
    print('  share counts: a filed zero is read as missing, not as a jump')


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
    assert r['consensus']['models_used'] == ['dcf', 'epv'], r['consensus']
    assert r['models']['earnings_multiple']['role'] == 'price_limit'
    assert crit(r, 'earnings_stability').verdict == 'PASS'
    print(f'  clean company: {r["verdict"]}, completeness {r["completeness"]:.0%}')


def test_dispersion_is_one_measure_for_any_model_count():
    def models(*vals):
        keys = ('dcf', 'epv', 'justified_pb')
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
        keys = ('dcf', 'epv', 'justified_pb')
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
    # A value the new rules cannot compute is not blamed on the filings.
    had = prev.assign(IntrinsicValue=50.0)
    ch = T.detect_changes(had, had.assign(IntrinsicValue=None, ConfigVersion='r5'),
                          pd.Timestamp('2026-11-08'))
    assert list(ch['detail']) == ['not computable under the new rules'], ch.to_dict('records')
    print('  changes: a rule change is labelled, not reported as the company moving')


def test_a_company_that_changed_registration_keeps_its_history():
    """ExxonMobil moved its listing to a new Texas holding company in 2026: the
    new CIK had one quarterly report, the old one the whole history, and the
    panel failed with "no annual duration facts"."""
    from valuation import sources as SRC
    assert SRC.cik_family(34088) == [2115436, 34088] == SRC.cik_family(2115436)
    assert SRC.cik_family(97517) == [1811074, 97517]       # Texas Pacific Land, trust to corp
    assert SRC.cik_family(320193) == [320193]
    old = facts_doc({'NetIncomeLoss': {'USD': [
        dur('2023-01-01', '2023-12-31', 36e9, '2024-02-28'),
        dur('2024-01-01', '2024-12-31', 33e9, '2025-02-26')]}})
    new = facts_doc({'NetIncomeLoss': {'USD': [
        dur('2024-01-01', '2024-12-31', 33.5e9, '2027-02-20'),      # restated by the new filer
        dur('2025-01-01', '2025-12-31', 30e9, '2027-02-20')]}})
    panel, _ = F.build_panel('XOM', SRC.merge_facts([new, old]), {'sic': '2911'}, CFG,
                             splits=pd.Series(dtype=float))
    assert list(panel['net_income']) == [36e9, 33.5e9, 30e9], list(panel['net_income'])

    class Ref:
        def __init__(self, source, sha1):
            self.cik, self.source, self.sha1 = 1, source, sha1
    merged = SRC.MergedFacts([Ref('cache', 'a'), Ref('stale-cache', 'b')])
    assert merged.source == 'stale-cache' and merged.sha1
    assert SRC.MergedFacts([Ref('cache', 'a'), Ref('cache', None)]).sha1 is None
    print('  successions: one history across a change of registration')


# ---------------------------------------------------------------------------
# Suggested improvements (2026-11-09, config r6)
# ---------------------------------------------------------------------------

def test_dividend_record_reaches_back_through_the_payment_history():
    """Graham's twenty years cannot be read from XBRL, which starts in 2009;
    the price record's dividend history covers the years before it."""
    p = pd.DataFrame({'dividends_per_share': 1.0, 'cfo': 1.0}, index=years(10, 2016))
    pay = pd.Series(0.5, index=pd.date_range('1996-02-15', '2025-11-15', freq='3MS')
                    + pd.Timedelta(days=14))
    pay.attrs['since'] = pd.Timestamp('1995-01-03')
    r = M.dividend_record(p, 20, pay)
    assert r['verifiable'] and r['streak'] >= 20 and not r['interrupted'], r
    gap = pay[pay.index.year != 2010]
    gap.attrs['since'] = pay.attrs['since']
    r = M.dividend_record(p, 20, gap)
    assert r['interrupted'] and 'fiscal 2010' in r['note'], r
    # A gap before the last twenty years refutes nothing (Microsoft began
    # paying in fiscal 2003); the note once called it refuted anyway.
    old = pay[pay.index.year != 2002]
    old.attrs['since'] = pay.attrs['since']
    r = M.dividend_record(p, 20, old)
    assert r['streak'] == 23 and 'refuted' not in r['note'], r
    assert 'since fiscal 2003' in r['note'], r
    # The year a company listed counts when it paid in it, though the record
    # covers only part of it (Mastercard: listed May 2006, paid that August).
    part = pay[pay.index >= '2006-08-01']
    part.attrs['since'] = pd.Timestamp('2006-05-25')
    r = M.dividend_record(p, 20, part)
    assert r['streak'] == 20 and r['verifiable'] and not r['interrupted'], r
    late = pay[pay.index >= '2012-01-01']
    late.attrs['since'] = pd.Timestamp('2012-01-03')
    r = M.dividend_record(p, 20, late)
    assert not r['verifiable'] and not r['interrupted'], r
    print('  dividends: twenty years from filings plus the payment record')


def test_preferred_stock_from_shares_and_redemption_price():
    """American Express files $1.6bn of preferred as 1,600 shares at
    $1,000,000 and a PreferredStockValue of zero (par)."""
    p = pd.DataFrame({'equity': [30e9] * 3, 'preferred_equity': [0.0] * 3,
                      'preferred_shares': [1600.0] * 3, 'preferred_price_per_share': [1e6] * 3,
                      'shares_diluted': [700e6] * 3, 'net_income': [10e9] * 3}, index=years(3))
    out = F._derive_columns(p)
    assert out['preferred_claim'].iloc[-1] == 1.6e9, out['preferred_claim']
    assert abs(out['common_equity'].iloc[-1] - 28.4e9) < 1
    print('  preferred: shares x redemption price where no total is filed')


def test_book_value_per_share_uses_year_end_shares():
    p = pd.DataFrame({'equity': [100.0] * 3, 'shares_diluted': [10.0] * 3,
                      'shares_outstanding': [9.0] * 3, 'net_income': [1.0] * 3}, index=years(3))
    assert abs(F._derive_columns(p)['bvps'].iloc[-1] - 100 / 9) < 1e-9
    p['shares_outstanding'] = 3.0          # one class's count: not used
    assert abs(F._derive_columns(p)['bvps'].iloc[-1] - 10.0) < 1e-9
    print('  book value: per share outstanding at the year end')


def test_discount_rates_follow_leverage():
    """The DCF's rate is the equity's, which debt makes riskier; EPV's is the
    whole company's, which debt's tax shield makes cheaper."""
    p, info, prices = synthetic_company()
    price = float(prices['Close'].iloc[-1])
    r = CFG.buffett.discount_rate
    rates = B.required_returns(p, info, price, MACRO, CFG)
    assert rates['equity'] > r > rates['enterprise'], rates
    unlevered = p.copy()
    unlevered['total_debt'] = 0.0
    flat = B.required_returns(unlevered, info, price, MACRO, CFG)
    assert flat['equity'] == flat['enterprise'] == r
    assert B.required_returns(p, dict(info, group='bank'), price, MACRO, CFG)['equity'] == r
    dcf = lambda q: S.evaluate('X', q, info, prices, MACRO, CFG)['models']['dcf']['value']  # noqa: E731
    assert dcf(p) < dcf(unlevered), 'debt did not raise the equity rate'
    print(f'  rates: {rates["equity"]:.2%} equity, {rates["enterprise"]:.2%} whole company')


def test_banks_get_two_value_models_and_15x_is_a_price_limit():
    p, info, prices = synthetic_company()
    r = S.evaluate('BANK', p, dict(info, group='bank'), prices, MACRO, CFG)
    assert set(r['consensus']['models_used']) == {'justified_pb', 'earnings_value'}, r['consensus']
    assert r['models']['earnings_multiple']['role'] == 'price_limit'
    print('  models: banks valued on justified P/B and earnings; 15x EPS a ceiling')


def test_reits_and_utilities_get_a_second_value_model():
    """None of 27 REITs or 37 utilities had a measurable margin: a REIT's EPV
    is carved out, and a utility's net debt swamps its EPV. A REIT's FFO is
    now also capitalised without growth, and a utility is valued like a bank -
    its regulator sets its return on book equity."""
    p, info, prices = synthetic_company()
    reit = S.evaluate('REIT', p, dict(info, group='reit'), prices, MACRO, CFG)
    used = set(reit['consensus']['models_used'])
    assert used == {'dcf', 'earnings_value'}, reit['consensus']
    ev, dcf = reit['models']['earnings_value'], reit['models']['dcf']
    assert ev['label'].startswith('Average FFO') and 0 < ev['value'] < dcf['value'], (ev, dcf)
    assert f"{dcf['detail']['rate']:.1%}" in ev['note'], ev['note']
    util = S.evaluate('UTIL', p, dict(info, group='utility'), prices, MACRO, CFG)
    assert {'justified_pb', 'earnings_value'} <= set(util['consensus']['models_used']), util['consensus']
    print('  models: REITs add capitalised FFO; utilities valued on book and earnings')


def test_one_dollar_window_matches_the_retained_years():
    p, info, prices = synthetic_company()
    change, note = S._market_value_change(prices, p, 10, None)
    start, end = p.index[-11], p.index[-1]
    close = prices['Close']
    want = (float(close[close.index <= end].iloc[-1]) * p['shares_used'].iloc[-1]
            - float(close[close.index <= start].iloc[-1]) * p['shares_used'].iloc[-11])
    assert abs(change - want) < 1e-6 * abs(want) and f'{start:%Y-%m}' in note, (change, want, note)
    print(f'  one-dollar premise: market value over {note.split(":")[0]}')


def test_share_class_figures_are_read_from_the_filing():
    """Visa and Berkshire tag EPS and share counts only by class, which the
    companyfacts feed drops; the filing's own instance keeps the class."""
    from valuation import classes as C
    ctx = ('<context id="{id}"><entity><identifier scheme="x">1</identifier><segment>{dims}'
           '</segment></entity><period><startDate>2024-10-01</startDate>'
           '<endDate>2025-09-30</endDate></period></context>')
    member = ('<xbrldi:explicitMember dimension="us-gaap:StatementClassOfStockAxis">'
              'us-gaap:CommonClass{c}Member</xbrldi:explicitMember>')
    xml = ('<?xml version="1.0"?><xbrl xmlns="http://www.xbrl.org/2003/instance" '
           'xmlns:xbrldi="http://xbrl.org/2006/xbrldi" xmlns:us-gaap="http://fasb.org/us-gaap/2024">'
           + ctx.format(id='a', dims=member.format(c='A'))
           + ctx.format(id='b', dims=member.format(c='B'))
           + ctx.format(id='c', dims=member.format(c='A') + '<xbrldi:explicitMember '
                        'dimension="srt:SegmentsAxis">x:Member</xbrldi:explicitMember>')
           + '<us-gaap:EarningsPerShareDiluted contextRef="a" unitRef="u">10.20</us-gaap:EarningsPerShareDiluted>'
           + '<us-gaap:EarningsPerShareDiluted contextRef="b" unitRef="u">40.80</us-gaap:EarningsPerShareDiluted>'
           + '<us-gaap:EarningsPerShareDiluted contextRef="c" unitRef="u">99</us-gaap:EarningsPerShareDiluted>'
           + '<us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding contextRef="a" unitRef="s">'
           + '1950000000</us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding></xbrl>').encode()
    rows = C.parse_instance(xml, 'CommonClassAMember')
    assert sorted((r['concept'], r['val']) for r in rows) == [
        ('EarningsPerShareDiluted', 10.2),
        ('WeightedAverageNumberOfDilutedSharesOutstanding', 1.95e9)], rows
    with tempfile.TemporaryDirectory() as d:
        real = C.STORE_PATH
        C.STORE_PATH = Path(d) / 'classes.json'
        try:
            C.STORE_PATH.write_text(json.dumps({'V': {
                'member': 'CommonClassAMember', 'read': ['x'],
                'facts': [dict(r, filed='2025-11-10', accn='x') for r in rows]}}))
            blob = {'facts': facts_doc({'NetIncomeLoss': {'USD': [
                dur('2024-10-01', '2025-09-30', 20e9, '2025-11-10')]}})}
            panel, _ = F.build_panel('V', C.inject(blob, 'V')['facts'], {'sic': '7389'}, CFG,
                                     splits=pd.Series(dtype=float))
            assert panel['eps'].iloc[-1] == 10.2 and panel['shares_used'].iloc[-1] == 1.95e9
            assert C.signature('V') and C.signature('KO') is None
        finally:
            C.STORE_PATH = real
    print('  share classes: the traded class read from the filing, as plain facts')


def test_watchlist_can_include_an_index():
    import run_valuation as RV
    from valuation import sources as SRC
    real = SRC.index_constituents
    SRC.index_constituents = lambda name, **k: ['AAPL', 'KO', 'NEWCO']
    try:
        with tempfile.TemporaryDirectory() as d:
            w = Path(d) / 'w.txt'
            w.write_text('KO\nbrk.b  # a comment\n@sp500\n', encoding='utf-8')
            tickers, lists = RV.read_watchlist(w, with_lists=True)
    finally:
        SRC.index_constituents = real
    assert tickers == ['KO', 'BRK-B', 'AAPL', 'NEWCO'], tickers
    assert lists['KO'] == ['Watchlist', 'S&P 500'] and lists['NEWCO'] == ['S&P 500'], lists
    print('  watchlist: @sp500 adds the index, tagged by list')


def test_large_dashboards_load_details_on_demand():
    p, info, prices = synthetic_company()
    r = S.evaluate('TEST', p, info, prices, MACRO, CFG)
    today = pd.Timestamp.today().normalize()
    changes = pd.DataFrame([
        {'date': today, 'Ticker': 'TEST', 'change': 'below Graham Number', 'before': '1',
         'after': '2', 'detail': 'x'},
        {'date': today, 'Ticker': 'TEST', 'change': 'new', 'before': None, 'after': 'NO',
         'detail': 'first evaluation'}])
    pay = D.build_payload({'TEST': r}, {'TEST': p}, MACRO, None, CFG, changes=changes,
                          lists={'TEST': ['Watchlist']})
    assert [c['change'] for c in pay['changes']] == ['below Graham Number'], pay['changes']
    assert pay['companies'][0]['lists'] == ['Watchlist']
    real = D.INLINE_LIMIT
    D.INLINE_LIMIT = 0
    try:
        with tempfile.TemporaryDirectory() as d:
            html = D.render(pay, path=Path(d) / 'dash.html', archive=False).read_text(encoding='utf-8')
            js = (Path(d) / D.DETAILS_DIR / 'TEST.js').read_text(encoding='utf-8')
    finally:
        D.INLINE_LIMIT = real
    start = html.index('id="data">') + len('id="data">')
    data = json.loads(html[start:html.index('</script>', start)])
    c = data['companies'][0]
    assert 'criteria' not in c and c['file'] == 'TEST.js' and data['details'] == 'valuation_details/'
    assert js.startswith('window.VT_DETAIL("TEST",') and '"criteria"' in js
    assert 'criteria' in pay['companies'][0], 'render changed the payload it was given'
    print('  dashboard: details in one script file per company when the list is long')


def test_alerts_list_only_companies_that_moved():
    import run_valuation as RV
    day = pd.Timestamp('2026-11-09')
    ch = pd.DataFrame([
        {'date': day, 'Ticker': 'A', 'change': 'below intrinsic value', 'before': '10 vs 11',
         'after': '9 vs 11', 'detail': 'x|y'},
        {'date': day, 'Ticker': 'B', 'change': 'new', 'before': None, 'after': 'NO', 'detail': ''},
        {'date': day, 'Ticker': 'C', 'change': 'verdict (rules r5 -> r6)', 'before': 'NO',
         'after': 'WATCH', 'detail': ''}])
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / 'alerts.md'
        assert RV.write_alerts(ch, f) == 1
        text = f.read_text(encoding='utf-8')
        assert '| A | below intrinsic value |' in text and 'x/y' in text, text
        assert '| B |' not in text and '| C |' not in text, text
        assert RV.write_alerts(ch.iloc[1:], Path(d) / 'none.md') == 0
        assert not (Path(d) / 'none.md').exists()
        # Beyond the names chosen by hand, only a verdict entering or leaving
        # the priced-below-value group is news.
        idx = pd.DataFrame([
            {'date': day, 'Ticker': 'W', 'change': 'margin band', 'before': 'THIN',
             'after': 'PREMIUM', 'detail': ''},
            {'date': day, 'Ticker': 'X', 'change': 'margin band', 'before': 'THIN',
             'after': 'PREMIUM', 'detail': ''},
            {'date': day, 'Ticker': 'Y', 'change': 'verdict', 'before': 'GOOD BUSINESS, DEAR',
             'after': 'CANDIDATE', 'detail': ''},
            {'date': day, 'Ticker': 'Z', 'change': 'verdict', 'before': 'NO',
             'after': 'GOOD BUSINESS, DEAR', 'detail': ''}])
        assert RV.write_alerts(idx, f, focus=['W']) == 2
        text = f.read_text(encoding='utf-8')
        assert '| W |' in text and '| Y |' in text and '| X |' not in text and '| Z |' not in text
    print('  alerts: companies that moved, not new ones or rule changes')


def test_crossings_and_bands_must_clear_a_buffer():
    """Nike sat at 1.003x its estimate: a cent either way flips it between
    *fair price* and *dear*. A move counts only once it clears 2% - measured
    from the state last reported, so a drift back inside is not news and the
    next real move is still caught."""
    day = pd.Timestamp('2026-11-10')

    def state(price, margin, band, verdict):
        return pd.DataFrame([{'Ticker': 'N', 'Price': price, 'IntrinsicValue': 100.0,
                              'MarginOfSafety': margin, 'MoSBand': band, 'Verdict': verdict,
                              'ConfigVersion': 'r8'}])
    dear = state(100.3, -0.003, 'PREMIUM', 'GOOD BUSINESS, DEAR')
    T.detect_changes(None, dear, day)
    wobble = state(99.5, 0.005, 'THIN', 'GOOD BUSINESS, FAIR PRICE')
    assert T.detect_changes(dear, wobble, day).empty, 'a 0.5% wobble was reported'
    assert wobble['_ref_verdict'].iloc[0] == 'GOOD BUSINESS, DEAR'
    clear = state(96.0, 0.04, 'THIN', 'GOOD BUSINESS, FAIR PRICE')
    ch = T.detect_changes(wobble, clear, day)
    assert set(ch['change']) == {'verdict', 'margin band', 'below intrinsic value'}, ch
    assert ch.set_index('change').loc['verdict', 'before'] == 'GOOD BUSINESS, DEAR'
    back = state(99.0, 0.01, 'THIN', 'GOOD BUSINESS, FAIR PRICE')
    assert T.detect_changes(clear, back, day).empty
    print('  changes: a crossing or band move must clear 2% from the last reported state')


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


# ---------------------------------------------------------------------------
# Point in time (backtest_valuation.py)
# ---------------------------------------------------------------------------

def _point_in_time_doc() -> dict:
    """Five annual reports, FY2011-15, each filed in February and each
    restating the year before - the shape of a real filer's document - with
    a balance sheet and a cover-page share count in every report."""
    eps, ni, sh, eq, cover = [], [], [], [], []
    for i, fy in enumerate(range(2011, 2016)):
        filed = f'{fy + 1}-02-20'
        start, end = f'{fy}-01-01', f'{fy}-12-31'
        eps.append(dur(start, end, 2.0 + 0.2 * i, filed))
        ni.append(dur(start, end, (2.0 + 0.2 * i) * 1e9, filed))
        sh.append(dur(start, end, 1e9, filed))
        eq.append({'end': end, 'val': 10e9 + 1e9 * i, 'filed': filed, 'form': '10-K'})
        cover.append({'end': f'{fy + 1}-02-01', 'val': 1e9 - 1e6 * i, 'filed': filed})
        if i:                       # the prior year, restated in this report
            prior = (f'{fy - 1}-01-01', f'{fy - 1}-12-31')
            eps.append(dur(*prior, 1.9 + 0.2 * (i - 1), filed))
            ni.append(dur(*prior, (1.9 + 0.2 * (i - 1)) * 1e9, filed))
    return facts_doc({'EarningsPerShareDiluted': {'USD/shares': eps},
                      'NetIncomeLoss': {'USD': ni},
                      'WeightedAverageNumberOfDilutedSharesOutstanding': {'shares': sh},
                      'StockholdersEquity': {'USD': eq}},
                     dei={'EntityCommonStockSharesOutstanding': {'shares': cover}})


def _rewrite_after(doc: dict, as_of) -> dict:
    """A copy of `doc` in which every fact filed after `as_of` is changed, and
    to which a later fiscal year and a new tag, both filed later, are added."""
    import copy
    cut = pd.Timestamp(as_of)
    out = copy.deepcopy(doc)
    for concepts in out['facts'].values():
        for node in concepts.values():
            for rows in node['units'].values():
                for r in rows:
                    if pd.Timestamp(r['filed']) > cut:
                        r['val'] = r['val'] * 1000 + 7
    later = (cut + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    gaap = out['facts']['us-gaap']
    gaap['NetIncomeLoss']['units']['USD'].append(dur('2030-01-01', '2030-12-31', 9e15, later))
    gaap['Revenues'] = {'units': {'USD': [dur('2012-01-01', '2012-12-31', 5e15, later)]}}
    out['facts']['dei']['EntityCommonStockSharesOutstanding']['units']['shares'].append(
        {'end': later, 'val': 3e12, 'filed': later})
    return out


def test_no_fact_filed_after_the_as_of_date_reaches_a_panel():
    """The backtest rebuilds each panel as of a past day. Every fact filed
    after that day is rewritten, and a later year and a new tag are added:
    the panel and everything in `info` that the scorer reads must not move.
    The days include a filing date itself (that report counts) and the day
    before one (it does not)."""
    doc = _point_in_time_doc()
    keys = ('n_years', 'first_fy', 'last_fy', 'cover_shares', 'cover_filed',
            'shares_outstanding_now', 'shares_now_source', 'tags_used',
            'per_share_reliable', 'share_unit_fixes', 'last_fy_age_days', 'filing_stale')
    for day in ('2012-12-31', '2013-02-19', '2013-02-20', '2013-12-31', '2014-12-31',
                '2016-06-30'):
        a, ia = F.build_panel('PIT', doc, {'sic': '2000'}, CFG,
                              splits=pd.Series(dtype=float), as_of=day)
        b, ib = F.build_panel('PIT', _rewrite_after(doc, day), {'sic': '2000'}, CFG,
                              splits=pd.Series(dtype=float), as_of=day)
        pd.testing.assert_frame_equal(a, b, check_freq=False)
        for k in keys:
            assert ia.get(k) == ib.get(k), (day, k, ia.get(k), ib.get(k))
        assert a.index[-1] < pd.Timestamp(day), (day, a.index[-1])

    # What a reader saw at the end of 2013: FY2012 as first filed (2.20, the
    # 2.10 restatement came in 2014), FY2011 as restated in 2013 (1.90), and
    # the cover count of the report filed in 2013.
    p, info = F.build_panel('PIT', doc, {'sic': '2000'}, CFG,
                            splits=pd.Series(dtype=float), as_of='2013-12-31')
    assert [round(x, 2) for x in p['eps_diluted']] == [1.9, 2.2], list(p['eps_diluted'])
    assert info['cover_shares'] == 1e9 - 1e6 and info['last_fy_age_days'] == 365, info
    assert info['shares_outstanding_now'] == 1e9 - 1e6 and not info['filing_stale'], info
    # Splits after the day restate everything onto today's basis, as today's
    # split-adjusted price history is.
    p2, _ = F.build_panel('PIT', doc, {'sic': '2000'}, CFG, as_of='2013-12-31',
                          splits=pd.Series([2.0], index=pd.DatetimeIndex(['2014-06-02'])))
    assert [round(x, 2) for x in p2['eps_diluted']] == [0.95, 1.1], list(p2['eps_diluted'])
    # A fact with no filing date cannot be shown to have been public.
    undated = facts_doc({'NetIncomeLoss': {'USD': [{'start': '2011-01-01', 'end': '2011-12-31',
                                                   'val': 1e9, 'form': '10-K'}]}})
    assert F.facts_as_of(undated, '2020-01-01')['facts']['us-gaap'] == {}
    assert 'filed' not in undated['facts']['us-gaap']['NetIncomeLoss']['units']['USD'][0]

    # No date is today: every fact counts, and the clock is today's - by
    # which this five-year-old filing is stale and its cover count unusable.
    now, inow = F.build_panel('PIT', doc, {'sic': '2000'}, CFG, splits=pd.Series(dtype=float))
    today, itoday = F.build_panel('PIT', doc, {'sic': '2000'}, CFG,
                                  splits=pd.Series(dtype=float), as_of=pd.Timestamp.today())
    pd.testing.assert_frame_equal(now, today, check_freq=False)
    assert {k: inow.get(k) for k in keys} == {k: itoday.get(k) for k in keys}
    assert len(now) == 5 and inow['filing_stale'] and inow['shares_outstanding_now'] is None
    print('  point in time: nothing filed after the day reaches the panel')


def test_price_age_is_measured_from_the_as_of_day():
    p, info, prices = synthetic_company()
    last = prices.index[-1]
    r = S.evaluate('X', p, info, prices, MACRO, CFG, as_of=last + pd.Timedelta(days=3))
    assert r['price_stale_days'] == 3, r['price_stale_days']
    r = S.evaluate('X', p, info, prices, MACRO, CFG)
    assert r['price_stale_days'] == (pd.Timestamp.today().normalize() - last).days
    print('  point in time: price age from the as-of day')


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_') and callable(v)]
    for fn in tests:
        print(f'{fn.__name__}:')
        fn()
    print(f'\nall {len(tests)} tests passed')
