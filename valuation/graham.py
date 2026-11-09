"""Benjamin Graham's tests, as he wrote them.

Two separate instruments, kept separate:

* **The defensive investor's seven criteria** (*The Intelligent Investor*, 4th
  revised edition, chapter 14). Quoted, numbered as Graham numbered them.
* **The 1976 ten-point screen** Graham described near the end of his life -
  five tests of reward and five of risk. Much stricter, and a different tool;
  averaging the two together would produce a number that means nothing.

Also here: the Graham Number, net current asset value, and the 1962 growth
formula - which is included because it is widely quoted and flagged because
Graham stopped defending it.

**Three verdicts, not two.** PASS, FAIL, and N/A. A criterion is N/A when the
data to evaluate it does not exist, or when it does not apply to the kind of
company being examined - a current-ratio test on a bank is a category error,
not a failure. Collapsing N/A into FAIL is the single most common way a
Graham screen gets built wrong: it marks every bank, insurer and REIT as
unsuitable for reasons Graham never stated, and it reports a company with a
short filing history as having failed a test that was never run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import pandas as pd

from . import metrics as M
from .config import NA_CRITERIA_BY_GROUP, PROVENANCE, ValuationConfig

PASS, FAIL, NA = 'PASS', 'FAIL', 'N/A'


@dataclass
class Criterion:
    """One test, its verdict, and enough context to argue with it."""

    key: str
    name: str
    verdict: str
    value: float | None = None
    threshold: str = ''
    detail: str = ''
    number: str = ''              # Graham's own numbering, where it exists
    fmt: str = '{:,.2f}'
    # True when the test does not exist for this kind of company (a current
    # ratio for a bank), as opposed to existing but lacking data. Only the
    # second kind counts against data completeness.
    structural: bool = False

    @property
    def source_tier(self) -> str:
        return PROVENANCE.get(self.key, ('derived', ''))[0]

    @property
    def citation(self) -> str:
        return PROVENANCE.get(self.key, ('', 'no citation recorded'))[1]

    def display_value(self) -> str:
        if self.value is None or (isinstance(self.value, float) and not np.isfinite(self.value)):
            return '--'
        try:
            return self.fmt.format(self.value)
        except (ValueError, TypeError):
            return str(self.value)

    def to_row(self) -> dict:
        return {
            'key': self.key, 'number': self.number, 'name': self.name,
            'verdict': self.verdict, 'value': self.value,
            'display': self.display_value(), 'threshold': self.threshold,
            'detail': self.detail, 'source_tier': self.source_tier,
            'citation': self.citation, 'structural': self.structural,
        }


class Test:
    """One criterion's fixed identity - key, name, number, threshold, format.

    Each branch of a test then states only its verdict, value and reason.
    Spelling the identity out in every branch let it drift: the same test
    reported three different names depending on which branch ran, and the
    threshold printed beside a result could disagree with the comparison that
    produced it (K1 printed "< 1.00x" over a <= test).
    """

    def __init__(self, key: str, name: str, number: str = '', threshold: str = '',
                 fmt: str = '{:,.2f}'):
        self.key, self.name, self.number = key, name, number
        self.threshold, self.fmt = threshold, fmt

    def _make(self, verdict, value, detail, structural=False) -> Criterion:
        return Criterion(self.key, self.name, verdict, value, self.threshold, detail,
                         self.number, self.fmt, structural)

    def judge(self, ok: bool, value, detail: str) -> Criterion:
        return self._make(PASS if ok else FAIL, value, detail)

    def fail(self, detail: str, value=None) -> Criterion:
        return self._make(FAIL, value, detail)

    def na(self, detail: str, value=None) -> Criterion:
        """The test applies, but the data to run it is missing - a data gap."""
        return self._make(NA, value, detail)

    def carve(self, detail: str) -> Criterion:
        """The test does not apply to this kind of company - not a data gap."""
        return self._make(NA, None, detail, structural=True)


def _basis(info: dict):
    """Why the filing's per-share figures cannot be set against the quoted
    price, or None when they can.

    Set by `score.evaluate`, which checks that price x the latest fiscal
    year's share count describes a plausible company. Berkshire files its
    per-share figures per Class A share; the ticker most people hold is Class
    B, at 1/1,500th of the price. A P/E across that gap is not a P/E.
    """
    return (info or {}).get('per_share_basis_note')


def _last(panel: pd.DataFrame, col: str):
    """The latest fiscal year's value - never an older one standing in for it."""
    return M.latest(panel.get(col))


def _frac(x: float) -> str:
    f = Fraction(x).limit_denominator(12)
    return str(f.numerator) if f.denominator == 1 else f'{f.numerator}/{f.denominator}'


def _times(x: float, noun: str) -> str:
    return noun if abs(x - 1.0) < 1e-12 else f'{x:g}x {noun}'


def reit_eps_reason(group: str) -> str:
    return ('depreciation on appreciating property makes GAAP EPS a poor earnings '
            f'measure for a {group}')


# ---------------------------------------------------------------------------
# The defensive seven
# ---------------------------------------------------------------------------

def defensive_criteria(panel: pd.DataFrame, info: dict, summary: dict,
                       price: float | None, macro: dict,
                       cfg: ValuationConfig) -> list:
    """Graham's chapter 14 criteria for the defensive investor."""
    g = cfg.graham
    group = info.get('group', 'operating')
    na_keys = NA_CRITERIA_BY_GROUP.get(group, set())
    basis = _basis(info)
    out = []

    # --- 1. Adequate size -------------------------------------------------
    # For a utility Graham set a test on total assets instead of sales.
    cpi = macro.get('cpi_ratio')
    lims = g.inflate(cpi) if cpi else None
    if 'min_revenue' in na_keys:
        measure, lim = _last(panel, 'assets'), (lims or {}).get('min_assets_utility')
        t = Test('min_revenue', 'Adequate size (total assets)', '1',
                 f'>= ${lim / 1e9:.2f}bn' if lim else '', '${:,.2f}bn')
        what, base = 'total assets', '$50m of 1973 assets'
    else:
        measure, lim = _last(panel, 'revenue'), (lims or {}).get('min_revenue')
        t = Test('min_revenue', 'Adequate size (annual revenue)', '1',
                 f'>= ${lim / 1e9:.2f}bn' if lim else '', '${:,.2f}bn')
        what, base = 'revenue', '$100m of 1973 sales'
    if lims is None:
        # In 1973 dollars nearly every listed company clears $100m of sales;
        # running the test there would be a PASS that means nothing.
        out.append(t.na("CPI unavailable - Graham's 1973 threshold cannot be restated "
                        "in today's dollars"))
    elif measure is None:
        out.append(t.na(f'{what} not filed for the latest year'))
    else:
        out.append(t.judge(measure >= lim, measure / 1e9,
                           f"Graham's {base}, inflated x{cpi:.1f}"))

    # --- 2. Strong financial condition ------------------------------------
    # "For industrial companies current assets should be at least twice
    # current liabilities... Also, long-term debt should not exceed the net
    # current assets. For public utilities the debt should not exceed twice
    # the stock equity." Two conditions that must both hold, so one test:
    # scored as two, a company that failed one still collected half the
    # credit. And the current ratio is Graham's test for industrials only -
    # every utility used to fail it, at the 0.6x its business runs on.
    if group == 'utility':
        t = Test('financial_condition', 'Strong financial condition (utility: debt vs equity)',
                 '2', f'debt <= {g.max_debt_to_equity_utility:.1f}x equity', '{:,.2f}x')
        de, equity = _last(panel, 'debt_to_equity'), _last(panel, 'equity')
        if equity is not None and equity <= 0:
            out.append(t.fail(f'equity is ${equity / 1e9:,.2f}bn - liabilities exceed assets'))
        elif de is None:
            out.append(t.na('debt or equity not filed'))
        else:
            out.append(t.judge(de <= g.max_debt_to_equity_utility, de,
                               f'debt is {de:,.2f}x book equity - Graham\'s utility test, in '
                               'place of the industrial current-ratio and working-capital tests'))
    else:
        t = Test('financial_condition', 'Strong financial condition', '2',
                 f'current ratio >= {g.min_current_ratio:.1f}x and LT debt <= working capital',
                 '{:,.2f}x')
        parts = []                      # (passed: bool | None, what, value)
        if 'current_ratio' not in na_keys:
            cr = _last(panel, 'current_ratio')
            parts.append((None, 'current ratio not filed', None) if cr is None else
                         (cr >= g.min_current_ratio,
                          f'current ratio {cr:,.2f}x (needs >= {g.min_current_ratio:.1f}x)', cr))
        if 'ltdebt_vs_working_capital' not in na_keys:
            ltd, wc = _last(panel, 'lt_debt'), _last(panel, 'working_capital')
            if ltd is None or wc is None:
                parts.append((None, 'long-term debt or working capital not filed', None))
            elif wc <= 0:
                parts.append((False, f'working capital is negative (${wc / 1e9:,.2f}bn), so any '
                                     'long-term debt exceeds it', None))
            else:
                r = ltd / wc
                parts.append((r <= g.max_ltdebt_to_working_capital,
                              f'LT debt ${ltd / 1e9:,.2f}bn is {r:,.2f}x working capital '
                              f'(needs <= {g.max_ltdebt_to_working_capital:.2f}x)', r))
        if not parts:
            out.append(t.carve(f'not meaningful for a {group}: "current assets" has no '
                               'operating sense where deposits or float are the liabilities'))
        else:
            detail = '; '.join(what for _, what, _ in parts)
            failed = [v for ok, _, v in parts if ok is False]
            if failed:
                out.append(t.fail(detail, failed[0]))
            elif any(ok is None for ok, _, _ in parts):
                out.append(t.na(detail, parts[0][2]))
            else:
                out.append(t.judge(True, parts[0][2], detail))

    # --- 3. Earnings stability --------------------------------------------
    want = g.earnings_stability_years
    t = Test('earnings_stability', f'Positive earnings, {want} years', '3',
             f'{want}/{want} positive', '{:,.0f}')
    n_pos, n_yrs, neg = summary['eps_positive'], summary['eps_years'], summary['eps_negative_years']
    if n_yrs == 0:
        out.append(t.na('no EPS history'))
    elif n_yrs < want and neg:
        # A loss inside a short window already refutes the test; the missing
        # years cannot rescue it.
        out.append(t.fail(f'only {n_yrs} years on file, but loss-making in {", ".join(neg)} '
                          f'- a {want}-year record is already refuted', n_pos))
    elif n_yrs < want:
        out.append(t.na(f'all {n_yrs} years on file positive, but {want} are needed '
                        '- consistent with the test, not proof of it', n_pos))
    else:
        out.append(t.judge(n_pos == n_yrs, n_pos,
                           f'all {n_yrs} years positive' if not neg
                           else f'loss-making in {", ".join(neg)}'))

    # --- 4. Dividend record -----------------------------------------------
    t = Test('dividend_record', f'Uninterrupted dividends, {g.dividend_years} years', '4',
             f'{g.dividend_years}y', '{:,.0f}y')
    dr = summary['dividend']
    if not dr['verifiable']:
        out.append(t.na(dr['note'], dr['streak']))
    else:
        out.append(t.judge(dr['streak'] >= g.dividend_years, dr['streak'], dr['note']))

    # --- 5. Earnings growth -----------------------------------------------
    t = Test('eps_growth', f'EPS growth over {g.growth_window_years} years', '5',
             f'>= {g.min_eps_growth_total:.0%}', '{:,.1%}')
    gr, detail = summary['eps_growth_total'], summary['eps_growth_detail']
    if 'eps_growth' in na_keys:
        out.append(t.carve(reit_eps_reason(group)))
    elif gr is None:
        out.append(t.na(detail.get('note', 'not computable')))
    else:
        out.append(t.judge(gr >= g.min_eps_growth_total, gr,
                           f'3-year averages: {detail["note"]} '
                           f'({detail["begin_years"][0][:4]}-{detail["begin_years"][-1][:4]} vs '
                           f'{detail["end_years"][0][:4]}-{detail["end_years"][-1][:4]})'))

    # --- 6. Moderate P/E --------------------------------------------------
    t = Test('max_pe_3yr_avg', 'P/E on 3-year average earnings', '6',
             f'<= {g.max_pe_3yr_avg:.0f}x', '{:,.1f}x')
    eps3 = summary.get('eps_avg3')
    if 'max_pe_3yr_avg' in na_keys:
        out.append(t.carve(reit_eps_reason(group)))
    elif price is None:
        out.append(t.na('no current price'))
    elif basis:
        out.append(t.na(basis))
    elif eps3 is None:
        out.append(t.na(summary.get('eps_avg3_note') or '3 years of EPS unavailable'))
    elif eps3 <= 0:
        out.append(t.fail(f'3-year average EPS is {eps3:.2f} - no positive earnings to capitalise'))
    else:
        pe = price / eps3
        out.append(t.judge(pe <= g.max_pe_3yr_avg, pe,
                           f'price {price:,.2f} / 3y average EPS {eps3:,.2f}'))

    # --- 7. Moderate price to assets --------------------------------------
    # "Current price should not be more than 1 1/2 times the book value last
    # reported. However, a multiplier of earnings below 15 could justify a
    # correspondingly higher multiplier of assets... the product of the
    # multiplier times the ratio of price to book value should not exceed
    # 22.5." The product is a second way to meet criterion 7, not an eighth
    # test: scored separately, a company failing both was charged twice for
    # one criterion, and one passing both was credited twice.
    t = Test('price_to_book', 'Price to book (or P/E x P/B)', '7',
             f'<= {g.max_price_to_book:.1f}x, or P/E x P/B <= {g.max_pe_times_pb:.1f}', '{:,.2f}x')
    bvps = _last(panel, 'bvps')
    if price is None:
        out.append(t.na('no current price'))
    elif basis:
        out.append(t.na(basis))
    elif bvps is None:
        # Missing data is N/A. An earlier version failed this test on
        # "equity or share count not filed", which reported a gap in the
        # filing as a verdict on the company.
        out.append(t.na('equity or share count not filed for the latest year'))
    elif bvps <= 0:
        out.append(t.fail(f'book value per share is {bvps:,.2f} - liabilities and preferred '
                          'stock exceed assets'))
    else:
        pb = price / bvps
        over = f'P/B {pb:,.2f}x is above {g.max_price_to_book:.1f}x'
        if pb <= g.max_price_to_book:
            out.append(t.judge(True, pb, f'price {price:,.2f} / book value per share {bvps:,.2f}'))
        elif 'pe_times_pb' in na_keys:
            out.append(t.fail(f'{over}, and the P/E x P/B alternative is not available: '
                              f'{reit_eps_reason(group)}', pb))
        elif eps3 is None:
            out.append(t.na(f'{over}; without a 3-year average EPS the P/E x P/B alternative '
                            'cannot be checked', pb))
        elif eps3 <= 0:
            out.append(t.fail(f'{over}, and with 3-year average EPS of {eps3:,.2f} the P/E x '
                              'P/B alternative cannot rescue it', pb))
        else:
            prod = (price / eps3) * pb
            out.append(t.judge(prod <= g.max_pe_times_pb, pb,
                               f'{over}, so Graham\'s alternative decides: P/E {price / eps3:,.1f}x '
                               f'x P/B {pb:,.2f}x = {prod:,.1f} (needs <= {g.max_pe_times_pb:.1f})'))
    return out


# ---------------------------------------------------------------------------
# The 1976 ten-point screen
# ---------------------------------------------------------------------------

def enterprising_criteria(panel: pd.DataFrame, info: dict, summary: dict,
                          price: float | None, macro: dict, cfg: ValuationConfig,
                          pe_5yr_high: float | None = None,
                          pe_5yr_high_note: str | None = None) -> list:
    """Graham's last published screen: five reward tests, five risk tests.

    Stricter than chapter 14 throughout - it asks for an earnings yield twice
    the AAA bond yield and a price under two-thirds of tangible book, which
    together exclude most of any developed market most of the time. That is
    the intended behaviour: the screen was designed to find a small number of
    statistically cheap securities, not to grade a watchlist.

    The comparisons follow Graham's wording: "at least" is >=, "less than",
    "below" and "greater than" are strict.
    """
    g = cfg.graham
    group = info.get('group', 'operating')
    na_keys = NA_CRITERIA_BY_GROUP.get(group, set())
    aaa = macro.get('aaa_yield')
    basis = _basis(info)
    eps = summary.get('eps_latest')
    out = []

    # R1. Earnings yield at least twice the AAA bond yield.
    mult = g.enterprising_earnings_yield_x_aaa
    need = mult * aaa / 100.0 if aaa is not None else None
    t = Test('enterprising', f'Earnings yield >= {mult:g}x AAA yield', 'R1',
             f'>= {need:.2%}' if need is not None else f'>= {mult:g}x AAA', '{:,.2%}')
    if 'earnings_yield' in na_keys:
        out.append(t.carve(reit_eps_reason(group)))
    elif aaa is None:
        out.append(t.na('AAA corporate bond yield unavailable'))
    elif price is None:
        out.append(t.na('no current price'))
    elif basis:
        out.append(t.na(basis))
    elif eps is None:
        out.append(t.na('EPS not filed for the latest fiscal year'))
    else:
        ey = eps / price
        out.append(t.judge(ey >= need, ey, f'EPS {eps:,.2f} / price {price:,.2f}; AAA {aaa:.2f}%'))

    # R2. P/E less than 40% of the highest average P/E of the last five years.
    frac = g.enterprising_pe_vs_5yr_high
    need = frac * pe_5yr_high if pe_5yr_high is not None else None
    t = Test('enterprising', f'P/E < {frac:.0%} of its 5-year high', 'R2',
             f'< {need:,.1f}x' if need is not None else f'< {frac:.0%} of the 5y high',
             '{:,.1f}x')
    if 'pe_vs_high' in na_keys:
        out.append(t.carve(reit_eps_reason(group)))
    elif price is None:
        out.append(t.na('no current price'))
    elif basis:
        out.append(t.na(basis))
    elif eps is None:
        out.append(t.na('EPS not filed for the latest fiscal year'))
    elif eps <= 0:
        # No earnings, no P/E - and a test that asks for a low one cannot be
        # passed by having none. R1 fails the same company on its negative
        # yield; this used to call it unmeasurable.
        out.append(t.fail(f'latest EPS is {eps:,.2f} - no positive earnings, so no P/E'))
    elif need is None:
        out.append(t.na(pe_5yr_high_note or 'five years of price and earnings history unavailable'))
    else:
        pe = price / eps
        out.append(t.judge(pe < need, pe,
                           f'current {pe:,.1f}x against a 5-year high of {pe_5yr_high:,.1f}x'))

    # R3. Dividend yield at least two-thirds of the AAA yield.
    mult = g.enterprising_div_yield_x_aaa
    need = mult * aaa / 100.0 if aaa is not None else None
    t = Test('enterprising', f'Dividend yield >= {_frac(mult)} AAA yield', 'R3',
             f'>= {need:.2%}' if need is not None else f'>= {_frac(mult)} AAA', '{:,.2%}')
    dps, dps_note = _last(panel, 'dividends_per_share'), None
    dr = summary.get('dividend') or {}
    if dps is None and dr.get('pays_now'):
        # Paid, but not tagged per share: total dividends over the vetted count.
        paid, shares = _last(panel, 'dividends_paid'), info.get('shares_outstanding_now')
        if paid is not None and shares:
            dps, dps_note = abs(paid) / shares, 'total dividends paid / shares outstanding'
    if aaa is None:
        out.append(t.na('AAA yield unavailable'))
    elif price is None:
        out.append(t.na('no current price'))
    elif basis:
        out.append(t.na(basis))
    elif dps is None and not dr.get('pays_now'):
        # A company that pays nothing has a yield of zero, which fails a test
        # that asks for a yield. Reporting it as unmeasurable would let every
        # non-payer through on a technicality.
        out.append(t.fail('no dividend recorded for the latest fiscal year', 0.0))
    elif dps is None:
        out.append(t.na('dividends paid, but neither the per-share amount nor a share '
                        'count is available to express them as a yield'))
    else:
        dy = dps / price
        out.append(t.judge(dy >= need, dy, f'dividend {dps:,.2f} / price {price:,.2f}'
                           + (f' ({dps_note})' if dps_note else '')))

    # R4. Price below two-thirds of tangible book value.
    lim = g.enterprising_price_to_tangible_book
    t = Test('enterprising', f'Price < {_frac(lim)} tangible book', 'R4', f'< {lim:.2f}x',
             '{:,.2f}x')
    tb = _last(panel, 'tangible_bvps')
    if price is None:
        out.append(t.na('no current price'))
    elif basis:
        out.append(t.na(basis))
    elif tb is None:
        out.append(t.na('tangible book value per share unavailable'))
    elif tb <= 0:
        out.append(t.fail(f'tangible book value per share is {tb:,.2f} - goodwill and '
                          'intangibles exceed equity'))
    else:
        r = price / tb
        out.append(t.judge(r < lim, r, f'price {price:,.2f} / tangible book {tb:,.2f}'))

    # R5. Price below two-thirds of net current asset value.
    lim = g.enterprising_price_to_ncav
    t = Test('enterprising', f'Price < {_frac(lim)} net current asset value', 'R5',
             f'< {lim:.2f}x', '{:,.2f}x')
    ncavps = _last(panel, 'ncav_per_share')
    if 'price_to_ncav' in na_keys:
        out.append(t.carve(f'not meaningful for a {group}'))
    elif price is None:
        out.append(t.na('no current price'))
    elif basis:
        out.append(t.na(basis))
    elif ncavps is None:
        out.append(t.na('net current asset value unavailable'))
    elif ncavps <= 0:
        out.append(t.fail(f'net current assets are negative ({ncavps:,.2f}/share) - '
                          'the classic Graham bargain test cannot be met'))
    else:
        r = price / ncavps
        out.append(t.judge(r < lim, r, f'price {price:,.2f} / NCAV per share {ncavps:,.2f}'))

    # K1. Total debt less than tangible book value.
    lim = g.enterprising_debt_to_tangible_book
    t = Test('enterprising', f'Total debt < {_times(lim, "tangible book value")}', 'K1',
             f'< {lim:.2f}x', '{:,.2f}x')
    debt, tangible = _last(panel, 'total_debt'), _last(panel, 'tangible_equity')
    if 'debt_to_tangible_book' in na_keys:
        out.append(t.carve(f'not meaningful for a {group}: borrowing is the business'))
    elif debt is None or tangible is None:
        out.append(t.na('debt or tangible equity unavailable'))
    elif tangible <= 0:
        out.append(t.fail('tangible equity is not positive'))
    else:
        r = debt / tangible
        out.append(t.judge(r < lim, r, f'debt ${debt / 1e9:,.2f}bn vs tangible equity '
                                       f'${tangible / 1e9:,.2f}bn'))

    # K2. Current ratio greater than two.
    lim = g.min_current_ratio
    t = Test('enterprising', f'Current ratio > {lim:g}', 'K2', f'> {lim:.2f}x', '{:,.2f}x')
    cr = _last(panel, 'current_ratio')
    if 'current_ratio' in na_keys:
        out.append(t.carve(f'not meaningful for a {group}'))
    elif cr is None:
        out.append(t.na('current assets or liabilities not filed'))
    else:
        out.append(t.judge(cr > lim, cr, 'current assets / current liabilities'))

    # K3. Total debt less than twice net current asset value.
    lim = g.enterprising_debt_to_ncav
    t = Test('enterprising', f'Total debt < {_times(lim, "net current assets")}', 'K3',
             f'< {lim:.2f}x', '{:,.2f}x')
    ncav = _last(panel, 'ncav')
    if 'debt_to_ncav' in na_keys:
        out.append(t.carve(f'not meaningful for a {group}'))
    elif debt is None or ncav is None:
        out.append(t.na('debt or net current assets unavailable'))
    elif ncav <= 0:
        out.append(t.fail('net current assets are negative'))
    else:
        r = debt / ncav
        out.append(t.judge(r < lim, r, f'debt ${debt / 1e9:,.2f}bn vs NCAV ${ncav / 1e9:,.2f}bn'))

    # K4. Ten-year earnings growth of at least 7% a year.
    lim = g.enterprising_min_eps_cagr
    t = Test('enterprising', f'EPS growth >= {lim:.0%}/yr over 10 years', 'K4',
             f'>= {lim:.0%}', '{:,.1%}')
    cg, note = summary.get('eps_cagr10'), summary.get('eps_cagr10_note')
    if 'eps_cagr' in na_keys:
        out.append(t.carve(reit_eps_reason(group)))
    elif cg is None:
        out.append(t.na(note or 'not computable'))
    else:
        out.append(t.judge(cg >= lim, cg, f'compound annual EPS growth, {note}'))

    # K5. No more than two declines of 5% or worse in ten years.
    limit, thr = g.enterprising_max_eps_declines, g.enterprising_decline_threshold
    t = Test('enterprising', f'At most {limit} EPS declines of {thr:.0%}+ in 10 years', 'K5',
             f'<= {limit}', '{:,.0f}')
    nd, ndnote = summary.get('eps_declines'), summary.get('eps_declines_note')
    if 'eps_declines' in na_keys:
        out.append(t.carve(reit_eps_reason(group)))
    elif nd is None:
        out.append(t.na(ndnote or 'not computable'))
    elif nd > limit:
        # Already over the limit on the years that were measurable; a missing
        # year cannot remove a decline that was seen.
        out.append(t.fail(ndnote, nd))
    elif not summary.get('eps_declines_complete', True):
        out.append(t.na(f'{ndnote} - a decline could sit in the unmeasured years', nd))
    else:
        out.append(t.judge(True, nd, ndnote))
    return out


# ---------------------------------------------------------------------------
# Graham's valuation anchors
# ---------------------------------------------------------------------------

def graham_number(panel: pd.DataFrame, summary: dict, cfg: ValuationConfig, info: dict):
    """sqrt(22.5 x EPS x book value per share).

    Arithmetic from the chapter 14 combined rule, not a formula Graham wrote
    out: 15 x earnings and 1.5 x book multiply to 22.5, and the square root of
    the product gives the price at which both tests bind at once. Uses the
    three-year average EPS rather than the latest year, which is what
    criterion 6 specifies.
    """
    group = info.get('group', 'operating')
    if 'graham_number' in NA_CRITERIA_BY_GROUP.get(group, set()):
        return None, (f'not meaningful for a {group}: GAAP EPS is struck after '
                      'depreciation on assets that typically appreciate')
    if _basis(info):
        return None, _basis(info)
    eps3, bvps = summary.get('eps_avg3'), _last(panel, 'bvps')
    if eps3 is None or bvps is None:
        return None, 'needs a 3-year average EPS and a current book value per share'
    if eps3 <= 0 or bvps <= 0:
        return None, (f'undefined: 3-year average EPS {eps3:,.2f}, '
                      f'book value per share {bvps:,.2f} - the square root of a '
                      'negative product is not a price')
    c = cfg.graham.graham_number_constant
    return math.sqrt(c * eps3 * bvps), f'sqrt({c:g} x {eps3:,.2f} x {bvps:,.2f})'


def ncav_per_share(panel: pd.DataFrame, info: dict):
    """Net current asset value per share - current assets less all liabilities.

    Graham's deep-value floor, and the basis of the record he reported in
    *Security Analysis*. N/A for financial companies by construction.
    """
    group = info.get('group', 'operating')
    if 'ncav' in NA_CRITERIA_BY_GROUP.get(group, set()):
        return None, f'not meaningful for a {group}'
    if _basis(info):
        return None, _basis(info)
    v = _last(panel, 'ncav_per_share')
    if v is None:
        return None, 'current assets or liabilities not filed'
    return v, 'current assets less total liabilities, per share'


def growth_formula_value(summary: dict, macro: dict, cfg: ValuationConfig, info: dict):
    """V = EPS x (8.5 + 2g), and the 1974 revision that divides by the AAA yield.

    Graham published this in 1962 and spent much of the following decade
    warning people off it: it is linear in a growth rate that nobody can
    forecast, so it will return whatever the forecaster already believed. It
    is here because it is quoted constantly and a tracker that omits it will
    be asked why - not because it should carry weight. `g` is capped at
    `growth_formula_max_g` for the same reason.
    """
    g = cfg.graham
    if _basis(info):
        return None, _basis(info)
    if 'eps_growth' in NA_CRITERIA_BY_GROUP.get(info.get('group', 'operating'), set()):
        return None, reit_eps_reason(info.get('group'))
    eps = summary.get('eps_avg3')
    if eps is None:
        eps = summary.get('eps_latest')
    cg = summary.get('eps_cagr10')
    if eps is None or eps <= 0:
        return None, 'needs positive earnings'
    if cg is None:
        return None, 'needs a defined 10-year EPS growth rate'
    g_pct = float(np.clip(cg * 100.0, -5.0, g.growth_formula_max_g))
    base = eps * (g.growth_formula_base + g.growth_formula_multiplier * g_pct)
    aaa = macro.get('aaa_yield')
    if aaa and aaa > 0:
        revised = base * g.growth_formula_norm_yield / aaa
        return revised, (f'EPS {eps:,.2f} x (8.5 + 2 x {g_pct:.1f}) x 4.4 / {aaa:.2f} '
                         '- 1974 revision, growth capped')
    return base, f'EPS {eps:,.2f} x (8.5 + 2 x {g_pct:.1f}) - 1962 form, growth capped'


def pe_five_year_high(panel: pd.DataFrame, prices: pd.DataFrame | None):
    """Highest annual-average P/E of the last five fiscal years.

    Needed by the 1976 screen's second test. The average price over each
    fiscal year is used rather than the year-end price, since a single day's
    close is not what Graham meant by "the highest average P/E".
    """
    if prices is None or prices.empty or 'eps' not in panel.columns:
        return None, 'no price history'
    close = prices['Close'].dropna() if 'Close' in prices else None
    if close is None or close.empty:
        return None, 'no price history'
    # The last five fiscal years of the panel, gaps kept: a P/E "high" taken
    # from whichever five years happen to be filed could reach a decade back.
    eps = panel['eps'].iloc[-5:]
    if eps.notna().sum() < 2:
        return None, 'fewer than 2 of the last 5 fiscal years have EPS'

    ratios = []
    for fy_end, e in eps.items():
        if pd.isna(e) or e <= 0:
            continue
        start = fy_end - pd.DateOffset(years=1)
        window = close.loc[(close.index > start) & (close.index <= fy_end)]
        if len(window) < 40:
            continue
        ratios.append(float(window.mean()) / float(e))
    if not ratios:
        return None, 'no fiscal year in the last five with positive EPS and enough price data'
    return max(ratios), f'highest of {len(ratios)} annual average P/E ratios'
