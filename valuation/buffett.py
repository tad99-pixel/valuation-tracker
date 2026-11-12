"""Buffett's quality tests, and what the business might be worth.

Buffett's own acquisition criteria are four, and three of them are
qualitative: a business we can understand, favourable long-term economics,
able and trustworthy management, and a sensible price. Only the price is
arithmetic. What follows are *proxies* for the other three, and the
`source_tier` on every criterion says plainly which have a basis in the
letters and which are third-party operationalisations that have accumulated
around them. "Gross margin above 40%" is from a book by Mary Buffett, not
from Warren Buffett, and a screen that presents the two with equal authority
is misrepresenting its own evidence.

**On the discounted cash flow.** A DCF is a function that returns whatever
its author believed before starting. Ten years of 12% growth and a 9%
discount rate will justify almost any price, and the honest response is not
to tune the inputs but to run the calculation backwards: given today's price,
what growth is being assumed? `reverse_dcf` does that, and it is the number
to read first - not because it is free of inputs (it moves with the discount
rate, which is why its range at +/-1 point is reported beside it) but because
it needs no growth forecast. The forward DCF is reported beside it, as one
estimate among several, never as "the" intrinsic value.

**On the range.** The models that capitalise earnings at the required return -
a DCF and EPV, or for a bank or insurer justified P/B and an earnings value -
feed a *median*, and its
dispersion - (highest - lowest) / (highest + lowest) - qualifies it. When the
highest estimate is more than about three times the lowest
(`max_model_dispersion` 0.5), the margin of safety is withheld rather than
reported: a precise-looking discount computed from estimates that span a
factor of three is arithmetic dressed up as analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import graham as G
from . import metrics as M
from .config import BOOK_VALUED_GROUPS, NA_CRITERIA_BY_GROUP, ValuationConfig
from .graham import Test, _last

# The range the reverse DCF searches for an implied growth rate. A price
# outside it is reported as "at the bound", never extrapolated.
REVERSE_DCF_BOUNDS = (-0.60, 1.00)


def _carved(info: dict, key: str) -> bool:
    return key in NA_CRITERIA_BY_GROUP.get(info.get('group', 'operating'), set())


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

def quality_criteria(panel: pd.DataFrame, info: dict, summary: dict,
                     price: float | None, cfg: ValuationConfig) -> list:
    """Proxies for "favourable long-term economics" and sound management."""
    b = cfg.buffett
    group = info.get('group', 'operating')
    na_keys = NA_CRITERIA_BY_GROUP.get(group, set())
    win = b.quality_window_years
    out = []

    # A ten-year median taken over two years is not a ten-year median.
    # Verizon "passed" the gross-margin test on two filed years before this.
    min_n = 5

    def thin(n):
        return f'only {n} of the last {win} years measurable, need {min_n}'

    # --- return on equity, measured across the cycle ----------------------
    need = b.min_years_above_roe
    t = Test('min_roe', f'ROE >= {b.min_roe:.0%} in {need} of {win} years', '',
             f'{need}/{win}', '{:,.1%}')
    n_above, n_yrs, med = summary['roe_years_above'], summary['roe_n'], summary.get('roe_median')
    if n_yrs == 0:
        out.append(t.na('ROE not measurable - no net income, or equity never positive'))
    elif n_yrs < need and n_above + (win - n_yrs) < need:
        # Even if every unmeasured year had cleared the bar, too few could.
        out.append(t.fail(f'{n_above} of {n_yrs} measurable years cleared {b.min_roe:.0%} - '
                          f'{need} cannot be reached', med))
    elif n_yrs < need:
        out.append(t.na(f'ROE measurable in only {n_yrs} of the last {win} years (undefined '
                        f'where equity was not positive); {n_above} cleared {b.min_roe:.0%}', med))
    else:
        out.append(t.judge(n_above >= need, med, f'{n_above} of {n_yrs} years cleared '
                                                 f'{b.min_roe:.0%}; median {med:.1%}'))

    # --- return on invested capital ---------------------------------------
    t = Test('min_roic', f'Median ROIC >= {b.min_roic:.0%}', '', f'>= {b.min_roic:.0%}', '{:,.1%}')
    roic_med, roic_n = summary.get('roic_median'), summary.get('roic_n', 0)
    if 'roic' in na_keys:
        out.append(t.carve(f'invested capital has no clean meaning for a {group}'))
    elif roic_med is None or roic_n < min_n:
        out.append(t.na(thin(roic_n)))
    else:
        out.append(t.judge(roic_med >= b.min_roic, roic_med,
                           f'median of {roic_n} years; {summary.get("roic_years_above", 0)} '
                           'cleared the bar. Leverage-insensitive, unlike ROE'))

    # --- debt --------------------------------------------------------------
    t = Test('max_ltdebt_to_net_income', 'Debt repayable from earnings', '',
             f'<= {b.max_ltdebt_to_net_income:.0f}y', '{:,.1f}y')
    debt, ni = _last(panel, 'total_debt'), _last(panel, 'net_income')
    if 'max_ltdebt_to_net_income' in na_keys:
        out.append(t.carve(f'not meaningful for a {group}: borrowed money is the raw '
                           'material, not a risk to it'))
    elif debt is None or ni is None:
        out.append(t.na('debt or net income not filed'))
    elif ni <= 0:
        out.append(t.fail('the latest year is loss-making, so debt is not repayable from earnings'))
    else:
        yrs = debt / ni
        out.append(t.judge(yrs <= b.max_ltdebt_to_net_income, yrs,
                           f'total debt ${debt / 1e9:,.2f}bn is {yrs:,.1f} years of current '
                           'earnings'))

    # --- margins -----------------------------------------------------------
    t = Test('min_gross_margin', f'Gross margin >= {b.min_gross_margin:.0%}', '',
             f'>= {b.min_gross_margin:.0%}', '{:,.1%}')
    gm, gm_n = summary.get('gross_margin_median'), summary.get('gross_margin_n', 0)
    if 'gross_margin' in na_keys:
        out.append(t.carve(f'cost of revenue is not reported by most {group} filers'))
    elif gm is None:
        out.append(t.na('cost of revenue or revenue not filed'))
    elif gm_n < min_n:
        out.append(t.na(thin(gm_n)))
    else:
        out.append(t.judge(gm >= b.min_gross_margin, gm, f'median over {gm_n} years'))

    t = Test('min_net_margin', f'Net margin >= {b.min_net_margin:.0%}', '',
             f'>= {b.min_net_margin:.0%}', '{:,.1%}')
    nm, nm_n = summary.get('net_margin_median'), summary.get('net_margin_n', 0)
    if nm is None:
        out.append(t.na('revenue or net income not filed'))
    elif nm_n < min_n:
        out.append(t.na(thin(nm_n)))
    else:
        out.append(t.judge(nm >= b.min_net_margin, nm, f'median over {nm_n} years'))

    # --- margin durability -------------------------------------------------
    t = Test('max_gross_margin_cv', 'Gross margin stability', '',
             f'cv <= {b.max_gross_margin_cv:.2f}', '{:,.3f}')
    cv, cvnote = summary.get('gross_margin_cv'), summary.get('gross_margin_cv_note')
    if 'gross_margin' in na_keys:
        out.append(t.carve(f'not reported by most {group} filers'))
    elif not gm_n:
        out.append(t.na('cost of revenue not filed, so there is no gross margin to measure'))
    elif cv is None:
        out.append(t.na(cvnote or 'not computable'))
    else:
        out.append(t.judge(cv <= b.max_gross_margin_cv, cv,
                           f'{cvnote} - dispersion, not level: a durable advantage holds its '
                           'margin through a cycle'))

    # --- cash conversion ---------------------------------------------------
    t = Test('min_fcf_margin', f'FCF margin >= {b.min_fcf_margin:.0%}', '',
             f'>= {b.min_fcf_margin:.0%}', '{:,.1%}')
    fm, fm_n = summary.get('fcf_margin_median'), summary.get('fcf_margin_n', 0)
    if 'fcf_margin' in na_keys:
        out.append(t.carve(f'capex and operating cash flow are not comparable for a {group}'))
    elif fm is None:
        out.append(t.na('operating cash flow or capex not filed'))
    elif fm_n < min_n:
        out.append(t.na(thin(fm_n)))
    else:
        out.append(t.judge(fm >= b.min_fcf_margin, fm,
                           f'median over {fm_n} years; cash, not accruals'))

    # --- dilution ----------------------------------------------------------
    t = Test('share_count', 'Share count not growing', '',
             f'<= {b.max_share_count_cagr:.1%}/yr', '{:,.2%}')
    sc, scnote = summary.get('share_count_cagr'), summary.get('share_count_note')
    if sc is None:
        out.append(t.na(scnote or 'not computable'))
    else:
        out.append(t.judge(sc <= b.max_share_count_cagr, sc,
                           f'{scnote} - dilution is a cost the income statement does not show'))

    # --- the one-dollar premise -------------------------------------------
    lim = b.min_market_value_per_retained_dollar
    t = Test('one_dollar_premise', 'Market value created per dollar retained', '',
             f'>= ${lim:,.2f}', '${:,.2f}')
    retained, rdetail = M.retained_earnings_test(panel, b.one_dollar_window_years)
    mcap_change = info.get('market_value_change')
    if rdetail.get('nothing_retained'):
        # Not a data gap: the premise asks what retained earnings created,
        # and a company that paid out everything retained none.
        out.append(t.carve(rdetail['note'] + ' - the premise tests retained earnings, and '
                                             'there were none'))
    elif retained is None:
        out.append(t.na(rdetail.get('note', 'not computable')))
    elif retained < b.min_retained_share * rdetail['earned']:
        out.append(t.carve(f'kept only {retained / rdetail["earned"]:.1%} of the '
                           f'${rdetail["earned"] / 1e9:,.1f}bn earned over {rdetail["years"]}y, '
                           'after dividends and buybacks - too little retained for a ratio '
                           'against it to mean anything'))
    elif mcap_change is None:
        # The real reason, not a fixed guess: it was printing "price history
        # too short" for Berkshire and Visa, whose problem is the share count.
        out.append(t.na('no market value for the start of the window: '
                        + (info.get('market_value_note') or 'not computable')))
    else:
        ratio = mcap_change / retained
        out.append(t.judge(ratio >= lim, ratio,
                           f'market value changed ${mcap_change / 1e9:,.1f}bn against '
                           f'${retained / 1e9:,.1f}bn retained over {rdetail.get("years")}y '
                           '(1984 letter)'))
    return out


def earnings_growth_test(panel: pd.DataFrame, info: dict, cfg: ValuationConfig):
    """Graham's earnings-growth test for the enterprising investor (ch. 15):
    "last year's earnings more than those of 1966" - four years before the
    1970 figures he was reading.

    Every other quality test here reads a ten-year median, and a business in
    decline goes on passing those for years: its good years are still in the
    window. The backtest's buy list was made of them - Franklin Resources in
    seven years of ten, Gilead and Biogen after their peaks, IBM - and it
    trailed the market by 7.6% a year. This test asks the one question the
    medians cannot: is it earning more than it was?

    It is shown, and decides nothing (`ScoreConfig.require_earnings_growth`
    says why): made a requirement of the buy-side verdicts, it would remove
    picks that lagged more over one year and less over three and five. It
    stays out of the quality tally either way - counted as a tenth
    test it was extra credit for whoever passed, and lifted Comcast from five
    of nine to six of ten.

    Per share, as Graham had it; a REIT's on funds from operations, its GAAP
    EPS being carved out everywhere else. A spin-off in between is taken out
    of the earlier figure by its price ratio - the market's own measure of how
    much of the company left - so a parent is not failed for having handed a
    business to its shareholders. Returns one `Criterion`.
    """
    years = cfg.buffett.earnings_growth_years
    group = info.get('group', 'operating')
    t = Test('earnings_growth', f'Earnings above those of {years} years before (shown, not scored)',
             '', '> 0%', '{:+,.1%}')
    if group == 'reit':
        oe, _ = M.owner_earnings(panel, cfg.buffett.maintenance_capex_method, group)
        shares = panel.get('shares_used')
        s = (M._series(oe) / M._series(shares).replace(0, np.nan)
             if oe is not None and shares is not None else pd.Series(dtype=float))
        what = 'FFO per share'
    else:
        s, what = M._series(panel.get('eps')), 'EPS'
    # The year `years` before by its date, not its row: a fiscal year missing
    # from the record must not turn four years into five.
    back = -1
    if len(s) and isinstance(s.index, pd.DatetimeIndex):
        back = int(s.index.get_indexer([s.index[-1] - pd.DateOffset(years=years)],
                                       method='nearest', tolerance=pd.Timedelta(days=45))[0])
    if back < 0 or pd.isna(s.iloc[-1]) or pd.isna(s.iloc[back]):
        return t.na(f'needs {what} for the latest fiscal year and the one {years} years before')
    now, then = float(s.iloc[-1]), float(s.iloc[back])
    first, last = s.index[back], s.index[-1]
    spun = 1.0
    for sp in info.get('spinoffs') or []:
        if first < pd.Timestamp(sp['date']) <= last:
            spun *= float(sp['ratio'])
    basis = then / spun
    span = f'{M._year_label(last)} against {M._year_label(first)}'
    note = (f' ({then:,.2f} as filed, less the {1 - 1 / spun:.0%} of the company spun off since)'
            if spun != 1.0 else '')
    if now <= 0:
        return t.fail(f'{what} {now:,.2f} in {M._year_label(last)} - a loss, against '
                      f'{basis:,.2f} in {M._year_label(first)}{note}')
    if basis <= 0:
        return t.judge(True, None, f'{what} {now:,.2f}, {span}\'s loss of {basis:,.2f}{note}')
    return t.judge(now > basis, now / basis - 1.0,
                   f'{what} {now:,.2f}, {span}\'s {basis:,.2f}{note}')


# ---------------------------------------------------------------------------
# Intrinsic value
# ---------------------------------------------------------------------------

def _valuation_shares(info: dict) -> tuple:
    """The vetted current share count, or None with the reason.

    Deliberately no fallback here. `fundamentals._vet_current_shares` and the
    price-basis check in `score.evaluate` have already decided whether a count
    exists on the same basis as the price; quietly substituting another one
    at this point is exactly how Mastercard was once valued at 7.4x its worth.
    """
    shares = info.get('shares_outstanding_now')
    if shares and shares > 0:
        return float(shares), None
    return None, ('no share count on the same basis as the price: '
                  + (info.get('shares_now_source') or 'none filed'))


def _cash_flow_base(summary: dict, info: dict) -> tuple:
    """The cash flow the DCF and the reverse DCF both discount: (base, basis, why).

    One function for both, so the implied growth and the credited growth the
    dashboard sets side by side are always solved over the same number.

    Banks and insurers are carved out (the `owner_earnings` key of the N/A
    table). Owner earnings mean nothing where deposits or float are the raw
    material: an Allstate DCF of 327.69 moved its median from a 16.6% premium
    to a "fair price".
    """
    if _carved(info, 'owner_earnings'):
        return None, None, (f'owner earnings are not separable from the balance sheet '
                            f'of a {info.get("group")} - valued on justified P/B instead')
    # Explicit None checks: `or` would read a legitimate 0.0 as missing.
    # Normal margin on present revenue first (`metrics.earning_power`); the
    # dollar means stand in only where there is no revenue to normalise on.
    if summary.get('owner_earnings_base') is not None:
        base = summary['owner_earnings_base']
        basis = f'owner earnings ({summary.get("owner_earnings_base_note")})'
    elif summary.get('owner_earnings_avg') is not None:
        base, basis = summary['owner_earnings_avg'], 'owner earnings (5y mean)'
    elif summary.get('owner_earnings_latest') is not None:
        base, basis = summary['owner_earnings_latest'], 'owner earnings (latest)'
    elif summary.get('fcf_avg') is not None:
        base, basis = summary['fcf_avg'], 'free cash flow (5y mean) - owner earnings not estimable'
    else:
        return None, None, 'no owner earnings or free cash flow to discount'
    if base <= 0:
        return None, None, f'{basis} is {base / 1e9:,.2f}bn - nothing positive to discount'
    return base, basis, None


def required_returns(panel: pd.DataFrame, info: dict, price, macro: dict,
                     cfg: ValuationConfig) -> dict:
    """The rate the DCF discounts at and the rate EPV capitalises at, both
    derived from one required return on the business.

    The DCF discounts owner earnings, struck after interest - a cash flow to
    the equity. EPV capitalises operating income, struck before interest - a
    cash flow to the whole company, debt included. Using the same 9% for both
    treated a heavily indebted company's equity as no riskier than its
    business, which flattered its DCF, and charged its EPV more than its
    capital costs. Here `discount_rate` is the return required of the
    business before debt, and with tax-deductible interest (Modigliani-Miller)

        equity rate       r + (r - debt cost) x (1 - tax) x debt / equity
        whole-company     r x (1 - tax x debt / (debt + equity))

    at market values. The debt cost is interest over debt, floored at the AAA
    yield; banks and insurers, whose borrowing is their business, are not
    adjusted. Returns ``{'equity', 'enterprise', 'note'}``.
    """
    b = cfg.buffett
    r, tax = b.discount_rate, b.effective_tax_rate
    out = {'equity': r, 'enterprise': r}
    debt, shares = _last(panel, 'total_debt'), info.get('shares_outstanding_now')
    if info.get('group') in BOOK_VALUED_GROUPS:
        return dict(out, note=f'{r:.1%} required return; no leverage adjustment for a '
                              'business whose returns are set on its book')
    if not price or not shares or debt is None or debt <= 0:
        why = 'no debt' if debt is not None and debt <= 0 else 'leverage unknown'
        return dict(out, note=f'{r:.1%} required return ({why})')
    equity = price * shares
    aaa = macro.get('aaa_yield')
    floor = aaa / 100.0 if aaa else 0.04
    interest = _last(panel, 'interest_expense')
    debt_cost = abs(interest) / debt if interest else floor + 0.01
    debt_cost = min(max(debt_cost, floor), r - 0.005)
    leverage = debt / equity
    out['equity'] = min(r + (r - debt_cost) * (1 - tax) * min(leverage, 3.0), r + 0.08)
    out['enterprise'] = r * (1 - tax * debt / (debt + equity))
    out['note'] = (f'{r:.1%} required of the business; {out["equity"]:.1%} of the equity and '
                   f'{out["enterprise"]:.1%} of the whole company, at debt of {leverage:.2f}x '
                   f'market value costing {debt_cost:.1%}')
    return out


def _two_stage_pv(cf0: float, g1: float, r: float, g_term: float,
                  n: int, fade: bool) -> float:
    """Present value of a cash flow growing at g1, fading to g_term, then forever.

    Returns a value in the same units as `cf0`. The fade matters: holding a
    high stage-one rate flat for ten years and then stepping straight to
    terminal growth is the standard way a DCF is made to say anything at all.
    """
    if r <= g_term:
        return float('nan')
    pv, cf = 0.0, float(cf0)
    for t in range(1, n + 1):
        g = (g1 + (g_term - g1) * (t - 1) / max(1, n - 1)) if fade else g1
        cf = cf * (1.0 + g)
        pv += cf / (1.0 + r) ** t
    terminal = cf * (1.0 + g_term) / (r - g_term)
    pv += terminal / (1.0 + r) ** n
    return pv


def dcf_value(panel: pd.DataFrame, summary: dict, info: dict,
              cfg: ValuationConfig, rates: dict):
    """Two-stage DCF on owner earnings, per share.

    Owner earnings are struck after interest, so they are a cash flow to
    equity and are discounted at a required equity return to give equity value
    directly. No net-debt adjustment is made or wanted here - subtracting debt
    from a stream that has already paid for it would double-count it.

    The growth rate is the historical owner-earnings or revenue growth,
    clipped hard. An unclipped historical rate extrapolated for ten years is
    how a cyclical peak becomes a valuation.
    """
    b = cfg.buffett
    base, basis, why = _cash_flow_base(summary, info)
    if base is None:
        return None, {'note': why}

    shares, why = _valuation_shares(info)
    if not shares:
        return None, {'note': why}

    # Each rate is already the lower of its total and per-share growth (see
    # `metrics.summarise`), and its basis says which.
    g_oe = summary.get('owner_earnings_cagr')
    g_rev = summary.get('revenue_cagr')
    oe_label = f'owner-earnings growth ({summary.get("owner_earnings_cagr_basis") or "basis unknown"})'
    rev_label = f'revenue growth ({summary.get("revenue_cagr_basis") or "basis unknown"})'
    if g_oe is not None and g_rev is not None and b.cap_growth_at_revenue:
        if g_rev < g_oe:
            g_hist = g_rev
            g_src = (f'{rev_label} {g_rev:.1%}, below {oe_label} {g_oe:.1%} - earnings '
                     'outgrowing revenue needs margin expansion, which is finite')
        else:
            g_hist = g_oe
            g_src = f'{oe_label} {g_oe:.1%}; {rev_label} {g_rev:.1%}'
    elif g_oe is not None:
        g_hist = g_oe
        g_src = f'{oe_label} {g_oe:.1%}'
    elif g_rev is not None:
        g_hist = g_rev
        g_src = f'{rev_label} {g_rev:.1%} (owner-earnings growth undefined)'
    else:
        g_hist = b.terminal_growth
        g_src = 'terminal rate (no defined historical growth)'

    # Growth above the terminal rate is credited only as far as it is still
    # showing. A ten-year rate cannot see a business that peaked: at the end
    # of 2018 Gilead's revenue was a fifth below its peak and falling, and its
    # ten-year rate - across the hepatitis C boom - still earned the 15% cap
    # (the backtest's rows for 2017 and 2018). The latest
    # annual report's own three years are the test, and they are on one basis
    # after a divestiture. They can take the premium away; they cannot impose
    # a decline. Two years show that growth has stopped, not that the business
    # will shrink for ten: read that way, Deere's farm-cycle dip halved its
    # value. A decline is credited only where the ten-year record shows one.
    g_now = summary.get('revenue_recent_cagr')
    if g_now is not None and b.cap_growth_at_recent:
        still = max(g_now, b.terminal_growth)
        if still < g_hist:
            g_src += (f'; the latest three reported years show revenue growing {g_now:.1%} a '
                      f'year ({summary.get("revenue_recent_basis")}), so no more than '
                      f'{still:.1%} is credited - growth only as far as it is still showing')
            g_hist = still

    g1 = float(np.clip(g_hist, b.min_stage1_growth, b.max_stage1_growth))
    r = rates['equity']
    pv = _two_stage_pv(base, g1, r, b.terminal_growth, b.stage1_years, b.fade_growth)
    if not np.isfinite(pv):
        return None, {'note': 'discount rate must exceed terminal growth'}

    detail = {
        'growth_stage1': g1,
        'growth_source': g_src,
        'rate': r,
        'note': (f'{basis} ${base / 1e9:,.2f}bn growing {g1:.1%} fading to '
                 f'{b.terminal_growth:.1%}, discounted at {r:.1%} ({rates["note"]})'),
    }
    if abs(g1 - g_hist) > 1e-9:
        bound = (f'{b.max_stage1_growth:.0%} cap' if g_hist > g1
                 else f'{b.min_stage1_growth:.0%} floor')
        detail['note'] += f' (the record\'s {g_hist:.1%} held to the {bound})'
    return pv / shares, detail


def justified_pb_value(panel: pd.DataFrame, summary: dict, info: dict,
                       cfg: ValuationConfig):
    """Book value times the P/B that a bank's returns justify.

    For a bank or insurer, owner earnings and invested capital are not
    separable from the balance sheet - deposits and float are both the raw
    material and the liability - so the DCF and EPV are carved out. That left
    a bank with one valuation model, which the consensus correctly refuses to
    call a range. This is the standard replacement:

        justified P/B = (ROE - g) / (r - g)

    A bank earning exactly its cost of equity is worth book; one earning more
    is worth a premium in proportion. It is the residual-income model in its
    single-stage form, and the closest formal statement of how Buffett has
    described judging banks - returns on equity, earned without excess
    leverage, against the price of the book. Uses the *median* ten-year ROE,
    so one exceptional year cannot set the multiple, and grows only at the
    terminal rate. It fits a regulated utility as well: its regulator sets
    its return as a return on book equity.
    """
    b = cfg.buffett
    if info.get('group', 'operating') not in BOOK_VALUED_GROUPS:
        return None, {'note': 'used for financial companies and utilities; DCF and EPV '
                              'cover the rest'}
    if G._basis(info):
        return None, {'note': G._basis(info)}
    roe, bvps = summary.get('roe_median'), _last(panel, 'bvps')
    if roe is None or bvps is None or bvps <= 0:
        return None, {'note': 'needs a median ROE and a positive book value per share'}
    g, r = b.terminal_growth, b.discount_rate
    if roe <= g:
        return None, {'note': f'median ROE {roe:.1%} does not exceed growth {g:.1%}; '
                              'the model has no positive value'}
    pb = (roe - g) / (r - g)
    return bvps * pb, {'note': (f'book {bvps:,.2f} x justified P/B {pb:.2f} '
                                f'[(ROE {roe:.1%} - g {g:.1%}) / (r {r:.1%} - g)]')}


def _implied_growth(base: float, target: float, r: float, b) -> tuple:
    """Stage-one growth at which `base` discounted at `r` is worth `target`.

    Returns ``(g, bound)``: `bound` is 'low' or 'high' when the answer lies
    outside `REVERSE_DCF_BOUNDS` (and `g` is that bound), else None. ``(None,
    None)`` when the rate does not exceed terminal growth.
    """
    def f(g):
        return _two_stage_pv(base, g, r, b.terminal_growth, b.stage1_years, b.fade_growth) - target

    lo, hi = REVERSE_DCF_BOUNDS
    f_lo, f_hi = f(lo), f(hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        return None, None
    if f_lo > 0:
        return lo, 'low'
    if f_hi < 0:
        return hi, 'high'
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2, None


def reverse_dcf(panel: pd.DataFrame, summary: dict, info: dict,
                price: float, cfg: ValuationConfig, rates: dict | None = None):
    """The growth rate the current price is assuming.

    This is the question worth asking. A forward DCF converts an assumption
    into a valuation and invites you to believe it; the reverse DCF converts
    today's price into the assumption it embeds, which can then be compared
    against what the business has actually done. It needs no growth forecast.
    It is not free of inputs, though: the answer moves with the discount rate
    - Coca-Cola's implied growth is about 20% at 8% and 29% at 10% - so the
    rate one point either side is solved too and reported as a range. A
    conclusion that survives the whole range ("the price needs 20%+, the
    record shows 4%") is a finding; one that flips inside it is not.

    The implied rate is growth of the company's total owner earnings: the
    price is set against today's market value as a whole. It is compared with
    the credited rate, which never includes growth that came from buybacks.

    Returns the implied stage-one growth rate, solved by bisection within
    `REVERSE_DCF_BOUNDS`.
    """
    b = cfg.buffett
    base, _, why = _cash_flow_base(summary, info)
    if base is None:
        return None, {'note': why}
    if not price or price <= 0:
        return None, {'note': 'no current price'}

    shares, why = _valuation_shares(info)
    if not shares:
        return None, {'note': why}

    target = price * shares          # market capitalisation to be justified
    r = (rates or {}).get('equity', b.discount_rate)
    g, bound = _implied_growth(base, target, r, b)
    if g is None:
        return None, {'note': 'discount rate must exceed terminal growth'}
    lo, hi = REVERSE_DCF_BOUNDS
    if bound == 'low':
        return lo, {'note': f'price is below the value of a business shrinking '
                            f'{abs(lo):.0%} a year - the model cannot imply less',
                    'at_bound': True}
    if bound == 'high':
        return hi, {'note': f'price implies more than {hi:.0%} annual growth for '
                            f'{b.stage1_years} years - beyond the solver\'s range',
                    'at_bound': True}
    around = [_implied_growth(base, target, r + d, b)[0] for d in (-0.01, 0.01)]
    note = (f'today\'s price implies {g:.1%} annual growth for '
            f'{b.stage1_years} years at a {r:.1%} discount rate')
    if None not in around:
        note += (f' ({around[0]:.1%} at {r - 0.01:.1%}, '
                 f'{around[1]:.1%} at {r + 0.01:.1%})')
    hist = summary.get('owner_earnings_cagr')
    if hist is not None:
        note += (f'; owner earnings have grown {hist:.1%} a year '
                 f'({summary.get("owner_earnings_cagr_basis") or "basis unknown"})')
    return g, {'note': note, 'at_bound': False, 'rate': r,
               'range': around if None not in around else None}


def epv_value(panel: pd.DataFrame, summary: dict, info: dict,
              cfg: ValuationConfig, rates: dict):
    """Earnings power value: normalised operating earnings, no growth credited.

    Greenwald's construction, not Buffett's or Graham's. It capitalises
    normalised operating earnings after tax at the cost of capital and
    credits growth with nothing at all, which makes it a deliberately
    conservative cross-check on a DCF whose answer depends on a growth
    forecast. Normalised as Greenwald does it - the average operating margin
    of the cycle on the sales the business has now (`metrics.earning_power`) -
    and not as a mean of seven years' dollars, which valued a growing company
    at the size it had three years ago and a shrinking one at a size it had
    lost. Net debt is subtracted here, because operating earnings are struck
    before interest and so belong to debt and equity together.
    """
    b = cfg.buffett
    group = info.get('group', 'operating')
    if _carved(info, 'epv'):
        # For a REIT, GAAP operating income is struck after depreciation on
        # property that usually appreciates, then the full debt stack is
        # subtracted - which took Realty Income to -18.57 a share. A negative
        # value from a profitable business is the model failing, not a finding.
        return None, {'note': (G.reit_eps_reason(group) if group == 'reit' else
                               f'operating earnings and invested capital are not '
                               f'separable for a {group}')}
    norm, note = summary.get('operating_income_base'), summary.get('operating_income_base_note')
    if 'operating_income_base' not in summary:        # a summary built by hand
        norm, note = M.earning_power(panel.get('operating_income'), panel.get('revenue'),
                                     b.epv_normalisation_years)
    if norm is None:
        return None, {'note': f'operating income: {note}'}
    if norm <= 0:
        return None, {'note': f'normalised operating income is not positive ({note})'}

    shares, why = _valuation_shares(info)
    if not shares:
        return None, {'note': why}

    # Net debt turns enterprise value into equity value, so an unknown figure
    # cannot be read as zero: that overstates the equity of exactly the
    # companies whose filings were hardest to read. "No debt tagged" is only
    # taken to mean "no debt" when the income statement agrees - no material
    # interest expense - and the inference is stated.
    nd = _last(panel, 'net_debt')
    debt_note = ''
    if nd is None:
        # Both facts must be *filed* to infer "no debt": an unfiled interest
        # line next to an unfiled debt line is two unknowns, not evidence of
        # none. Reading it as immaterial added cash to equity and took
        # nothing off - a $10bn borrower valued at 18.56 a share for 8.56.
        interest, revenue = _last(panel, 'interest_expense'), _last(panel, 'revenue')
        if interest is None or not revenue:
            return None, {'note': 'neither debt nor interest expense is filed in a readable '
                                  'form - net debt unknown'}
        if abs(interest) >= 0.002 * revenue:
            return None, {'note': 'debt not filed in a readable form, but interest '
                                  'expense says there is some - net debt unknown'}
        nd = -((_last(panel, 'cash') or 0.0) + (_last(panel, 'short_term_investments') or 0.0))
        debt_note = ' (no debt filed and no material interest expense, so debt taken as nil)'
    r = rates['enterprise']
    ev = norm * (1 - b.effective_tax_rate) / r
    return (ev - nd) / shares, {
        'note': (f'operating income ({note}) ${norm / 1e9:,.2f}bn after '
                 f'{b.effective_tax_rate:.0%} tax, capitalised at {r:.1%}, the '
                 f'whole-company rate, less ${nd / 1e9:,.2f}bn net debt'
                 f'{debt_note}; no growth credited')}


def earnings_value(summary: dict, info: dict, rates: dict):
    """Average earnings capitalised at the required return, crediting no growth.

    The book-valued groups have no owner earnings or EPV - their cash flows
    are not separable from the balance sheet - which left justified P/B as
    their only value model, and a single model is not a range. Their earnings
    are still earnings: three years' average EPS capitalised at the required
    return, with no growth credited, is the earning-power view of a bank.
    Graham's "15 x EPS" is not this - it is criterion 6's price ceiling, an
    earnings yield of 6.7% whatever the required return.

    A REIT's EPV is carved out (its GAAP earnings are depreciation-distorted),
    which left its DCF alone, and none of 27 had a measurable margin. Its
    no-growth counterpart is the DCF's own base - funds from operations, the
    five-year mean - capitalised at the same equity rate.
    """
    if info.get('group') == 'reit':
        return _ffo_value(summary, info, rates)
    if info.get('group') not in BOOK_VALUED_GROUPS:
        return None, {'note': 'used for financials, utilities and REITs; EPV covers the rest'}
    if G._basis(info):
        return None, {'note': G._basis(info)}
    eps3 = summary.get('eps_avg3')
    if eps3 is None:
        return None, {'note': summary.get('eps_avg3_note') or 'needs a 3-year average EPS'}
    if eps3 <= 0:
        return None, {'note': f'3-year average EPS is {eps3:,.2f} - nothing to capitalise'}
    r = rates['equity']
    return eps3 / r, {'note': f'3-year average EPS {eps3:,.2f} / required return {r:.1%}; '
                              'no growth credited'}


def _ffo_value(summary: dict, info: dict, rates: dict):
    """A REIT's funds from operations, capitalised at its equity rate."""
    base, basis, why = _cash_flow_base(summary, info)
    if base is None:
        return None, {'note': why}
    shares, why = _valuation_shares(info)
    if not shares:
        return None, {'note': why}
    r = rates['equity']
    return base / r / shares, {'note': (f'{basis.replace("owner earnings", "FFO")} '
                                        f'${base / 1e9:,.2f}bn capitalised at {r:.1%}, the '
                                        'equity rate the DCF uses; no growth credited')}


def earnings_multiple_value(summary: dict, cfg: ValuationConfig, info: dict):
    """Graham's criterion 6 read as a price: 15 x the three-year average EPS.

    A price limit, like the Graham Number (of which it is one factor): the
    most Graham's defensive investor would pay, not an estimate of worth. It
    once fed the estimate, and as the middle of three models it set the
    median for two companies in three - so "price to value" mostly restated
    "P/E above 15", which criterion 6 already tests.
    """
    if _carved(info, 'max_pe_3yr_avg'):
        return None, {'note': G.reit_eps_reason(info.get('group'))}
    if G._basis(info):
        return None, {'note': G._basis(info)}
    eps3 = summary.get('eps_avg3')
    if eps3 is None:
        return None, {'note': summary.get('eps_avg3_note') or 'needs a 3-year average EPS'}
    if eps3 <= 0:
        return None, {'note': f'3-year average EPS is {eps3:,.2f} - nothing to capitalise'}
    pe = cfg.graham.max_pe_3yr_avg
    return eps3 * pe, {'note': f'{pe:.0f} x 3-year average EPS {eps3:,.2f} '
                               '- criterion 6 restated as a price'}


def intrinsic_values(panel: pd.DataFrame, info: dict, summary: dict,
                     price: float | None, macro: dict,
                     cfg: ValuationConfig) -> dict:
    """Every applicable model, each with its reasoning and its own verdict.

    Returns ``{model_key: {'label', 'value', 'note', 'applicable', 'role',
    'source_tier', 'detail'}}``. A model that cannot be computed reports why.
    Nothing is filled in with a default, and the dispersion across whatever
    remains is the honest measure of how much any of this should be trusted.
    """
    out = {}

    def record(key, label, value, detail, tier):
        detail = detail if isinstance(detail, dict) else {'note': str(detail)}
        out[key] = {
            'label': label, 'value': value, 'note': detail.get('note'),
            'applicable': value is not None and bool(np.isfinite(value)) and value > 0,
            'source_tier': tier, 'detail': detail,
        }

    rates = required_returns(panel, info, price, macro, cfg)
    record('dcf', 'Two-stage DCF on owner earnings',
           *dcf_value(panel, summary, info, cfg, rates), 'derived')
    record('epv', 'Earnings power value (no growth)',
           *epv_value(panel, summary, info, cfg, rates), 'third-party')
    record('graham_number', 'Graham Number',
           *G.graham_number(panel, summary, cfg, info), 'graham-derived')
    record('ncav', 'Net current asset value', *G.ncav_per_share(panel, info), 'graham')
    record('earnings_multiple', "15 x 3-year average EPS (criterion 6's ceiling)",
           *earnings_multiple_value(summary, cfg, info), 'graham')
    if info.get('group') in BOOK_VALUED_GROUPS:
        record('justified_pb', 'Justified price-to-book (ROE-based)',
               *justified_pb_value(panel, summary, info, cfg), 'derived')
        record('earnings_value', 'Average earnings at the required return (no growth)',
               *earnings_value(summary, info, rates), 'derived')
    elif info.get('group') == 'reit':
        record('earnings_value', 'Average FFO at the required return (no growth)',
               *earnings_value(summary, info, rates), 'derived')
    record('growth_formula', 'Graham growth formula',
           *G.growth_formula_value(summary, macro, cfg, info), 'graham-disavowed')

    if price:
        g, d = reverse_dcf(panel, summary, info, price, cfg, rates)
        out['reverse_dcf'] = {
            'label': 'Growth implied by the current price', 'value': g,
            'note': d.get('note'), 'applicable': g is not None,
            'source_tier': 'derived', 'detail': d, 'is_rate': True,
        }

    # Graham's two-thirds rule applied to net current assets - the price at
    # which the chapter 15 bargain test would be met, rather than NCAV itself.
    ncav = out.get('ncav') or {}
    if ncav.get('applicable'):
        out['ncav_two_thirds'] = {
            'label': "2/3 of net current asset value (Graham's bargain price)",
            'value': float(ncav['value']) * cfg.graham.ncav_discount,
            'note': 'Graham bought below two-thirds of net current assets',
            'applicable': True, 'source_tier': 'graham', 'detail': {},
        }

    for k, m in out.items():
        if m.get('is_rate'):
            m['role'] = 'rate'
        elif k in VALUE_MODELS:
            m['role'] = 'value'            # feeds the consensus
        elif k in PRICE_LIMIT_MODELS:
            m['role'] = 'price_limit'      # a rule's threshold, not a valuation
        else:
            m['role'] = 'reference'        # shown, deliberately not relied on
    return out


# Models that estimate what the business is worth, by capitalising what it
# earns at the required return. These, and only these, go into the consensus.
# `justified_pb` and `earnings_value` only ever produce a value for financials
# and utilities (and `earnings_value`, on FFO, for REITs), where the first two
# cannot or do not alone.
VALUE_MODELS = ('dcf', 'epv', 'justified_pb', 'earnings_value')

# Prices at which Graham's rules would be satisfied. These are *not* estimates
# of intrinsic value and mixing them into one median was a mistake worth
# naming: the Graham Number is sqrt(22.5 x EPS x book value), so on an
# asset-light business trading at forty times book it does not report what the
# company is worth - it reports that Graham would never have bought it. Pooled
# with a DCF it produced a 76% spread on Apple and suppressed the margin of
# safety on almost every name, for a structural reason rather than a finding.
# 15 x EPS is the same kind of number - criterion 6's ceiling, and one factor
# of the Graham Number - and is shown beside it, not averaged in.
PRICE_LIMIT_MODELS = ('earnings_multiple', 'graham_number', 'ncav', 'ncav_two_thirds')


def consensus_value(models: dict, cfg: ValuationConfig) -> dict:
    """Median of the earnings-based models, with the spread that qualifies it.

    Four exclusions, each for its own reason:

    * **15 x average EPS.** Criterion 6's ceiling, the most Graham's defensive
      investor would pay - and as the middle of three models it once set the
      estimate for two companies in three.
    * **The growth formula.** Graham published it in 1962 and then spent years
      warning against it. It is linear in a forecast growth rate, so including
      it would let the least defensible model move the headline.
    * **The Graham Number.** An asset-and-earnings price limit, not a value
      estimate - see `PRICE_LIMIT_MODELS`.
    * **NCAV.** A liquidation floor. For most profitable going concerns it is
      negative, which says nothing about what the business is worth.

    All are still computed and reported; they are simply reported as what
    they are.
    """
    vals, used = [], []
    for k in VALUE_MODELS:
        m = models.get(k)
        if not m or not m.get('applicable') or m.get('is_rate'):
            continue
        vals.append(float(m['value']))
        used.append(k)

    if not vals:
        return {'value': None, 'n_models': 0, 'models_used': [],
                'dispersion': None, 'reliable': False,
                'note': 'no intrinsic-value model could be computed'}

    arr = np.array(vals, dtype=float)
    median = float(np.median(arr))
    # (high - low) / (high + low): monotonic in high/low and independent of how
    # many models apply, so the 0.5 limit means "highest at most 3x the
    # lowest" whether two models or three are in play. Two earlier measures
    # failed that: IQR for three models and full range for two scored the
    # same disagreement twice as harshly with two; half-range over the median
    # capped two models at 3x but let three span 3.78x (NVIDIA) while a 2.85x
    # spread (Deere) was withheld.
    if len(arr) >= 2 and arr.min() > 0:
        dispersion = float((arr.max() - arr.min()) / (arr.max() + arr.min()))
    else:
        dispersion = None

    reliable = len(arr) >= 2 and (
        dispersion is None or dispersion <= cfg.score.max_model_dispersion)
    note = f'median of {len(arr)} model(s): {", ".join(used)}'
    if dispersion is not None:
        note += f'; highest estimate {arr.max() / arr.min():.1f}x the lowest'
    if len(arr) < 2:
        note += ' - a single model is an opinion, not a range'
    elif not reliable:
        lim = (1 + cfg.score.max_model_dispersion) / (1 - cfg.score.max_model_dispersion)
        note += (f' - more than {lim:.0f}x apart, so the margin of safety is withheld')
    return {'value': median, 'n_models': len(arr), 'models_used': used,
            'dispersion': dispersion, 'reliable': reliable, 'note': note,
            'low': float(arr.min()), 'high': float(arr.max())}


def margin_of_safety(price: float | None, consensus: dict,
                     cfg: ValuationConfig) -> dict:
    """Graham's central idea, expressed as a discount to estimated value.

    Withheld - not reported as zero or as a small number - when the models
    behind it disagree too much to support it. `verdict` distinguishes the
    three cases the dashboard must show differently: a measured margin, a
    measured premium, and a margin that cannot be measured at all.
    """
    s = cfg.score
    if price is None or consensus.get('value') is None:
        return {'mos': None, 'verdict': 'UNKNOWN', 'band': None,
                'note': 'no price or no computable value'}
    value = consensus['value']
    mos = (value - price) / value if value > 0 else None
    # Price as a multiple of estimated value stays legible when the price is
    # far above it. The margin of safety does not: at four times value it
    # reads "-300%", a number that invites being averaged or ranked when it
    # should only ever be read as "three times too expensive on these models".
    p2v = (price / value) if value and value > 0 else None
    if mos is None:
        return {'mos': None, 'price_to_value': p2v, 'verdict': 'UNKNOWN',
                'band': None, 'note': 'estimated value is not positive'}
    if not consensus.get('reliable'):
        return {'mos': mos, 'price_to_value': p2v, 'verdict': 'UNRELIABLE',
                'band': None,
                'note': f'price is {p2v:,.2f}x the median estimate, but the '
                        f'models disagree too much to call that a margin '
                        f'({consensus.get("note")})'}
    if mos >= s.mos_strong:
        band = 'STRONG'
    elif mos >= s.mos_adequate:
        band = 'ADEQUATE'
    elif mos >= 0:
        band = 'THIN'
    else:
        band = 'PREMIUM'
    note = (f'price {price:,.2f} against an estimated {value:,.2f} '
            f'({consensus["n_models"]} models)')
    if band == 'PREMIUM':
        note = (f'price {price:,.2f} is {p2v:,.2f}x the estimated {value:,.2f} '
                f'({consensus["n_models"]} models) - no margin of safety')
    return {'mos': mos, 'price_to_value': p2v, 'verdict': 'MEASURED',
            'band': band, 'note': note}
