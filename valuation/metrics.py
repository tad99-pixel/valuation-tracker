"""Multi-year aggregations shared by the Graham and Buffett criteria.

Everything here returns ``(value, note)`` or None rather than a bare number,
because the interesting cases are the ones where the answer does not exist.
A growth rate measured from a loss-making base year is not a small number or a
large one - it is undefined, and the scanner's hard-won rule applies: a
missing measurement must never print as a measured zero.

**Windows are positional, gaps included.** "The last three years" means the
three most recent fiscal years in the panel; if one of them was not filed, the
statistic says so or declines to exist. An earlier version dropped missing
values first and took the tail second, which reached back across any gap to
whatever happened to be on file: Berkshire's "average EPS of the past three
years" came out as its 2012-14 figures, a decade stale, because nothing later
is tagged in a form this source reads. Every window here is cut from the panel
first and cleaned second, and growth rates are annualised over elapsed time
rather than over a count of observations, which a gap year would inflate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

def _series(series) -> pd.Series:
    """A panel column with infinities as NaN and gaps left where they are."""
    if series is None:
        return pd.Series(dtype=float)
    s = pd.Series(series)
    s = pd.to_numeric(s, errors='coerce')
    return s.replace([np.inf, -np.inf], np.nan)


def tail_years(series, n: int) -> pd.Series:
    """The last `n` fiscal years of the panel, missing years kept as NaN."""
    s = _series(series)
    return s.iloc[-n:] if n else s


def latest(series):
    """The most recent fiscal year's value, or None if that year was not filed.

    Never the last value that happens to exist. Last year's figure is a
    different fact from this year's, and a test of anything "current" that is
    quietly handed an older number answers a question nobody asked.
    """
    s = _series(series)
    if s.empty or pd.isna(s.iloc[-1]):
        return None
    return float(s.iloc[-1])


def _years_axis(s: pd.Series) -> np.ndarray:
    """Each row's position in years: real elapsed time when dates are known."""
    if isinstance(s.index, pd.DatetimeIndex) and len(s):
        return np.asarray((s.index - s.index[0]).days, dtype=float) / 365.25
    return np.arange(len(s), dtype=float)


def _year_label(d) -> str:
    return str(d.year) if hasattr(d, 'year') else str(d)


# ---------------------------------------------------------------------------
# Growth
# ---------------------------------------------------------------------------

def cagr(series, years: int | None = None, min_span: float = 1.0):
    """Compound annual growth between the first and last filed values of a window.

    Undefined - not small, not large - from or to a non-positive value: the
    formula returns a complex number or a nonsense sign flip there, and
    screens that paper over it with absolute values rank a company recovering
    from a loss above one that was profitable throughout.

    The end point must be the most recent fiscal year. A rate that ends in an
    older year because the latest was not filed is a measurement of a
    different decade. The rate is annualised over the time actually elapsed
    between the two values. Returns ``(rate, note)``.
    """
    s = _series(series)
    if years is not None:
        s = s.iloc[-(years + 1):]
    if s.empty:
        return None, 'no history'
    if pd.isna(s.iloc[-1]):
        return None, ('the latest fiscal year is not filed - a rate ending in an older '
                      'year measures a different period')
    pos = np.flatnonzero(s.notna().to_numpy())
    if len(pos) < 2:
        return None, f'need 2+ filed years, have {len(pos)}'
    t = _years_axis(s)
    span = float(t[pos[-1]] - t[pos[0]])
    if span < min_span:
        return None, (f'only {span:.1f} years between the first and last filed values, '
                      f'need {min_span:g}')
    start, end = float(s.iloc[pos[0]]), float(s.iloc[pos[-1]])
    if start <= 0:
        return None, f'base year is {start:,.2f} - growth rate undefined from a non-positive base'
    if end <= 0:
        return None, f'final year is {end:,.2f} - growth rate undefined to a non-positive value'
    label = f'{_year_label(s.index[pos[0]])}-{_year_label(s.index[pos[-1]])}'
    return (end / start) ** (1.0 / span) - 1.0, f'{span:.0f}y ({label})'


def smoothed_cagr(series, years: int = 10, avg: int = 3):
    """CAGR between `avg`-year medians at each end of the window.

    A plain endpoint CAGR on a cyclical business measures where the cycle
    happened to sit on two dates. Exxon's owner earnings "compounded at 13%"
    over ten years because the window opened in the 2015-16 oil trough; fed to
    a DCF and faded over a decade, that one artefact put its value at 260
    against an earnings-power value of 67. Graham specified three-year
    averages at both ends of his growth test for exactly this reason.

    Medians, not means, because these rates feed a valuation. A mean of three
    is still dragged by a single one-off: Coca-Cola's 2017 tax-reform charge
    sat inside the opening block and pushed the "smoothed" rate *above* the
    endpoint rate (11.6% vs 6.1%). The median of three ignores one outlier by
    construction. (Graham's own criterion 5 keeps his means - see
    `graham_eps_growth`.)

    One missing year per block is tolerated. The rate is annualised over the
    time between the block centres, measured from the actual dates.
    """
    s = _series(series).iloc[-(years + 1):]
    if len(s) < 2 * avg + 1:
        return None, f'need {2 * avg + 1}+ fiscal years for a smoothed rate, have {len(s)}'
    ok = s.notna().to_numpy()
    head_pos = [i for i in range(avg) if ok[i]]
    tail_pos = [i for i in range(len(s) - avg, len(s)) if ok[i]]
    need = max(2, avg - 1)
    if len(head_pos) < need or len(tail_pos) < need:
        return None, (f'need {need} of {avg} years filed in each end block; '
                      f'have {len(head_pos)} and {len(tail_pos)}')
    begin, end = float(s.iloc[head_pos].median()), float(s.iloc[tail_pos].median())
    t = _years_axis(s)
    span = float(np.mean(t[tail_pos]) - np.mean(t[head_pos]))
    if span < avg:
        return None, f'block centres only {span:.1f} years apart'
    if begin <= 0:
        return None, (f'opening {avg}-year median is {begin:,.2f} - growth '
                      'undefined from a non-positive base')
    if end <= 0:
        return None, f'closing {avg}-year median is {end:,.2f} - growth undefined'
    return (end / begin) ** (1.0 / span) - 1.0, f'{avg}y medians, centres {span:.0f}y apart'


def graham_eps_growth(eps, window: int = 10, avg: int = 3):
    """Graham's criterion 5: growth in per-share earnings over `window` years,
    between `avg`-year averages at each end.

    The averaging is the whole point of the test: comparing one starting year
    with one ending year lets a single recession year manufacture any growth
    rate you like. The two blocks have centres `window` years apart, so the
    test needs `window + avg` fiscal years on file. With fewer it is
    unmeasurable - an eight-year comparison against a ten-year threshold is a
    different test. One missing year per block is tolerated.

    Returns ``(total_growth, detail)`` where total growth is a fraction.
    """
    s = _series(eps)
    need_rows = window + avg
    if len(s) < need_rows:
        return None, {'note': f'needs {need_rows} fiscal years (two {avg}-year averages '
                              f'{window} years apart); {len(s)} on file'}
    end_block = s.iloc[-avg:].dropna()
    start_block = s.iloc[-(window + avg):-window].dropna()
    need = max(2, avg - 1)
    if len(end_block) < need or len(start_block) < need:
        return None, {'note': (f'EPS missing inside an averaging block '
                               f'({len(start_block)} and {len(end_block)} of {avg} years filed)')}
    begin = float(start_block.mean())
    end = float(end_block.mean())
    detail = {
        'begin_avg': begin,
        'end_avg': end,
        'begin_years': [str(d.date()) if hasattr(d, 'date') else str(d) for d in start_block.index],
        'end_years': [str(d.date()) if hasattr(d, 'date') else str(d) for d in end_block.index],
    }
    if begin <= 0:
        detail['note'] = (f'opening {avg}-year average EPS is {begin:.2f} - '
                          'growth undefined from a non-positive base')
        return None, detail
    detail['note'] = f'{begin:.2f} -> {end:.2f}'
    return end / begin - 1.0, detail


# ---------------------------------------------------------------------------
# Levels and records
# ---------------------------------------------------------------------------

def trailing_average(series, n: int, min_valid: int | None = None):
    """Mean over the last `n` fiscal years, or None.

    Needs `min_valid` filed values among the last ``min(n, len)`` rows.
    Graham's "average earnings of the past three years" needs all three (the
    default, `min_valid` = `n`); a one-year "average" is a different and much
    noisier statistic, and substituting it is how a cyclical peak gets valued
    as a norm. Cash-flow and earning-power bases tolerate a gap or two and
    pass a lower `min_valid`.
    """
    need = n if min_valid is None else min_valid
    w = _series(series).iloc[-n:]
    v = w.dropna()
    if len(v) < need:
        missing = ', '.join(_year_label(d) for d, x in w.items() if pd.isna(x))
        return None, (f'needs {need} of the last {n} fiscal years filed, has {len(v)}'
                      + (f' (missing {missing})' if missing else ''))
    note = f'{n}y mean' if len(v) == n else f'mean of the {len(v)} filed of the last {n} years'
    return float(v.mean()), note


def positive_years(series, window: int):
    """Positive years among the last `window`, how many were filed, and the losses."""
    v = tail_years(series, window).dropna()
    neg = [str(d.date()) if hasattr(d, 'date') else str(d) for d, x in v.items() if x <= 0]
    return int((v > 0).sum()), int(len(v)), neg


def consecutive_streak(series, predicate) -> int:
    """Length of the current run satisfying `predicate`, counting back.

    A missing value ends the run: an unbroken record cannot be proven across
    a year nobody filed.
    """
    n = 0
    for v in reversed(_series(series).tolist()):
        if not pd.isna(v) and predicate(v):
            n += 1
        else:
            break
    return n


def dividend_record(panel: pd.DataFrame, want_years: int):
    """Graham's criterion 4, reported against the history that actually exists.

    Graham asked for twenty years of uninterrupted dividends. XBRL begins
    around 2009, so for most companies twenty years cannot be checked from
    this source at all, and reporting FAIL there would confuse an
    unverifiable claim with a refuted one. An interruption *inside* the
    visible record is a different matter: it refutes the claim outright,
    however short the record.

    Each year is one of three states: **paid**, **not paid** (a dividend tag
    filed as zero), or **unknown** (no dividend tag this source can read). An
    unknown year cannot refute anything. An earlier version filled unknown
    years with zero, and Colgate - which has paid since 1895 but tagged
    2007-16 dividends under a tag outside the chain - failed as "refuted",
    along with Travelers and Simon Property.

    The one inference made: years with statements filed *after* the last
    paying year, and no dividend recorded, are not paid. A company that paid
    last year and files a cash-flow statement with no dividend line has
    stopped - Intel, 2025.
    """
    dps = panel.get('dividends_per_share')
    paid = panel.get('dividends_paid')
    if dps is None and paid is None:
        return {'streak': 0, 'window': 0, 'verifiable': False, 'interrupted': False,
                'pays_now': False, 'note': 'no dividend tags filed'}

    nan = pd.Series(np.nan, index=panel.index)
    d = _series(dps) if dps is not None else nan
    c = _series(paid).abs() if paid is not None else nan
    is_paid = (d > 0) | (c > 0)
    is_zero = ((d == 0) | (c == 0)) & ~is_paid
    state = pd.Series(np.nan, index=panel.index)
    state[is_paid] = 1.0
    state[is_zero] = 0.0

    statements = (panel['cfo'].notna() if 'cfo' in panel.columns
                  else pd.Series(True, index=panel.index))
    paid_pos = np.flatnonzero(state.eq(1.0).to_numpy())
    if len(paid_pos):
        for i in range(paid_pos[-1] + 1, len(state)):
            if pd.isna(state.iloc[i]) and bool(statements.iloc[i]):
                state.iloc[i] = 0.0

    streak, stop_at = 0, None
    for d_, v in zip(reversed(state.index), reversed(state.tolist())):
        if v == 1.0:
            streak += 1
        else:
            stop_at = (d_, 'zero' if v == 0.0 else 'unknown')
            break

    window = int(len(state))
    interrupted = stop_at is not None and stop_at[1] == 'zero'
    out = {
        'streak': int(streak),
        'window': window,
        'verifiable': streak >= want_years or interrupted,
        'interrupted': interrupted,
        'pays_now': bool(len(state) and state.iloc[-1] == 1.0),
    }
    if interrupted:
        out['note'] = (f'no dividend in fiscal {_year_label(stop_at[0])} - current streak '
                       f'{streak}y, so {want_years}y uninterrupted is refuted')
    elif out['verifiable']:
        out['note'] = f'{streak}y uninterrupted'
    elif stop_at is None:
        out['note'] = (f'{streak}y unbroken across the whole {window}y record; Graham '
                       f'asks for {want_years}y, which XBRL cannot reach')
    else:
        since = (f' since fiscal {_year_label(state.index[window - streak])}'
                 if streak else '')
        out['note'] = (f'{streak}y unbroken{since}; no readable dividend tag for fiscal '
                       f'{_year_label(stop_at[0])}, so {want_years}y cannot be verified')
    return out


def count_declines(series, threshold: float = 0.05, window: int = 10):
    """Year-on-year EPS declines of `threshold` or more, for the 1976 test K5.

    Graham's test covers the past ten years, so the window is the last
    `window` year-pairs, not the whole history. A pair with a missing year is
    not measurable; a pair from a loss year is skipped because a percentage is
    undefined there (a fall from profit into loss does count).

    Returns ``(declines, note, complete)``. `complete` is False when pairs are
    missing, in which case a count at or under the limit proves nothing - a
    decline may sit in the gap - while a count over it is already decisive.
    """
    vals = tail_years(series, window + 1).to_numpy()
    n = checked = skipped = 0
    for prev, cur in zip(vals[:-1], vals[1:]):
        if np.isnan(prev) or np.isnan(cur):
            continue
        checked += 1
        if prev <= 0:
            skipped += 1
            continue
        if (cur - prev) / prev <= -threshold:
            n += 1
    if checked == 0:
        return None, 'no two consecutive fiscal years of EPS filed', False
    complete = checked >= window
    note = (f'{n} decline(s) of {threshold:.0%} or more across {checked} of the '
            f'last {window} year-pairs')
    if skipped:
        note += f'; {skipped} pair(s) start from a loss, where a percentage is undefined'
    if not complete:
        note += f'; {window - checked} pair(s) not measurable'
    return n, note, complete


# ---------------------------------------------------------------------------
# Owner earnings
# ---------------------------------------------------------------------------

def maintenance_capex(panel: pd.DataFrame, method: str = 'greenwald'):
    """Estimate the capex needed to hold position, which nobody discloses.

    Buffett's 1986 definition of owner earnings subtracts "the average annual
    amount of capitalized expenditures for plant and equipment etc. that the
    business requires to fully maintain its long-term competitive position and
    its unit volume" - and says in the same appendix that this figure is a
    guess. It is not in any filing, so it has to be estimated, and the
    estimate is the single largest source of uncertainty in owner earnings.

    Two estimators, both reported:

    ``greenwald``
        Growth capex is the PP&E intensity of the business applied to the
        change in sales; whatever remains of capex is maintenance. A year in
        which sales shrank spends nothing on growth, so all of its capex is
        maintenance.
    ``da``
        Treat depreciation and amortisation as the maintenance requirement.
        Cheap, standard, and systematically wrong for a business whose assets
        cost more to replace than their historical book value.

    Returns ``(Series, note)``.
    """
    capex = panel.get('capex')
    if capex is None or capex.notna().sum() == 0:
        return None, 'no capex filed'
    capex = _series(capex).abs()

    if method == 'da':
        da = panel.get('depreciation_amortisation')
        if da is None or da.notna().sum() == 0:
            return None, 'no D&A filed'
        return _series(da).abs(), 'D&A as the maintenance requirement'

    rev = panel.get('revenue')
    ppe = panel.get('ppe_net')
    if rev is None or ppe is None or rev.notna().sum() < 2 or ppe.notna().sum() < 2:
        da = panel.get('depreciation_amortisation')
        if da is not None and da.notna().sum():
            return _series(da).abs(), 'D&A fallback - revenue or PP&E missing for the Greenwald estimate'
        return None, 'insufficient data for either estimator'

    rev, ppe = _series(rev), _series(ppe)
    # PP&E intensity moves slowly, so a year whose PP&E was not filed takes
    # the last known ratio rather than voiding that year's owner earnings -
    # Kroger's PP&E tag changed after ASC 842 and every later year went blank.
    intensity = (ppe / rev.replace(0, np.nan)).ffill()
    d_sales = rev.diff()
    growth_capex = (intensity * d_sales).clip(lower=0)
    maint = (capex - growth_capex).clip(lower=0)
    # In the first year there is no change in sales to work from.
    maint.iloc[0] = np.nan
    return maint, 'capex less (PP&E/sales x change in sales)'


def owner_earnings(panel: pd.DataFrame, method: str = 'greenwald',
                   group: str = 'operating'):
    """Buffett's owner earnings: net income + non-cash charges - maintenance capex.

    Returns ``(Series, detail)``. The detail records which maintenance-capex
    estimator was used, because the two can differ by a factor of two on a
    capital-intensive business and the number is meaningless without it.

    A year with no D&A filed is unknown, not a year with zero D&A - treating it
    as zero would understate that year's owner earnings by the whole
    depreciation charge. Only a company that files no D&A at all is computed
    without it, and flagged.

    **REITs** get funds from operations instead - net income plus
    depreciation, with no maintenance deduction. A REIT's "capex" line is
    mostly property *acquisitions*, so the Greenwald estimator treats the
    growth of the portfolio as the cost of standing still: Realty Income's
    owner earnings came out at $0.8bn against roughly $3.5bn of FFO. FFO
    overstates owner earnings for a REIT with material recurring capex
    (offices, apartments); for a net-lease REIT, where tenants carry the
    maintenance, it is close to exact.
    """
    ni = panel.get('net_income')
    if ni is None or ni.notna().sum() == 0:
        return None, {'note': 'no net income filed'}
    ni = _series(ni)
    da = panel.get('depreciation_amortisation')
    da = _series(da).abs() if da is not None else pd.Series(np.nan, index=panel.index)
    has_da = bool(da.notna().any())

    if group == 'reit':
        if not has_da:
            return None, {'note': 'REIT with no depreciation filed - FFO not computable'}
        ffo = ni + da
        return ffo, {
            'method': 'ffo',
            'maintenance_note': 'FFO proxy (net income + D&A); property acquisitions are '
                                'growth, not maintenance',
            'da_available': True, 'n_years': int(ffo.notna().sum()),
        }

    maint, note = maintenance_capex(panel, method=method)
    if maint is None:
        return None, {'note': f'maintenance capex not estimable: {note}'}

    oe = (ni + da - maint) if has_da else (ni - maint)
    detail = {
        'method': method,
        'maintenance_note': note,
        'da_available': has_da,
        'n_years': int(oe.notna().sum()),
    }
    if not has_da:
        detail['warning'] = ('no D&A filed in any year, so only the capex deduction '
                             'is applied - owner earnings understated')
    return oe, detail


def margin_stability(panel: pd.DataFrame, column: str = 'gross_margin',
                     years: int = 10):
    """Coefficient of variation of a margin - a proxy for a durable advantage.

    A business with pricing power holds its margin through a cycle. The level
    says how good the business is; the dispersion says whether the advantage
    is durable, which is the question Buffett's "moat" is actually about.
    """
    v = tail_years(panel.get(column), years).dropna()
    if len(v) < 4:
        return None, f'need 4+ of the last {years} years, have {len(v)}'
    mean = float(v.mean())
    if abs(mean) < 1e-9:
        return None, 'mean margin is zero'
    cv = float(v.std(ddof=1)) / abs(mean)
    return cv, f'cv over {len(v)}y, mean {mean:.1%}'


def retained_earnings_test(panel: pd.DataFrame, window: int = 10):
    """Buffett's one-dollar premise (1984 letter) - the retained side.

    "Unrestricted earnings should be retained only where there is a reasonable
    prospect - backed preferably by historical evidence... - that for every
    dollar retained by the corporation, at least one dollar of market value
    will be created for owners."

    Retained means kept: net income less dividends **and less buybacks**.
    Cash spent repurchasing stock has left the company as surely as a
    dividend, and the market value this is set against falls by the same
    cash as the share count shrinks. Counting buybacks as retained charged a
    buyback company on both sides of the ratio: Kimberly-Clark, Altria,
    Kroger, 3M, Nike and Wells Fargo "failed" on earnings they had paid out,
    and Apple "passed" on $770bn it had not kept.

    Every year's net income is required: a missing year would understate the
    retained total and flatter the ratio. Dividends come from the cash-flow
    statement where filed, otherwise from dividends per share times shares; a
    year with no buyback tag is a year without buybacks. Returns ``(retained,
    detail)``, with ``detail['nothing_retained']`` set when the company paid
    out everything it earned - the premise then has nothing to test.
    """
    ni = tail_years(panel.get('net_income'), window)
    if len(ni) < window or ni.isna().any():
        return None, {'note': f'needs net income for each of the last {window} fiscal '
                              f'years; {int(ni.notna().sum())} filed'}

    paid_all = _series(panel.get('dividends_paid')).abs()
    dps_all = _series(panel.get('dividends_per_share'))
    paid = paid_all.reindex(ni.index)
    alt = (dps_all * _series(panel.get('shares_used'))).reindex(ni.index)
    div = paid.fillna(alt)
    payer = bool((paid_all.fillna(0) > 0).any() or (dps_all.fillna(0) > 0).any())
    gaps = int(div.isna().sum()) if payer else 0
    div = div.fillna(0)
    bought = _series(panel.get('buybacks')).abs().reindex(ni.index).fillna(0)

    earned, paid_out = float(ni.sum()), float(div.sum()) + float(bought.sum())
    retained = earned - paid_out
    if retained <= 0:
        return None, {'note': (f'nothing retained over the last {window} years: '
                               f'${earned / 1e9:,.1f}bn earned, ${div.sum() / 1e9:,.1f}bn paid '
                               f'as dividends and ${bought.sum() / 1e9:,.1f}bn in buybacks'),
                      'nothing_retained': True, 'earned': earned}
    note = f'{window}y earnings less dividends and buybacks'
    if gaps:
        note += (f'; no dividend recorded for {gaps} year(s) of a paying company, '
                 'counted as unpaid')
    return retained, {'retained': retained, 'earned': earned, 'years': window, 'note': note}


def share_count_trend(panel: pd.DataFrame, years: int = 10):
    """Change in the share count: dilution is a cost the income statement hides."""
    s = tail_years(panel.get('shares_used'), years + 1)
    if s.notna().sum() < 3:
        return None, f'need 3+ filed years, have {int(s.notna().sum())}'
    rate, note = cagr(s, min_span=3)
    if rate is None:
        return None, note
    direction = 'shrinking' if rate < -0.001 else ('growing' if rate > 0.001 else 'flat')
    return rate, f'{direction} {abs(rate):.2%}/yr, {note}'


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarise(panel: pd.DataFrame, cfg, group: str = 'operating') -> dict:
    """The derived numbers both criteria modules need, computed once."""
    b, g = cfg.buffett, cfg.graham
    out = {}

    eps = panel.get('eps')
    out['eps_latest'] = latest(eps)
    out['eps_avg3'], out['eps_avg3_note'] = trailing_average(eps, 3)
    out['eps_growth_total'], out['eps_growth_detail'] = graham_eps_growth(
        eps, g.growth_window_years, g.growth_average_years)
    out['eps_cagr10'], out['eps_cagr10_note'] = cagr(eps, 10, min_span=8)
    (out['eps_declines'], out['eps_declines_note'],
     out['eps_declines_complete']) = count_declines(eps, g.enterprising_decline_threshold, 10)
    out['eps_positive'], out['eps_years'], out['eps_negative_years'] = positive_years(
        eps, g.earnings_stability_years)

    out['dividend'] = dividend_record(panel, g.dividend_years)
    out['share_count_cagr'], out['share_count_note'] = share_count_trend(
        panel, b.quality_window_years)

    # The growth credited to a valuation is the LOWER of the company's total
    # growth and its growth per share.
    #
    # Per share where shares were issued: the DCF divides by today's share
    # count, and growth bought with new stock belongs to the new holders.
    # Realty Income's owner earnings compounded at ~15% while its share count
    # compounded at ~14%; crediting today's holders with the total rate valued
    # them on growth they had been diluted out of.
    #
    # Total where shares were bought back. The DCF already counts every dollar
    # of owner earnings as value to today's shares, including the dollars
    # later spent on buybacks; per-share growth is pushed up by those same
    # buybacks, so crediting it counted the buyback twice - once as cash paid
    # out, once as growth. That had inflated Apple's DCF by a fifth.
    #
    # With too few per-share years, the per-share rate is the total rate
    # converted by the share-count trend; with no share trend either, the
    # total rate stands and the basis says dilution could not be checked.
    shares = _series(panel.get('shares_used'))
    sc = out['share_count_cagr']

    def credited_rate(total):
        """``(rate, basis)``: the lower of total and per-share growth."""
        if total is None:
            return None, None
        agg = smoothed_cagr(total, 10, 3)[0]
        if len(shares) and tail_years(shares, 11).notna().sum() >= 7:
            ps = smoothed_cagr(_series(total) / shares.replace(0, np.nan), 10, 3)[0]
        else:
            ps = (1 + agg) / (1 + sc) - 1 if agg is not None and sc is not None else None
        if ps is None:
            return agg, ('total, dilution unchecked' if agg is not None else None)
        if agg is None:
            return ps, 'per share'
        if ps < agg:
            return ps, 'per share, net of dilution'
        return agg, ('total, buybacks not credited' if ps > agg + 1e-9 else 'total')

    oe, _ = owner_earnings(panel, b.maintenance_capex_method, group)
    out['owner_earnings_latest'] = latest(oe) if oe is not None else None
    out['owner_earnings_avg'] = trailing_average(oe, 5, min_valid=3)[0] if oe is not None else None
    out['owner_earnings_cagr'], out['owner_earnings_cagr_basis'] = credited_rate(oe)
    out['revenue_cagr'], out['revenue_cagr_basis'] = credited_rate(panel.get('revenue'))

    # The alternative estimator, so the dashboard can show the spread rather
    # than presenting one guess as the answer.
    oe_da, _ = owner_earnings(panel, 'da', group)
    out['owner_earnings_da_latest'] = latest(oe_da) if oe_da is not None else None

    out['fcf_avg'], _ = trailing_average(panel.get('fcf'), 5, min_valid=3)
    out['gross_margin_cv'], out['gross_margin_cv_note'] = margin_stability(
        panel, 'gross_margin', b.quality_window_years)

    # Levels across the cycle: the median and how many years it rests on.
    for col in ('roe', 'roic', 'gross_margin', 'net_margin', 'fcf_margin'):
        v = tail_years(panel.get(col), b.quality_window_years).dropna()
        out[f'{col}_median'] = float(v.median()) if len(v) else None
        out[f'{col}_n'] = int(len(v))
        if col == 'roe':
            out['roe_years_above'] = int((v >= b.min_roe).sum())
        elif col == 'roic':
            out['roic_years_above'] = int((v >= b.min_roic).sum())
    for col in ('current_ratio', 'debt_to_equity'):
        out[f'{col}_latest'] = latest(panel.get(col))
    return out
