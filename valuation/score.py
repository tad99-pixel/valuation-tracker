"""Roll the criteria up into one row per company - and refuse to, when the
evidence is too thin to support a number.

The composite score is a sorting device, nothing more. Neither Graham nor
Buffett proposed weighting a set of tests into a single figure, and the
per-criterion table is where the analysis actually is. The score exists so a
watchlist of two hundred names can be put in an order worth reading top-down.

What the score is careful about is the denominator. A company whose filings
support eight of the twenty tests is not comparable with one that supports all
twenty, so the score is computed over *applicable* criteria only, the share
that could actually be measured is reported beside it, and below
`min_data_completeness` the verdict becomes INSUFFICIENT DATA rather than a
number. This is the scanner's rule carried across: a missing measurement must
never print as a measured zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import buffett as B
from . import graham as G
from . import metrics as M
from .config import CONFIG_VERSION, ScoreConfig, ValuationConfig
from .fundamentals import PER_SHARE_COLUMNS
from .graham import FAIL, NA, PASS

# Every verdict, in display order - the single list the tracker's RunLog
# columns, the dashboard's badges and filters, and the run summary read.
# (label, RunLog column, status glyph, counts as a candidate)
VERDICTS = (
    ('CANDIDATE', 'candidates', 'good', True),
    ('CHEAP, QUALITY PASSES', 'cheap_quality', 'good', True),
    ('GOOD BUSINESS, FAIR PRICE', 'good_fair_price', 'warning', False),
    ('STATISTICALLY CHEAP ONLY', 'cheap_only', 'warning', False),
    ('GOOD BUSINESS, DEAR', 'good_but_dear', 'warning', False),
    ('GOOD BUSINESS, VALUE UNCLEAR', 'good_value_unclear', 'neutral', False),
    ('WATCH', 'watch', 'warning', False),
    ('NO', 'no', 'neutral', False),
    ('INSUFFICIENT DATA', 'insufficient', 'none', False),
    ('NO PRICE', 'no_price', 'none', False),
)
VERDICT_SLUGS = {label: slug for label, slug, _, _ in VERDICTS}


def _tally(criteria: list) -> dict:
    passed = sum(1 for c in criteria if c.verdict == PASS)
    failed = sum(1 for c in criteria if c.verdict == FAIL)
    na = sum(1 for c in criteria if c.verdict == NA)
    applicable = passed + failed
    return {
        'passed': passed, 'failed': failed, 'na': na,
        'applicable': applicable, 'total': len(criteria),
        'rate': (passed / applicable) if applicable else None,
    }


def _market_value_change(prices: pd.DataFrame | None, panel: pd.DataFrame,
                         shares_now, window_years: int, basis_note=None):
    """Change in market capitalisation over the window, for the 1984 test.

    Needs a price from the start of the window and the share count that
    prevailed then, both of which the tracker has. Returns None rather than
    approximating when the price history is too short - the test is about a
    decade of capital allocation and running it over three years answers a
    different question.
    """
    if prices is None or prices.empty or 'Close' not in prices:
        return None, 'no price history'
    close = prices['Close'].dropna()
    if close.empty:
        return None, 'no price history'
    start = close.index[-1] - pd.DateOffset(years=window_years)
    early = close.loc[close.index <= start]
    if early.empty:
        span = (close.index[-1] - close.index[0]).days / 365.25
        return None, (f'price history spans {span:.1f}y, need {window_years}y')

    p0, p1 = float(early.iloc[-1]), float(close.iloc[-1])
    if basis_note:
        # The historical share count is a per-share fact; the same guard as
        # every other price comparison applies.
        return None, basis_note
    if not shares_now:
        # The same rule as the valuation models: without a count vetted
        # against the price there is no market value, only a product of two
        # numbers that may describe different shares.
        return None, 'no share count on the same basis as the price'
    sh = panel.get('shares_used')
    if sh is None or sh.notna().sum() == 0:
        return None, 'no share count history'
    sh = sh.dropna()
    near = sh.index[np.argmin(np.abs((sh.index - start).days))]
    if abs((near - start).days) > 400:
        return None, f'no share count within a year of {start.year}'
    s0 = float(sh.loc[near])
    s1 = float(shares_now)
    return (p1 * s1) - (p0 * s0), f'{window_years}y: {p0:,.2f} -> {p1:,.2f}'


def _truncate_at_break(panel: pd.DataFrame, info: dict) -> tuple:
    """Keep per-share history only from the last uncorrectable basis break on.

    Per-share columns (`fundamentals.PER_SHARE_COLUMNS`) are blanked before
    the break so no test reaches across it; dollar columns keep everything.

    Discarding the whole history would throw away a decade of good dollar
    figures to protect a per-share comparison; truncating at the break keeps
    everything that is on today's basis. The ten-year tests then find too few
    years and report themselves unmeasurable, which is the correct outcome.
    """
    brk = info.get('share_basis_break')
    if info.get('per_share_reliable', True) or brk is None:
        return panel, None
    p = panel.copy()
    before = p.index < pd.Timestamp(brk)
    for c in PER_SHARE_COLUMNS:
        if c in p.columns:
            p.loc[before, c] = np.nan
    kept = int((~before).sum())
    return p, (f'per-share history used only from {pd.Timestamp(brk).year} '
               f'({kept} years): {info.get("share_basis_note")}')


def _recent(panel: pd.DataFrame, col: str, rows: int = 2):
    """The most recent filed value within the last `rows` fiscal years."""
    if col not in panel.columns:
        return None
    v = panel[col].iloc[-rows:].dropna()
    return float(v.iloc[-1]) if len(v) else None


def _implausible_market_value(price, shares, panel: pd.DataFrame):
    """Why price x shares cannot be a real market value - or None if it can.

    A share count on a different basis from the quoted price - Berkshire's
    Class A count against a Class B price is 1,500x - produces a market value
    absurd against both revenue and book. Requiring both ratios to be absurd
    keeps the check from misfiring on real extremes: a near-bankrupt retailer
    can trade at 0.003x sales and a pre-revenue biotech at 1,000x, but neither
    is also at 0.001x or 5,000x book. Where only one ratio is computable it
    decides alone.
    """
    if not price or not shares:
        return None
    mcap = price * shares
    rev, eq = _recent(panel, 'revenue'), _recent(panel, 'equity')
    ps = mcap / rev if rev and rev > 0 else None
    pb = mcap / eq if eq and eq > 0 else None
    bad_ps = ps is not None and not (0.002 <= ps <= 500)
    bad_pb = pb is not None and not (0.01 <= pb <= 1000)
    bad = (bad_ps and bad_pb) if (ps is not None and pb is not None) else (bad_ps or bad_pb)
    if not bad:
        return None
    parts = []
    if ps is not None:
        parts.append(f'{ps:.3g}x revenue')
    if pb is not None:
        parts.append(f'{pb:.3g}x book')
    return (f'price x {shares:,.0f} shares is {" and ".join(parts)}, which no company '
            'trades at - the count and the price refer to different share classes')


def _check_price_basis(price, panel: pd.DataFrame, info: dict):
    """Withdraw the vetted current share count if it cannot match the price."""
    why = _implausible_market_value(price, info.get('shares_outstanding_now'), panel)
    if why is None:
        return None
    info['shares_outstanding_now'] = None
    info['shares_now_source'] = 'rejected - ' + why
    return info['shares_now_source']


def _per_share_basis(price, panel: pd.DataFrame, info: dict):
    """Can the filing's per-share figures be set against the quoted price?

    EPS and book value per share are stated on the filer's share basis, the
    price on the traded class's. They agree for almost every company and
    disagree badly for a few multi-class filers, so before any P/E, P/B or
    per-share model compares the two, price x the latest fiscal year's share
    count must describe a plausible company. A latest year with no share count
    cannot be checked at all, and is treated the same way: an unverifiable
    P/E is not reported as a P/E.

    Sets ``info['per_share_basis_note']``, which every price comparison in
    `graham` and `buffett` reads, and returns it.
    """
    note = None
    if price and not info.get('splits_known', True):
        # No verified split history: per-share facts may sit on a pre-split
        # basis while the price is post-split, an error of exactly the split
        # ratio that no plausibility band catches.
        note = ('per-share figures withheld from price comparisons: the split history '
                'could not be fetched, so the per-share basis cannot be confirmed')
    elif price:
        sh = panel['shares_used'].iloc[-1] if 'shares_used' in panel.columns and len(panel) else np.nan
        if pd.isna(sh) or sh <= 0:
            note = ('per-share figures withheld from price comparisons: the latest fiscal '
                    'year has no share count to tie them to the quoted share class')
        else:
            why = _implausible_market_value(price, float(sh), panel)
            if why:
                note = 'per-share figures withheld from price comparisons: ' + why
    info['per_share_basis_note'] = note
    return note


def evaluate(ticker: str, panel: pd.DataFrame, info: dict,
             prices: pd.DataFrame | None, macro: dict,
             cfg: ValuationConfig) -> dict:
    """Run every test on one company and return a single structured result."""
    info = dict(info)

    price, price_date, stale_days = None, None, None
    if prices is not None and not prices.empty and 'Close' in prices:
        close = prices['Close'].dropna()
        if len(close):
            price = float(close.iloc[-1])
            price_date = close.index[-1]
            stale_days = int((pd.Timestamp.today().normalize() - price_date).days)

    panel, truncation = _truncate_at_break(panel, info)
    basis_problem = _check_price_basis(price, panel, info)
    per_share_problem = _per_share_basis(price, panel, info)
    summary = M.summarise(panel, cfg, info.get('group', 'operating'))

    mv_change, mv_note = _market_value_change(
        prices, panel, info.get('shares_outstanding_now'),
        cfg.buffett.one_dollar_window_years, info.get('per_share_basis_note'))
    info['market_value_change'] = mv_change
    info['market_value_note'] = mv_note

    pe_high, pe_high_note = G.pe_five_year_high(panel, prices)

    defensive = G.defensive_criteria(panel, info, summary, price, macro, cfg)
    enterprising = G.enterprising_criteria(panel, info, summary, price, macro,
                                           cfg, pe_high, pe_high_note)
    quality = B.quality_criteria(panel, info, summary, price, cfg)

    models = B.intrinsic_values(panel, info, summary, price, macro, cfg)
    consensus = B.consensus_value(models, cfg)
    mos = B.margin_of_safety(price, consensus, cfg)

    t_def, t_ent, t_qual = _tally(defensive), _tally(enterprising), _tally(quality)

    # --- data completeness, over tests that could apply at all ------------
    all_criteria = defensive + enterprising + quality
    # An N/A for a structural reason (a bank has no current ratio) is not a
    # data gap - it is a test that does not exist for this company, and
    # counting it against completeness would mark every bank as unanalysable.
    # The flag is set where the carve-out is decided, not inferred here from
    # the wording of the message.
    structural = sum(1 for c in all_criteria if c.verdict == NA and c.structural)
    could_apply = len(all_criteria) - structural
    measured = sum(1 for c in all_criteria if c.verdict in (PASS, FAIL))
    completeness = (measured / could_apply) if could_apply else 0.0

    # --- composite -------------------------------------------------------
    s = cfg.score
    parts, weights = [], []
    if t_def['rate'] is not None:
        parts.append(t_def['rate']); weights.append(s.weight_graham_defensive)
    if t_qual['rate'] is not None:
        parts.append(t_qual['rate']); weights.append(s.weight_buffett_quality)

    # The valuation component is ALWAYS in the composite. An earlier version
    # dropped it when the margin of safety could not be measured and
    # re-normalised over what remained - which meant JPMorgan's composite
    # *rose* from 0.47 to 0.67 the moment its valuation became unmeasurable.
    # A company was being rewarded for having too little data to value.
    #
    # Graham puts the burden of proof on the margin of safety: a margin you
    # cannot demonstrate is not one you have. So an unmeasured margin scores
    # zero, exactly like a measured premium, and can never outrank a measured
    # discount.
    if mos['verdict'] == 'MEASURED':
        # Map the margin onto 0..1: no margin scores zero, Graham's one-third
        # scores full marks. Clipped, so a wildly cheap-looking estimate - more
        # often a broken model than a bargain - cannot dominate the composite.
        val_component = float(np.clip(mos['mos'] / s.mos_strong, 0.0, 1.0))
    else:
        val_component = 0.0
    parts.append(val_component); weights.append(s.weight_valuation)

    if price is None:
        # Usually a delisting or an acquisition - Walgreens, taken private in
        # 2025, still has a decade of filings and no quote. Scoring it on the
        # quality side alone and calling the result "NO" would read as a
        # judgement on a business that can no longer be bought at all.
        composite = None
        verdict = 'NO PRICE'
    elif parts and completeness >= s.min_data_completeness:
        composite = float(np.average(parts, weights=weights))
        verdict = _verdict_band(composite, mos, t_def, t_qual, s)
    else:
        composite = None
        verdict = 'INSUFFICIENT DATA'

    reasons = []
    if price is None:
        reasons.append('no current price - delisted, acquired, or the ticker has changed')
    if completeness < s.min_data_completeness:
        reasons.append(f'only {completeness:.0%} of applicable tests measurable '
                       f'(need {s.min_data_completeness:.0%})')
    if info.get('filing_stale'):
        reasons.append(f'last annual filing is {info.get("last_fy_age_days")} days old')
    if truncation:
        reasons.append(truncation)
    elif not info.get('per_share_reliable', True):
        reasons.append(info.get('share_basis_note') or 'per-share history unreliable')
    if basis_problem:
        reasons.append('cash-flow valuations withheld: ' + basis_problem)
    elif not info.get('shares_outstanding_now') and price is not None:
        reasons.append('cash-flow valuations withheld: ' + (info.get('shares_now_source') or 'no share count'))
    if per_share_problem:
        reasons.append(per_share_problem)
    if info.get('share_unit_fixes'):
        reasons.append('share counts filed at the wrong scale were corrected ('
                       + ', '.join(info['share_unit_fixes']) + ')')
    if info.get('share_events'):
        # Real events, kept in the per-share history - but a ten-year growth
        # rate that spans a merger is not organic growth, and it should say so.
        reasons.append('per-share history spans large share-count changes that are not '
                       'splits (' + '; '.join(info['share_events']) + ')')
    if stale_days is not None and stale_days > 7:
        reasons.append(f'price is {stale_days} days stale')
    if info.get('facts_source') == 'stale-cache':
        reasons.append('fundamentals served from an out-of-date cache')

    return {
        'ticker': ticker,
        'name': info.get('name'),
        'group': info.get('group'),
        'sic_description': info.get('sic_description'),
        'price': price,
        'price_date': price_date,
        'price_stale_days': stale_days,
        'n_years': info.get('n_years'),
        'last_fy': info.get('last_fy'),
        'composite': composite,
        'verdict': verdict,
        'caveats': reasons,
        'completeness': completeness,
        'defensive': defensive, 'defensive_tally': t_def,
        'enterprising': enterprising, 'enterprising_tally': t_ent,
        'quality': quality, 'quality_tally': t_qual,
        'models': models, 'consensus': consensus, 'margin_of_safety': mos,
        'valuation_component': val_component,
        'summary': summary, 'info': info,
        # The panel as the tests saw it - per-share history truncated at an
        # uncorrectable break - so the dashboard charts what was measured.
        'panel_used': panel,
        'pe_5yr_high': pe_high, 'pe_5yr_high_note': pe_high_note,
        'config_version': CONFIG_VERSION,
    }


def _verdict_band(composite: float, mos: dict, t_def: dict, t_qual: dict,
                  s: ScoreConfig | None = None) -> str:
    """A label from `VERDICTS`, deliberately coarse.

    Finer gradations would imply a precision the inputs do not have. The
    valuation side and the quality side are kept distinct, because a cheap bad
    business and an expensive good one are different mistakes and a single
    number hides which one is on offer.
    """
    s = s or ScoreConfig()
    measured = mos['verdict'] == 'MEASURED'
    cheap = measured and mos.get('band') in ('STRONG', 'ADEQUATE')
    fair = measured and mos.get('band') == 'THIN'
    dear = measured and mos.get('band') == 'PREMIUM'
    quality_ok = (t_qual['rate'] or 0) >= s.quality_pass_rate
    graham_ok = (t_def['rate'] or 0) >= s.graham_pass_rate

    if quality_ok and cheap and graham_ok:
        return 'CANDIDATE'
    if quality_ok and cheap:
        return 'CHEAP, QUALITY PASSES'
    if quality_ok and fair:
        # "It's far better to buy a wonderful company at a fair price than a
        # fair company at a wonderful price." The scorer once had no label
        # for this, and Nike - eight of nine quality tests passed, priced at
        # 0.99x its estimate - fell through to NO.
        return 'GOOD BUSINESS, FAIR PRICE'
    if quality_ok and dear:
        return 'GOOD BUSINESS, DEAR'
    if quality_ok:
        return 'GOOD BUSINESS, VALUE UNCLEAR'
    if cheap:
        return 'STATISTICALLY CHEAP ONLY'
    if composite >= s.watch_composite:
        return 'WATCH'
    return 'NO'


def to_row(result: dict) -> dict:
    """Flatten one result into a tracker / dashboard row."""
    c = result['consensus']
    m = result['margin_of_safety']
    s = result['summary']
    models = result['models']

    def mv(key):
        d = models.get(key) or {}
        return d.get('value') if d.get('applicable') else None

    rev = models.get('reverse_dcf') or {}
    dcf_detail = (models.get('dcf') or {}).get('detail') or {}
    measured = m.get('verdict') == 'MEASURED'
    return {
        'Ticker': result['ticker'],
        'Name': result['name'],
        'Group': result['group'],
        'Industry': result['sic_description'],
        'Price': result['price'],
        'PriceDate': result['price_date'],
        'Verdict': result['verdict'],
        'Composite': result['composite'],
        'Completeness': result['completeness'],
        'GrahamDefensivePassed': result['defensive_tally']['passed'],
        'GrahamDefensiveApplicable': result['defensive_tally']['applicable'],
        'GrahamEnterprisingPassed': result['enterprising_tally']['passed'],
        'GrahamEnterprisingApplicable': result['enterprising_tally']['applicable'],
        'BuffettQualityPassed': result['quality_tally']['passed'],
        'BuffettQualityApplicable': result['quality_tally']['applicable'],
        'IntrinsicValue': c.get('value'),
        'IntrinsicLow': c.get('low'),
        'IntrinsicHigh': c.get('high'),
        'ModelsUsed': c.get('n_models'),
        'ModelDispersion': c.get('dispersion'),
        # Emitted only when it was actually measured. When the models disagree
        # too much the engine withholds the margin, and a column that carried
        # the number anyway would let it be sorted on - which is exactly the
        # use the withholding exists to prevent. The raw figure is kept under
        # its own name for the record.
        'MarginOfSafety': m.get('mos') if measured else None,
        'MarginOfSafetyRaw': m.get('mos'),
        'PriceToValue': m.get('price_to_value'),
        'MoSVerdict': m.get('verdict'),
        'MoSBand': m.get('band'),
        'DCF': mv('dcf'),
        'EPV': mv('epv'),
        'EarningsMultipleValue': mv('earnings_multiple'),
        'JustifiedPB': mv('justified_pb'),
        'GrahamNumber': mv('graham_number'),
        'NCAVPerShare': mv('ncav'),
        'GrahamBargainPrice': mv('ncav_two_thirds'),
        'GrowthFormula': mv('growth_formula'),
        'ImpliedGrowth': rev.get('value') if rev.get('applicable') else None,
        # The raw historical rate can be absurd from a small base - NVIDIA's
        # owner earnings compounded at 203% - and reported next to the implied
        # rate it makes any price look cheap. The DCF clips it; the row shows
        # both so the clip is visible rather than silent.
        'HistoricalGrowth': s.get('owner_earnings_cagr'),
        'DCFGrowthUsed': dcf_detail.get('growth_stage1'),
        'EPS': s.get('eps_latest'),
        'EPS3yAvg': s.get('eps_avg3'),
        'EPSGrowth10y': s.get('eps_growth_total'),
        'ROEMedian': s.get('roe_median'),
        'ROICMedian': s.get('roic_median'),
        'GrossMarginMedian': s.get('gross_margin_median'),
        'NetMarginMedian': s.get('net_margin_median'),
        'FCFMarginMedian': s.get('fcf_margin_median'),
        'CurrentRatio': s.get('current_ratio_latest'),
        'DebtToEquity': s.get('debt_to_equity_latest'),
        'ShareCountCAGR': s.get('share_count_cagr'),
        'DividendStreak': (s.get('dividend') or {}).get('streak'),
        'OwnerEarnings': s.get('owner_earnings_latest'),
        'OwnerEarningsDA': s.get('owner_earnings_da_latest'),
        'FiscalYears': result['n_years'],
        'LastFY': result['last_fy'],
        'Caveats': '; '.join(result['caveats']) if result['caveats'] else '',
        'ConfigVersion': result['config_version'],
    }


def evaluate_all(panels: dict, infos: dict, prices: dict, macro: dict,
                 cfg: ValuationConfig, verbose: bool = True) -> tuple:
    """Evaluate every company, returning ``(results, DataFrame of rows)``."""
    results, rows, failures = {}, [], []
    for t, panel in panels.items():
        try:
            r = evaluate(t, panel, infos.get(t, {}), prices.get(t), macro, cfg)
        except Exception as exc:  # noqa: BLE001
            failures.append(f'{t}: {type(exc).__name__}: {exc}')
            continue
        results[t] = r
        rows.append(to_row(r))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['Composite', 'MarginOfSafety'],
                            ascending=False, na_position='last').reset_index(drop=True)
    if verbose:
        print(f'  evaluated {len(results)} companies')
        if failures:
            print(f'  {len(failures)} evaluation failure(s): {"; ".join(failures[:4])}')
        if not df.empty:
            vc = df['Verdict'].value_counts()
            print('  verdicts: ' + ', '.join(f'{k} {v}' for k, v in vc.items()))
            insufficient = int((df['Verdict'] == 'INSUFFICIENT DATA').sum())
            if insufficient:
                print(f'  {insufficient} scored INSUFFICIENT DATA rather than a '
                      'number (too few tests measurable)')
    return results, df
