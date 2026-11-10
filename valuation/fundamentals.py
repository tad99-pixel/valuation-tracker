"""Turn an SEC companyfacts document into a tidy annual fundamental panel.

Three things about XBRL will produce badly wrong numbers if taken at face
value, and all three were confirmed against Apple's filings before this module
was written.

**1. The same period is filed many times, with different values.** Apple's
FY2012 diluted EPS appears as 44.15 (as originally filed) and as 6.31
(restated after the 2014 seven-for-one split). Of 68 annual EPS periods, 58
appear more than once and 14 carry a materially different value. Taking the
first, the last-in-document, or the maximum all give nonsense - a naive reader
would compute Apple's ten-year EPS growth as a catastrophic decline. Every
series here is therefore deduplicated on the period and the **most recently
filed** value kept, which is the split- and restatement-adjusted figure, and
the only one consistent with today's share count and today's price.

**2. `form == "10-K"` and `fp == "FY"` does not mean an annual period.**
Apple's 10-K rows span 90 to 370 days: quarterly facts are filed inside the
annual report under the same labels. Annual rows are selected on the measured
**period length**, not the form or the fiscal-period label.

**3. The `fy` field is the filing's fiscal year, not the period's.** The year
ending 2025-09-27 appears under `fy` 2025 and again under `fy` 2026. Periods
are keyed on their own end date.

What this module does not do: restate anything itself, or fill a gap by
interpolation. A missing year stays missing, so a test that needs ten years
can tell the difference between a company with nine good years and a company
with nine good years and one bad one.

**Point in time.** `build_panel(..., as_of=day)` builds the panel a reader
could have built at the close of that day: only facts filed on or before it
(`facts_as_of`), and that day, not today, as the clock every staleness rule
reads. The backtest uses it; a live run passes no date and is unchanged.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG_VERSION, FUND_DIR, SIC_GROUPS, ValuationConfig

warnings.filterwarnings('ignore')

# An annual period, in days. 52/53-week fiscal years and the odd transition
# period mean this cannot be an equality test.
ANNUAL_MIN, ANNUAL_MAX = 330, 400
ANNUAL_FORMS = ('10-K', '10-K/A', '20-F', '20-F/A', '40-F', '10-KT')

# Concept -> ordered list of XBRL tags. First tag with data for a period wins,
# so the chain runs most-specific first. The variation is not incidental: a
# company reporting under the post-2018 revenue standard uses
# RevenueFromContractWithCustomer*, an older filing uses SalesRevenueNet, and
# a screen that only knows one of them silently reads zero revenue.
DURATION_CONCEPTS = {
    # Read as the largest figure filed for the year (see LARGEST_OF), so the
    # parts below the totals only matter where no total was filed.
    'revenue': [
        'RevenueFromContractWithCustomerExcludingAssessedTax',
        'RevenueFromContractWithCustomerIncludingAssessedTax',
        'Revenues',
        'SalesRevenueNet',
        'SalesRevenueGoodsNet',
        'RevenuesNetOfInterestExpense',          # banks
        'OperatingLeaseLeaseIncome',             # lessors: REITs, towers, rental fleets
        'LeaseIncome',
        'PremiumsEarnedNet',                     # insurers
    ],
    # A bank's revenue is mostly interest, which many banks tag only as these
    # two lines (see `_derive_columns`).
    'net_interest_income': ['InterestIncomeExpenseNet'],
    'noninterest_income': ['NoninterestIncome'],
    'cost_of_revenue': [
        'CostOfRevenue', 'CostOfGoodsAndServicesSold', 'CostOfGoodsSold',
        'CostOfServices',
    ],
    'gross_profit': ['GrossProfit'],
    'operating_income': ['OperatingIncomeLoss'],
    # Not ...BeforeIncomeTaxesDomestic: that is the domestic slice only, and
    # for a multinational it understated EBIT and distorted the tax rate ROIC
    # is struck at.
    'pretax_income': [
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest',
        'IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments',
    ],
    'income_tax': ['IncomeTaxExpenseBenefit'],
    'net_income': [
        'NetIncomeLoss',
        'NetIncomeLossAvailableToCommonStockholdersBasic',
        'ProfitLoss',
    ],
    'eps_diluted': ['EarningsPerShareDiluted', 'EarningsPerShareBasicAndDiluted'],
    'eps_basic': ['EarningsPerShareBasic'],
    'shares_diluted': [
        'WeightedAverageNumberOfDilutedSharesOutstanding',
        'WeightedAverageNumberOfSharesOutstandingDiluted',
    ],
    'shares_basic': [
        'WeightedAverageNumberOfSharesOutstandingBasic',
        'WeightedAverageNumberOfSharesOutstanding',
    ],
    'dividends_per_share': [
        'CommonStockDividendsPerShareDeclared',
        'CommonStockDividendsPerShareCashPaid',
    ],
    # Cash-flow tags first; the equity-statement ones close gaps for filers
    # that tag dividends only there, which would otherwise read as a year in
    # which no dividend was paid.
    'dividends_paid': [
        'PaymentsOfDividendsCommonStock', 'PaymentsOfDividends',
        'PaymentsOfDividendsAndDividendEquivalentsOnCommonStockAndRestrictedStockUnits',
        'PaymentsOfOrdinaryDividends',            # Colgate, 2007-16
        'DividendsCommonStockCash', 'DividendsCommonStock',
    ],
    'cfo': [
        'NetCashProvidedByUsedInOperatingActivities',
        'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations',
    ],
    # Verizon moved its capital expenditure line to ...OtherProductiveAssets
    # in 2019; without it, seven years of capex read as missing and the DCF
    # had nothing to discount. Last in the chain, so it is only used where no
    # PP&E-specific tag was filed for the period.
    'capex': [
        'PaymentsToAcquirePropertyPlantAndEquipment',
        'PaymentsToAcquireProductiveAssets',
        'PaymentsForCapitalImprovements',
        'PaymentsToAcquirePropertyPlantAndEquipmentExcludingInterestCapitalized',
        'PaymentsToAcquireOtherProductiveAssets',
    ],
    # D&A, like debt, is assembled from parts - see `_derive_columns`. These
    # are the *total* tags; `Depreciation` alone is not one of them. AbbVie
    # files `Depreciation` ($0.76bn) and amortisation of intangibles ($7.38bn)
    # separately, and reading the first as the whole understated its owner
    # earnings by roughly two-thirds.
    'depreciation_amortisation': [
        'DepreciationDepletionAndAmortization',
        'DepreciationAmortizationAndAccretionNet',
        'DepreciationAndAmortization',
    ],
    'depreciation_only': ['Depreciation'],
    'amortisation_intangibles': ['AmortizationOfIntangibleAssets'],
    'share_based_comp': ['ShareBasedCompensation', 'AllocatedShareBasedCompensationExpense'],
    'buybacks': ['PaymentsForRepurchaseOfCommonStock', 'PaymentsForRepurchaseOfEquity'],
    # Not InterestIncomeExpenseNet: it is net of interest *income*, positive
    # for a cash-rich company, and adding it to pre-tax income to rebuild
    # EBIT gets the sign backwards.
    'interest_expense': [
        'InterestExpense', 'InterestExpenseDebt',
        'InterestExpenseNonoperating', 'InterestAndDebtExpense',
    ],
}

INSTANT_CONCEPTS = {
    'assets': ['Assets'],
    'assets_current': ['AssetsCurrent'],
    'liabilities': ['Liabilities'],
    'liabilities_current': ['LiabilitiesCurrent'],
    'equity': [
        'StockholdersEquity',
        'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
    ],
    # --- debt, by component -------------------------------------------------
    # A single fallback chain cannot be right here, because the tags are not
    # alternatives - they are different slices of the same liability, and
    # which slices a filer uses varies by industry:
    #
    #   Exxon      LongTermDebtAndCapitalLeaseObligations (noncurrent) + DebtCurrent
    #   Prologis   LongTermDebt - which by taxonomy definition INCLUDES current
    #              maturities, so adding LongTermDebtCurrent on top double-counts
    #   Realty Inc NotesPayable + LoansPayable, and nothing else; the flat chain
    #              read its $26.7bn of debt as zero
    #   Coca-Cola  noncurrent + LongTermDebtAndCapitalLeaseObligationsCurrent
    #              + short-term notes
    #
    # Each slice is collected separately and `_compose_debt` assembles them by
    # rule. Look-alike tags that are NOT debt are deliberately absent:
    # DebtSecurities is an investment (an asset), LineOfCreditFacility*
    # is unused capacity, and NotesAndLoansReceivable is money owed *to* the
    # company.
    'lt_debt_noncurrent': [
        'LongTermDebtNoncurrent', 'LongTermDebtAndCapitalLeaseObligations',
        'LongTermNotesPayable',
    ],
    'lt_debt_incl_current': [
        'LongTermDebt', 'DebtInstrumentCarryingAmount',
    ],
    # Already long-term plus short-term: used as the total, with nothing
    # added. It sat in the chain above, where short-term borrowings were added
    # on top of it and counted twice.
    'debt_total_combined': ['DebtLongtermAndShorttermCombinedAmount'],
    'notes_payable': ['NotesPayable', 'SeniorNotes', 'UnsecuredDebt'],
    'loans_payable': ['LoansPayable'],
    'debt_current': ['DebtCurrent'],
    'debt_current_maturities': [
        'LongTermDebtCurrent', 'LongTermDebtAndCapitalLeaseObligationsCurrent',
    ],
    'st_borrowings': [
        'ShortTermBorrowings', 'CommercialPaper', 'NotesAndLoansPayableCurrent',
        'ShortTermBankLoansAndNotesPayable', 'OtherShortTermBorrowings',
    ],
    'cash': [
        'CashAndCashEquivalentsAtCarryingValue',
        'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents',
        'CashAndDueFromBanks',
    ],
    'short_term_investments': [
        'ShortTermInvestments', 'MarketableSecuritiesCurrent',
        'AvailableForSaleSecuritiesDebtSecuritiesCurrent',
    ],
    'inventory': ['InventoryNet'],
    'goodwill': ['Goodwill'],
    'intangibles': [
        'IntangibleAssetsNetExcludingGoodwill', 'FiniteLivedIntangibleAssetsNet',
    ],
    # After ASC 842 several filers (Kroger from 2021) report PP&E only
    # combined with finance-lease assets.
    'ppe_net': [
        'PropertyPlantAndEquipmentNet',
        'PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization',
    ],
    'retained_earnings': ['RetainedEarningsAccumulatedDeficit'],
    # The claim ahead of the common, at liquidation value where filed - the
    # amount Graham deducts. `PreferredStockValue` is last because it can be
    # par value alone: American Express files $0 there for $1.6bn of
    # preferred carried in paid-in capital. Bank of America files only the
    # carrying value including paid-in capital.
    'preferred_equity': [
        'PreferredStockLiquidationPreferenceValue',
        'PreferredStockIncludingAdditionalPaidInCapitalNetOfDiscount',
        'PreferredStockIncludingAdditionalPaidInCapital',
        'PreferredStockValue',
    ],
    # For a filer that tags no total: shares outstanding times the price each
    # is redeemed at. American Express files its $1.6bn as 1,600 shares at
    # $1,000,000, with a PreferredStockValue of zero (par).
    'preferred_shares': ['PreferredStockSharesOutstanding'],
    'preferred_price_per_share': ['PreferredStockLiquidationPreference',
                                  'PreferredStockRedemptionPricePerShare'],
    'minority_interest': ['MinorityInterest'],
    # `CommonStockSharesIssued` is deliberately NOT a fallback here. Issued
    # shares include treasury stock: Exxon files 8.02bn issued against 4.73bn
    # outstanding, so blending the two tags into one series produced a
    # 0.59x "share count collapse" in 2009 that never happened.
    'shares_outstanding': ['CommonStockSharesOutstanding'],
}

# Concepts stated per share, which a split rescales downward, and share
# counts, which a split rescales upward. Everything else - revenue, net
# income, equity, cash flow - is denominated in dollars and is split-immune.
PER_SHARE_CONCEPTS = ('eps_diluted', 'eps_basic', 'dividends_per_share')
SHARE_COUNT_CONCEPTS = ('shares_diluted', 'shares_basic', 'shares_outstanding')
# Every panel column on a per-share basis, filed or derived. `score` blanks
# these before an uncorrectable basis break; dollar columns keep their history.
PER_SHARE_COLUMNS = PER_SHARE_CONCEPTS + (
    'eps', 'bvps', 'tangible_bvps', 'ncav_per_share', 'fcf_per_share',
    'revenue_per_share', 'shares_used', 'shares_year_end', 'payout_ratio')


# ---------------------------------------------------------------------------
# Fact extraction
# ---------------------------------------------------------------------------

def facts_as_of(facts: dict, as_of) -> dict:
    """The companyfacts document as it stood at the close of `as_of`.

    Every fact filed on or before that day and none filed after it, so the
    restatements, new fiscal years and cover-page counts that later filings
    added are gone, and latest-filed-wins (trap 1) picks among what had been
    filed by then. Apple's FY2012 EPS is 44.15 in a document cut at the end
    of 2013 and 6.31 in one cut at the end of 2014, after the restatement. A
    fact with no filing date is dropped: it cannot be shown to have been
    public that day. Returns a new document; `facts` is not modified.
    """
    cutoff = pd.Timestamp(as_of).strftime('%Y-%m-%d')
    out = {k: v for k, v in facts.items() if k != 'facts'}
    out['facts'] = {}
    for taxonomy, concepts in (facts.get('facts') or {}).items():
        kept = {}
        for name, node in concepts.items():
            units = {}
            for unit, rows in (node.get('units') or {}).items():
                # ISO dates compare as strings; the slice drops any time part.
                rows = [r for r in rows if r.get('filed') and str(r['filed'])[:10] <= cutoff]
                if rows:
                    units[unit] = rows
            if units:
                kept[name] = dict(node, units=units)
        out['facts'][taxonomy] = kept
    return out


def _clock(as_of=None) -> pd.Timestamp:
    """The day every staleness rule measures from: `as_of` in a backtest,
    otherwise today."""
    return (pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.today()).normalize()


def _units_for(concept: str) -> tuple:
    if concept in PER_SHARE_CONCEPTS or concept == 'preferred_price_per_share':
        return ('USD/shares',)
    if concept in SHARE_COUNT_CONCEPTS or concept == 'preferred_shares':
        return ('shares',)
    return ('USD',)


def _pick_latest(rows: list, key_fn) -> dict:
    """Deduplicate facts on their period, keeping the most recently filed.

    This is trap 1. `filed` is an ISO date string, so a string comparison is
    already chronological; `accn` breaks the rare tie where two accessions
    were filed the same day, and sorts to the later accession number.
    """
    best: dict = {}
    for r in rows:
        k = key_fn(r)
        if k is None:
            continue
        prev = best.get(k)
        if prev is None:
            best[k] = r
            continue
        cand = (str(r.get('filed', '')), str(r.get('accn', '')))
        held = (str(prev.get('filed', '')), str(prev.get('accn', '')))
        if cand > held:
            best[k] = r
    return best


# Concepts where the first figure in the chain gives way to another filed for
# the same year that is at least `COMPONENT_RATIO` times larger - the sign the
# first is only a component. ASC 606 "revenue from contracts with customers"
# leaves out lease income, insurance premiums and interest, so for a lessor,
# an insurer or a bank it is a fraction of revenue: American Tower's $0.8bn of
# $10.1bn, Charter's $0.9bn of $55bn (an operating margin of 1,452%), Humana's
# $5.8bn of $130bn. Simply taking the largest figure went wrong the other way:
# Hasbro and Republic Services tag a total before eliminations as Revenues,
# 15% above the revenue they report. Revenue including excise taxes never
# displaces revenue excluding them (Brown-Forman: $5.3bn gross, $4.2bn net).
LARGEST_OF = {'revenue'}
COMPONENT_RATIO = 2.0
_NET_OF_TAX = {'RevenueFromContractWithCustomerIncludingAssessedTax':
               'RevenueFromContractWithCustomerExcludingAssessedTax'}


def _annual_duration_series(facts: dict, tags: list, concept: str):
    """Annual values for a duration concept, indexed by period end date.

    Returns ``(values, tags_used, filed_dates)``. The filing dates are not
    diagnostic trivia - they are what the split adjustment keys on, because a
    per-share figure is stated on the share basis that existed when it was
    filed.
    """
    gaap = facts.get('facts', {}).get('us-gaap', {})
    units_wanted = _units_for(concept)
    collected: dict = {}
    largest: dict = {}          # LARGEST_OF concepts: the biggest figure per period
    tags_by_end: dict = {}

    for tag in tags:
        node = gaap.get(tag)
        if not node:
            continue
        for unit in units_wanted:
            rows = node.get('units', {}).get(unit)
            if not rows:
                continue

            annual = []
            for r in rows:
                s, e = r.get('start'), r.get('end')
                if not s or not e:
                    continue
                try:
                    days = (dt.date.fromisoformat(e) - dt.date.fromisoformat(s)).days
                except ValueError:
                    continue
                # Trap 2: select on measured period length, not form/fp labels.
                if not (ANNUAL_MIN <= days <= ANNUAL_MAX):
                    continue
                annual.append(r)

            if not annual:
                continue
            # Prefer facts that came from an annual report when the same period
            # is available from several form types.
            forms = {r.get('form') for r in annual}
            if forms & set(ANNUAL_FORMS):
                annual = [r for r in annual if r.get('form') in ANNUAL_FORMS]

            # Keyed on the end date alone. Keying on (start, end) let a year
            # re-tagged with a start one day different survive twice, and the
            # first one inserted - not the latest filed - won: 120 such picks
            # across the watchlist, MMM and CAT among them.
            best = _pick_latest(annual, lambda r: r['end'])
            for e, r in best.items():
                # Trap 3: key on the period's own end date.
                if r.get('val') is None:
                    continue
                val = float(r['val'])
                if e not in collected:
                    collected[e] = (val, tag, r.get('filed'))
                if concept in LARGEST_OF:
                    seen = tags_by_end.setdefault(e, set())
                    seen.add(tag)
                    if _NET_OF_TAX.get(tag) in seen:
                        continue
                    if e not in largest or val > largest[e][0]:
                        largest[e] = (val, tag, r.get('filed'))

    for e, big in largest.items():
        first = collected.get(e)
        if first is not None and first[0] > 0 and big[0] >= COMPONENT_RATIO * first[0]:
            collected[e] = big

    if not collected:
        return None
    collected = _merge_near_duplicate_ends(collected)
    idx = pd.to_datetime(sorted(collected))
    keys = [d.strftime('%Y-%m-%d') for d in idx]
    vals = [collected[k][0] for k in keys]
    tags_used = [collected[k][1] for k in keys]
    filed = pd.Series(pd.to_datetime([collected[k][2] for k in keys]), index=idx)
    return pd.Series(vals, index=idx, name=concept), tags_used, filed


def _merge_near_duplicate_ends(collected: dict, days: int = 14) -> dict:
    """One entry per fiscal year when a later filing moves the year-end a day.

    Deere's FY2015 ends 2015-10-31 in one 10-K and 2015-11-01 in the next;
    keyed on the exact date, that was two fiscal years, and Deere's panel had
    21 rows for 19 years - a fake 21-year dividend record, a "ten-year" growth
    test spanning nine, ROE averaged against a duplicate. Ends closer than two
    weeks are the same year; the most recently filed fact keeps it.
    """
    out: dict = {}
    for k in sorted(collected):
        if out:
            prev = max(out)
            if (pd.Timestamp(k) - pd.Timestamp(prev)).days <= days:
                if str(collected[k][2] or '') >= str(out[prev][2] or ''):
                    del out[prev]
                    out[k] = collected[k]
                continue
        out[k] = collected[k]
    return out


def _instant_series(facts: dict, tags: list, concept: str):
    """Balance-sheet values for an instant concept, indexed by the date."""
    gaap = facts.get('facts', {}).get('us-gaap', {})
    units_wanted = _units_for(concept)
    collected: dict = {}

    for tag in tags:
        node = gaap.get(tag)
        if not node:
            continue
        for unit in units_wanted:
            rows = node.get('units', {}).get(unit)
            if not rows:
                continue
            inst = [r for r in rows if r.get('end') and not r.get('start')]
            if not inst:
                # Some filers tag instants with a start date too; an instant is
                # then a zero- or one-day span.
                inst = []
                for r in rows:
                    s, e = r.get('start'), r.get('end')
                    if not e:
                        continue
                    if not s:
                        inst.append(r)
                        continue
                    try:
                        if (dt.date.fromisoformat(e) - dt.date.fromisoformat(s)).days <= 1:
                            inst.append(r)
                    except ValueError:
                        continue
            if not inst:
                continue
            best = _pick_latest(inst, lambda r: r['end'])
            for e, r in best.items():
                if e not in collected and r.get('val') is not None:
                    collected[e] = (float(r['val']), tag, r.get('filed'))

    if not collected:
        return None
    idx = pd.to_datetime(sorted(collected))
    keys = [d.strftime('%Y-%m-%d') for d in idx]
    vals = [collected[k][0] for k in keys]
    filed = pd.Series(pd.to_datetime([collected[k][2] for k in keys]), index=idx)
    return (pd.Series(vals, index=idx, name=concept),
            [collected[k][1] for k in keys], filed)


def _dei_shares_outstanding(facts: dict):
    """Cover-page share count of the latest filing that carries one.

    Returns ``(value, as_of, filed)``. The filing date matters as much as the
    value: companies that adopt class-dimensioned tagging stop filing the plain
    cover fact, so the "latest" plain one can be years old and on a pre-split
    basis. Mastercard's is from before its 2014 ten-for-one split - 122.5m
    shares against 906m today - and dividing a DCF by it overstated value per
    share 7.4x. `_vet_current_shares` decides whether it can be used.
    """
    dei = facts.get('facts', {}).get('dei', {})
    node = dei.get('EntityCommonStockSharesOutstanding')
    if not node:
        return None, None, None
    rows = node.get('units', {}).get('shares') or []
    if not rows:
        return None, None, None
    best = max(rows, key=lambda r: (str(r.get('filed', '')), str(r.get('end', ''))))
    try:
        return (float(best['val']), pd.Timestamp(best['end']),
                pd.Timestamp(best['filed']) if best.get('filed') else None)
    except Exception:
        return None, None, None


# Ratios a stock split can take. An unadjusted split shows up as a year-on-year
# share count change of almost exactly one of these.
SPLIT_RATIOS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 50, 100)


def _repair_unit_scale(s: pd.Series, implied: pd.Series | None = None) -> tuple:
    """Undo figures filed in thousands or millions too many or too few - share
    counts under a `shares` unit, and the D&A total.

    A real share count does not change a thousandfold in a year, so a value
    that sits ~1,000x or ~1,000,000x away from the median of its neighbours is
    a scale error in the filing, not an event. Seen in practice: Hershey 2009
    (228 thousand, i.e. thousands - which turned its computed EPS into
    $1,903.95), NVIDIA FY2008-09 (thousands), Berkshire 2008-09 (1.55
    *trillion*, i.e. millions too many).

    Only exact orders of magnitude are corrected, within a factor of 3.3, and
    only when two references agree. There are three: the median of the whole
    history, the median of the surrounding +/-3 years, and - when `implied`
    is given - the same year's figure derived another way, within a factor
    of 1.5: net income / EPS for a share count, depreciation plus
    amortisation for a D&A total (Coherent tagged its FY2023 total as
    $681.7bn, a thousand times its $267.6m + $414.1m of parts). Any
    one alone misfires. A local window can be outvoted - Berkshire's two bad
    years, 2008 and 2009, made its correct 2007 figure look a millionfold too
    small, and an earlier version "repaired" it; and a run of bad years
    outvotes it for every year but the first - McDonald's filed 2021-25 in
    millions, and only 2021 was repaired until net income / EPS had a vote.
    That cannot decide alone either, since net income can be mis-scaled too -
    unless its whole series runs smoothly where the filed one jumps, which
    says which of the two is wrong (see below). A
    global median can sit far from both ends of a company that grew its share
    count several-fold. Anything that is not an exact order of magnitude by
    two measures is left alone: a genuine 2x or 9x change is a corporate
    event, and "repairing" it would be inventing data.
    """
    s = s.copy()
    vals = s.dropna()
    vals = vals[vals > 0]
    fixed = []
    if len(vals) < 3:
        return s, fixed
    v = vals.to_numpy(dtype=float).copy()
    d = pd.DatetimeIndex(vals.index)
    own = (implied.reindex(vals.index).to_numpy(dtype=float) if implied is not None
           else np.full(len(v), np.nan))

    def scale_error(r, tol=3.3):
        for scale in (1e3, 1e6):
            if scale / tol <= r <= scale * tol:
                return 1.0 / scale, f'/{scale:,.0f}'
            if 1 / (scale * tol) <= r <= tol / scale:
                return scale, f'x{scale:,.0f}'
        return None, None

    def jumps(x):
        return any(scale_error(r)[0] is not None for r in x[1:] / x[:-1])

    # When the figures derived another way run smoothly across the years and
    # the filed ones jump a thousandfold, the derived series settles it: the
    # filed years that disagree with it by an exact order of magnitude are the
    # wrong ones. A run of bad years can carry both medians - ConocoPhillips
    # filed 2010-19 in thousands, ten of its nineteen years, and the vote below
    # left eight of them and "repaired" its correct 2009 instead. Were net
    # income the mis-scaled figure, the derived series would be the one jumping.
    has_own = np.isfinite(own) & (own > 0)
    if has_own.sum() >= 3 and jumps(v) and not jumps(own[has_own]):
        for i in np.flatnonzero(has_own):
            f, label = scale_error(v[i] / own[i], tol=1.5)
            if f is not None:
                v[i] *= f
                s[vals.index[i]] = v[i]
                fixed.append(f'{d[i].year}: {label}')
    global_ref = float(np.median(v))

    # Each value's neighbours: the others within three calendar years, or all
    # the others when fewer than two are that close. One boolean matrix rather
    # than a filtered Series per value.
    lo = (d - pd.DateOffset(years=3)).to_numpy()
    hi = (d + pd.DateOffset(years=3)).to_numpy()
    at = d.to_numpy()
    others = ~np.eye(len(v), dtype=bool)
    near = (at[None, :] >= lo[:, None]) & (at[None, :] <= hi[:, None]) & others
    for i in range(len(v)):
        mask = near[i] if near[i].sum() >= 2 else others[i]
        local_ref = float(np.median(v[mask]))
        votes = [scale_error(v[i] / global_ref), scale_error(v[i] / local_ref)]
        if np.isfinite(own[i]) and own[i] > 0:
            votes.append(scale_error(v[i] / own[i], tol=1.5))
        factors = [f for f, _ in votes if f is not None]
        agreed = next((f for f in factors if factors.count(f) >= 2), None)
        if agreed is not None:
            s[vals.index[i]] = v[i] * agreed
            fixed.append(f'{d[i].year}: ' + next(lab for f, lab in votes if f == agreed))
    return s, sorted(fixed)


# ---------------------------------------------------------------------------
# Panel assembly
# ---------------------------------------------------------------------------

def _classify_sic(sic) -> str:
    """Map the filer's own SIC code to a tests-carve-out group."""
    try:
        code = int(sic)
    except (TypeError, ValueError):
        return 'operating'
    for group in ('bank', 'insurance', 'reit', 'utility'):
        for lo, hi in SIC_GROUPS[group]:
            if lo <= code <= hi:
                return group
    for lo, hi in SIC_GROUPS['financial']:
        if lo <= code <= hi:
            return 'financial'
    return 'operating'


def _split_factors(filed: pd.Series, splits: pd.Series) -> pd.Series:
    """Cumulative split ratio applied *after* each fact was filed.

    A value filed before a split is quoted on the pre-split basis, so dividing
    a per-share figure by this factor (or multiplying a share count by it)
    restates the whole history onto today's basis. Keying on the filing date
    rather than the period end is the part that matters: EDGAR restates the two
    or three comparative years inside a later annual report, so the same fiscal
    year can be correct on one basis in one filing and another basis in
    another.
    """
    if splits is None or len(splits) == 0:
        return pd.Series(1.0, index=filed.index)
    s = splits.sort_index()
    # suffix[i] is the product of every split from the i-th on, so the factor
    # for a fact is suffix[first split dated after its filing].
    suffix = np.append(np.cumprod(s.to_numpy(dtype=float)[::-1])[::-1], 1.0)
    when = pd.to_datetime(pd.Series(filed)).to_numpy().astype('datetime64[ns]')
    at = pd.DatetimeIndex(s.index).to_numpy().astype('datetime64[ns]')
    out = suffix[np.searchsorted(at, when, side='right')]
    # Unknown filing date: assume the oldest basis, which is the conservative
    # choice - it never leaves a pre-split figure un-adjusted and visible as
    # spurious growth.
    out[pd.isna(when)] = suffix[0]
    return pd.Series(out, index=filed.index)


def build_panel(ticker: str, facts: dict, meta: dict,
                cfg: ValuationConfig, splits: pd.Series | None = None,
                as_of=None) -> tuple:
    """Return ``(annual DataFrame, info dict)`` for one company.

    The frame is indexed by fiscal-period end date, one row per fiscal year,
    with NaN wherever the company did not file that concept. `info` carries
    everything that is not a time series: the SIC group, the current share
    count, which XBRL tags were actually used, and how complete the panel is.

    Per-share series are restated onto today's share basis using `splits`.
    Pass `splits=None` only when the company has never split; otherwise the
    per-share history spans several bases and every per-share test built on it
    is meaningless.

    `as_of` (a date) builds the panel as it could have been built at the
    close of that day: facts filed after it are dropped first
    (`facts_as_of`), and the history window, the filing's age and the
    current share count are judged from that day. Left None - every live
    run - nothing is dropped and the day is today. Splits after `as_of`
    still restate the per-share figures, onto the same basis as today's
    split-adjusted price history: a constant factor on every per-share
    figure and on the price alike, which no ratio sees.
    """
    if as_of is not None:
        facts = facts_as_of(facts, as_of)
    duration, tags_used, filed_dates = {}, {}, {}
    for concept, tags in DURATION_CONCEPTS.items():
        got = _annual_duration_series(facts, tags, concept)
        if got is not None:
            duration[concept], used, filed_dates[concept] = got
            tags_used[concept] = sorted(set(used))

    instant = {}
    for concept, tags in INSTANT_CONCEPTS.items():
        got = _instant_series(facts, tags, concept)
        if got is not None:
            instant[concept], used, filed_dates[concept] = got
            tags_used[concept] = sorted(set(used))

    # Restate per-share facts and share counts onto the current basis before
    # anything is aligned or derived from them.
    split_report = {'applied': False, 'n_adjusted': 0, 'max_factor': 1.0}
    if splits is not None and len(splits):
        for concept in PER_SHARE_CONCEPTS + SHARE_COUNT_CONCEPTS:
            src = duration if concept in duration else (instant if concept in instant else None)
            if src is None or concept not in filed_dates:
                continue
            f = _split_factors(filed_dates[concept], splits)
            f = f.reindex(src[concept].index).fillna(1.0)
            if concept in PER_SHARE_CONCEPTS:
                src[concept] = src[concept] / f
            else:
                src[concept] = src[concept] * f
            n = int((f != 1.0).sum())
            split_report['n_adjusted'] += n
            split_report['max_factor'] = max(split_report['max_factor'], float(f.max()))
        split_report['applied'] = True
        split_report['n_splits'] = int(len(splits))

    if 'net_income' not in duration and 'equity' not in instant:
        return None, {'ticker': ticker, 'error': 'no usable XBRL facts'}

    # The fiscal calendar comes from whichever annual series is longest, so a
    # company that stopped reporting one concept does not lose its history.
    anchor = None
    for key in ('revenue', 'net_income', 'cfo', 'eps_diluted'):
        if key in duration and (anchor is None or len(duration[key]) > len(anchor)):
            anchor = duration[key]
    if anchor is None:
        anchor = max(duration.values(), key=len) if duration else None
    if anchor is None:
        return None, {'ticker': ticker, 'error': 'no annual duration facts'}

    fy_ends = pd.DatetimeIndex(sorted(anchor.index))
    panel = pd.DataFrame(index=fy_ends)
    panel.index.name = 'FYEnd'

    # Nearest within two weeks rather than exact: each concept's year-end is
    # the date its latest filing used, and for the same fiscal year that can
    # differ by a day between concepts.
    for concept, s in duration.items():
        panel[concept] = s.reindex(fy_ends, method='nearest',
                                   tolerance=pd.Timedelta(days=14))

    # Balance-sheet instants are matched to the fiscal year end. 52/53-week
    # years move the date by a few days, so the match is nearest-within-
    # tolerance rather than exact - but the tolerance is tight enough that a
    # quarter-end can never be mistaken for a year-end.
    for concept, s in instant.items():
        panel[concept] = s.reindex(fy_ends, method='nearest',
                                   tolerance=pd.Timedelta(days=10))

    panel = panel.sort_index()
    if cfg.history_years:
        cutoff = _clock(as_of) - pd.DateOffset(years=cfg.history_years + 1)
        panel = panel.loc[panel.index >= cutoff]

    panel = _derive_columns(panel)
    unit_fixes = list(panel.attrs.get('share_unit_fixes', []))
    da_fixes = list(panel.attrs.get('da_unit_fixes', []))
    from_identity = int(panel.attrs.get('shares_from_identity', 0))

    cover, cover_date, cover_filed = _dei_shares_outstanding(facts)
    if cover is not None and splits is not None and len(splits):
        f = _split_factors(pd.Series([cover_filed], index=[0]), splits).iloc[0]
        cover = float(cover * f)
    sic = meta.get('sic')
    info = {
        'ticker': ticker,
        'cik': facts.get('cik'),
        'name': meta.get('name') or facts.get('entityName'),
        'sic': sic,
        'sic_description': meta.get('sicDescription'),
        'group': _classify_sic(sic),
        'fiscal_year_end': meta.get('fiscalYearEnd'),
        'exchanges': meta.get('exchanges', []),
        'shares_outstanding_date': cover_date,
        'cover_shares': cover,
        'cover_filed': cover_filed,
        'share_unit_fixes': unit_fixes,
        'da_unit_fixes': da_fixes,
        'shares_from_identity': from_identity,
        'n_years': int(len(panel)),
        'first_fy': panel.index[0] if len(panel) else None,
        'last_fy': panel.index[-1] if len(panel) else None,
        'tags_used': tags_used,
        'splits': split_report,
        # None means the split history could not be fetched and nothing was
        # ever verified - distinct from an empty Series, "never split".
        # `score` withholds every price comparison when it is False.
        'splits_known': splits is not None,
        # For `score`, which asks whether a split-shaped jump in the share
        # count had a split on record near it.
        'split_dates': ([pd.Timestamp(d).strftime('%Y-%m-%d') for d in splits.index]
                        if splits is not None and len(splits) else []),
        'error': None,
    }
    info.update(_share_basis_diagnostic(panel))
    return panel, _dated_info(panel, info, cfg, as_of)


def _dated_info(panel: pd.DataFrame, info: dict, cfg, as_of=None) -> dict:
    """The parts of `info` that depend on today's date (or `as_of`'s).

    Recomputed on every run, including for a panel reused from the cache: a
    cover-page count that was fresh when the panel was built goes stale
    without the filing changing, and so does the filing itself.
    """
    shares_now, why = _vet_current_shares(panel, info.get('cover_shares'),
                                          info.get('cover_filed'), cfg, as_of)
    info['shares_outstanding_now'], info['shares_now_source'] = shares_now, why
    if len(panel):
        age = int((_clock(as_of) - panel.index[-1]).days)
        info['last_fy_age_days'], info['filing_stale'] = age, age > cfg.max_filing_age_days
    else:
        info['last_fy_age_days'], info['filing_stale'] = None, True
    return info


def _vet_current_shares(panel: pd.DataFrame, cover, cover_filed, cfg, as_of=None) -> tuple:
    """The share count a per-share valuation divides by - or None.

    A DCF or EPV produces a value for the whole company; dividing it by the
    wrong share count produces a confident, wrong per-share number, so the
    count is chosen conservatively:

    1. The cover-page count, already restated for splits after its filing
       date - but only if it was filed recently, and only if it agrees with
       the latest fiscal year's weighted count. With no current fiscal-year
       count, it must instead be within 2x of the last filed one; with no
       share history at all it cannot be checked, and is not used. (It used
       to be accepted unchecked in exactly that case - where a count filed in
       thousands, or for one class of several, would go unnoticed.)
    2. Otherwise the latest fiscal year's weighted diluted count, if that
       year is itself recent.
    3. Otherwise None, and every per-share valuation is reported as
       unmeasurable. Visa and Berkshire land here: both tag their share
       classes only with dimensions this source does not carry, and a value
       per share on an unknown share basis is not a value per share.

    "Recently" is measured from `as_of` when one is given (see `build_panel`).
    """
    today = _clock(as_of)
    shares = (panel['shares_used'] if 'shares_used' in panel.columns
              else pd.Series(dtype=float))
    latest = None
    if len(shares) and pd.notna(shares.iloc[-1]) and \
            (today - panel.index[-1]).days <= cfg.max_filing_age_days:
        latest = float(shares.iloc[-1])

    if cover is not None and cover > 0 and cover_filed is not None \
            and (today - pd.Timestamp(cover_filed)).days <= 460:
        if latest is not None:
            if 0.8 <= cover / latest <= 1.25:
                return float(cover), 'cover page of the latest filing'
        else:
            recent = shares.iloc[-3:].dropna()
            ratio = cover / float(recent.iloc[-1]) if len(recent) else None
            if ratio is not None and 0.5 <= ratio <= 2.0:
                return float(cover), ('cover page of the latest filing (checked against '
                                      f'the {recent.index[-1].year} share count)')
            return None, ('no usable share count: the cover-page count could not be '
                          'confirmed (' + ('no recent share count to check its scale against'
                                           if ratio is None else
                                           f'{ratio:.2f}x the last filed count') + ')')
    if latest is not None:
        why = 'latest fiscal-year diluted count'
        if cover is not None:
            why += (' (cover-page count rejected: '
                    + ('stale' if cover_filed is None or (today - pd.Timestamp(cover_filed)).days > 460
                       else f'{cover / latest:.2f}x the fiscal-year count') + ')')
        return latest, why
    return None, ('no usable share count: the cover-page figure is stale or absent and '
                  'the latest fiscal year has none - typical of multi-class filers')


def _share_basis_diagnostic(panel: pd.DataFrame) -> dict:
    """Is the per-share *history* on one basis after split and unit repair?

    The test looks for the signature of a split that was not corrected - a
    year-on-year change in the share count of almost exactly a split ratio -
    rather than for any large change. An earlier, blunter version flagged
    every move beyond 1.5x, which marked Prologis's 2011 stock-for-stock
    merger with AMB as a data error; that was a real 1.69x increase that every
    shareholder lived through, and the per-share history that includes it is
    the correct one. Large moves that are not split-shaped are reported as an
    event, not an error.

    Failing sets `per_share_reliable` False, and `score` then reports every
    test that depends on per-share *history* as unmeasurable.
    """
    out = {'per_share_reliable': True, 'share_basis_note': None, 'share_events': []}
    if 'shares_used' not in panel.columns:
        out.update(per_share_reliable=False, share_basis_note='no share count filed')
        return out
    s = panel['shares_used'].dropna()
    if len(s) == 0:
        out.update(per_share_reliable=False, share_basis_note='no share count filed')
        return out
    if len(s) < 2:
        return out

    # Every pair is scanned and the LAST break recorded: truncation keeps
    # history from the break on, so with two breaks the first-found one left a
    # second basis change inside the kept window - a 4:1 and a later 10:1
    # truncated at the 4:1 still counted the 10:1 as an EPS collapse.
    ratio = (s / s.shift(1)).dropna()
    breaks = []
    for d, x in ratio.items():
        split_like = None
        for k in SPLIT_RATIOS:
            for target in (k, 1.0 / k):
                if abs(x / target - 1) <= 0.04:
                    split_like = target
                    break
            if split_like:
                break
        # 3-for-2 needs a tighter band: mergers land near 1.5x often enough
        # (Realty Income/VEREIT was 1.47x) to make 4% indiscriminate.
        if split_like is None and (abs(x / 1.5 - 1) <= 0.012 or abs(x * 1.5 - 1) <= 0.012):
            split_like = 1.5 if x > 1 else 1 / 1.5
        if split_like is not None:
            breaks.append((d, 'split', split_like, x, (
                f'share count moved {x:.2f}x in {d.year} - the signature of a '
                f'{split_like if split_like > 1 else 1 / split_like:g}-for-1 '
                f'{"split" if split_like > 1 else "reverse split"} that the split '
                'history did not correct; per-share history not comparable')))
        elif x > 20 or x < 0.05:
            # No merger moves a share count twentyfold in a year. This is a
            # scale error the repair could not place with confidence, and it
            # must not be reported as a corporate event.
            breaks.append((d, 'scale', None, x, (
                f'share count moved {x:,.3g}x in {d.year} - a filing scale error '
                'that could not be repaired; per-share history not comparable')))
        elif x > 1.4 or x < 0.7:
            out['share_events'].append(f'{d.year}: share count {x:.2f}x '
                                       '(merger, conversion or large issuance)')
    # Every break, for `score`: a split-shaped one can still turn out to be a
    # merger or a listing once the price record is in hand.
    out['share_breaks'] = [{'date': d.strftime('%Y-%m-%d'), 'kind': kind, 'target': target,
                            'ratio': float(x), 'note': note}
                           for d, kind, target, x, note in breaks]
    if breaks:
        out['per_share_reliable'] = False
        out['share_basis_break'] = breaks[-1][0]       # the current basis starts here
        out['share_basis_note'] = breaks[-1][-1] + (
            f' ({len(breaks)} breaks; history kept from the last)' if len(breaks) > 1 else '')
    return out


def _compose_debt(p: pd.DataFrame) -> pd.DataFrame:
    """Assemble long-term, current and total debt from whichever slices exist.

    Precedence, per fiscal year:

    1. A strictly **noncurrent** figure is filed: long-term debt is that, and
       current debt (``DebtCurrent``, else maturities plus short-term
       borrowings) is added for the total.
    2. Only a figure **including** current maturities (``LongTermDebt``) is
       filed: that already is long-term plus current maturities, so only
       short-term borrowings are added - adding ``LongTermDebtCurrent`` here
       would count the same bonds twice.
    3. Only **notes** (the REIT pattern): notes plus loans plus short-term
       borrowings. ``NotesPayable`` and ``LoansPayable`` are disjoint by
       taxonomy definition, so summing them does not double-count.
    4. Nothing long-term at all: total is current debt, or unknown.

    Unknown stays NaN. A company that tagged none of these has debt nobody
    measured, which is not the same as no debt - and the difference decides
    every leverage test downstream.
    """
    def c(name):
        return p[name] if name in p.columns else pd.Series(np.nan, index=p.index)

    noncurrent = c('lt_debt_noncurrent')
    incl_current = c('lt_debt_incl_current')
    combined = c('debt_total_combined')
    notes = c('notes_payable')
    loans = c('loans_payable')
    debt_cur = c('debt_current')
    maturities = c('debt_current_maturities')
    st_borrow = c('st_borrowings')

    any_cur_part = maturities.notna() | st_borrow.notna()
    current = debt_cur.where(debt_cur.notna(),
                             (maturities.fillna(0) + st_borrow.fillna(0)).where(any_cur_part))

    use_nc = noncurrent.notna()
    use_comb = ~use_nc & combined.notna()
    use_ic = ~use_nc & ~use_comb & incl_current.notna()
    use_notes = ~use_nc & ~use_comb & ~use_ic & notes.notna()
    none_lt = ~use_nc & ~use_comb & ~use_ic & ~use_notes
    # A long-term slice filed in an earlier year and missing now is a tag
    # change, not a repayment. Falling through to "current debt only" read
    # Deere's last four years at its commercial paper alone, tens of billions
    # short. Those years are unknown.
    had_lt = (noncurrent.notna() | combined.notna() | incl_current.notna()
              | notes.notna()).cummax()
    lapsed = none_lt & had_lt

    lt = pd.Series(np.nan, index=p.index)
    total = pd.Series(np.nan, index=p.index)

    lt[use_nc] = noncurrent[use_nc]
    total[use_nc] = noncurrent[use_nc] + current[use_nc].fillna(0)

    total[use_comb] = combined[use_comb]
    lt[use_comb] = (combined - current.fillna(0))[use_comb]

    lt[use_ic] = (incl_current - maturities.fillna(0))[use_ic]
    total[use_ic] = (incl_current + st_borrow.fillna(0))[use_ic]

    lt[use_notes] = (notes + loans.fillna(0))[use_notes]
    total[use_notes] = (notes + loans.fillna(0) + st_borrow.fillna(0))[use_notes]

    current_only = none_lt & ~lapsed
    total[current_only] = current[current_only]

    p['lt_debt'] = lt
    p['total_debt'] = total
    p['st_debt'] = total - lt.fillna(0)
    p['debt_source'] = np.select(
        [use_nc, use_comb, use_ic, use_notes, lapsed, current_only & current.notna()],
        ['noncurrent+current', 'combined total', 'incl-current+borrowings',
         'notes+loans', 'lapsed - unknown', 'current only'], default='unknown')
    return p


def _derive_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Columns that are pure arithmetic on what was filed.

    Nothing here estimates or interpolates. Where an input is missing the
    output is NaN, which the criteria read as "not measurable" rather than as
    zero. The scanner learned this the hard way: a missing measurement that
    prints as a measured zero is worse than no number at all.
    """
    p = panel.copy()

    def col(name):
        return p[name] if name in p.columns else pd.Series(np.nan, index=p.index)

    # A bank's revenue is net interest income plus fees. Many banks tag no
    # revenue total, and their one revenue tag holds the fees alone:
    # Huntington's $2.0bn, beside $5.3bn of net interest income.
    # Held to the same standard as any other revenue figure: it takes over only
    # where the filed one is missing or less than half of it.
    banking = col('net_interest_income') + col('noninterest_income')
    filed = col('revenue')
    p['revenue'] = filed.where(~(banking >= COMPONENT_RATIO * filed) & filed.notna(),
                               banking.where(banking > 0))

    # Gross profit, when only cost of revenue was filed.
    if 'gross_profit' not in p.columns or p['gross_profit'].isna().all():
        p['gross_profit'] = col('revenue') - col('cost_of_revenue')
    else:
        p['gross_profit'] = p['gross_profit'].fillna(col('revenue') - col('cost_of_revenue'))

    # Operating income is simply not filed by some large filers - Exxon tags
    # it in none of its nineteen years, and nor do several utilities. Without
    # a fallback, ROIC is NaN for an entire sector for no better reason than a
    # tagging convention. Pre-tax income plus interest expense is EBIT by
    # definition, so it is reconstructed and the source recorded rather than
    # left blank or quietly conflated with the filed figure.
    ebit = col('pretax_income') + col('interest_expense').fillna(0)
    if 'operating_income' in p.columns:
        p['operating_income_source'] = np.where(
            p['operating_income'].notna(), 'filed',
            np.where(ebit.notna(), 'pretax+interest', 'unavailable'))
        p['operating_income'] = p['operating_income'].fillna(ebit)
    else:
        p['operating_income'] = ebit
        p['operating_income_source'] = np.where(ebit.notna(), 'pretax+interest',
                                                'unavailable')

    p = _compose_debt(p)

    # D&A: a total tag where one was filed, otherwise depreciation plus
    # amortisation of intangibles. Never both - a filer that tags the total
    # usually tags the parts too, and summing all three counts the
    # amortisation twice. Owner earnings add back every non-cash charge
    # Buffett lists, amortisation included.
    dep, amort = col('depreciation_only'), col('amortisation_intangibles')
    da_parts = (dep + amort.fillna(0)).where(dep.notna(), amort)
    # The total can be filed at the wrong scale; its parts and its other
    # years are the references (see `_repair_unit_scale`).
    da_total, da_fixes = _repair_unit_scale(col('depreciation_amortisation'), da_parts)
    p.attrs['da_unit_fixes'] = da_fixes
    p['depreciation_amortisation'] = da_total.fillna(da_parts)
    p['da_source'] = np.where(da_total.notna(), 'total',
                              np.where(da_parts.notna(), 'depreciation+amortisation',
                                       'unavailable'))

    p['working_capital'] = col('assets_current') - col('liabilities_current')
    p['current_ratio'] = col('assets_current') / col('liabilities_current').replace(0, np.nan)

    # Liabilities is frequently not filed as a single tag (only its parts are),
    # so fall back to assets less equity, which is the same identity.
    liab = col('liabilities')
    p['liabilities'] = liab.fillna(col('assets') - col('equity'))

    # Claims ahead of the common. Graham deducts preferred stock from net
    # current assets ("after deducting all prior claims") and from the book
    # value of the common ("deducting all liabilities and stock issues ahead
    # of the common"). Left in, a bank's book value per share carried its
    # preferred: $20bn at JPMorgan, $19bn at Wells Fargo.
    filed_pref = col('preferred_equity')
    implied_pref = col('preferred_shares') * col('preferred_price_per_share')
    # The implied total replaces a filed one only where that is missing or is
    # plainly par value, and never exceeds half of equity - beyond that the
    # per-share price is more likely per depositary share than per share.
    use_implied = (implied_pref.notna() & (implied_pref > 0)
                   & (filed_pref.isna() | (filed_pref < 0.5 * implied_pref))
                   & (implied_pref <= 0.5 * col('equity').abs()))
    preferred = filed_pref.where(~use_implied, implied_pref).fillna(0).clip(lower=0)
    p['preferred_claim'] = preferred
    p['common_equity'] = col('equity') - preferred
    p['ncav'] = col('assets_current') - p['liabilities'] - preferred
    p['tangible_equity'] = (p['common_equity'] - col('goodwill').fillna(0)
                            - col('intangibles').fillna(0))
    p['net_debt'] = p['total_debt'] - col('cash').fillna(0) - col('short_term_investments').fillna(0)

    p['fcf'] = col('cfo') - col('capex')
    p['gross_margin'] = p['gross_profit'] / col('revenue').replace(0, np.nan)
    p['operating_margin'] = col('operating_income') / col('revenue').replace(0, np.nan)
    p['net_margin'] = col('net_income') / col('revenue').replace(0, np.nan)
    p['fcf_margin'] = p['fcf'] / col('revenue').replace(0, np.nan)

    # ROE on average equity, which is the correct denominator for a flow
    # measured over a period against a stock measured at a point. Undefined
    # where that equity is not positive: a business that has bought back more
    # than its book value (Altria, Home Depot, McDonald's at times) would
    # otherwise show a *negative* ROE while earning billions, and fail a
    # quality test on an accounting artefact. ROIC still measures it.
    avg_equity = (col('equity') + col('equity').shift(1)) / 2
    p['roe'] = col('net_income') / avg_equity.where(avg_equity > 0)
    p['roe_ending'] = col('net_income') / col('equity').where(col('equity') > 0)

    invested = col('equity') + p['total_debt'] - col('cash').fillna(0)
    avg_invested = (invested + invested.shift(1)) / 2
    nopat = col('operating_income') * (1 - (col('income_tax') /
                                            col('pretax_income').replace(0, np.nan)).clip(0, 0.5).fillna(0.21))
    p['roic'] = nopat / avg_invested.where(avg_invested > 0)
    p['nopat'] = nopat

    p['roa'] = col('net_income') / ((col('assets') + col('assets').shift(1)) / 2).replace(0, np.nan)
    # Undefined, not negative, when equity is not positive: -8.0 passed the
    # utility leverage test (<= 2.0) for a company whose liabilities exceed
    # its assets. The test itself fails negative equity explicitly.
    p['debt_to_equity'] = p['total_debt'] / col('equity').where(col('equity') > 0)
    p['equity_to_assets'] = col('equity') / col('assets').replace(0, np.nan)

    # --- share count ---
    #
    # The filed count is authoritative once its unit-scale errors are undone.
    #
    # An earlier version trusted `net income / EPS` over the filed count
    # whenever the two disagreed, on the theory that the identity is immune to
    # tagging errors. It is not immune to arithmetic: EPS is struck *after*
    # preferred dividends and rounded to the cent, so for Bank of America in
    # 2011 ($1.45bn of income, $0.01 of EPS) the quotient was 144.6 billion
    # shares, and for Wells Fargo in 2009 - TARP preferred dividends - it was
    # 7.0bn against 4.6bn filed. The identity is now a fallback for years with
    # no filed count at all, and only where it lands near the neighbouring
    # years. It also has a vote when a filed count's scale is in doubt.
    #
    # A count of zero is no count: it marks the years before there were shares
    # - Constellation Energy's carve-out years before its 2022 spin-off, Ares
    # before it became a corporation. Kept, it made the next year a jump of
    # "infx" and cut the per-share history there.
    def count(name):
        c = col(name)
        return c.where(c > 0)

    shares_raw = count('shares_diluted').fillna(count('shares_basic')).fillna(
        count('shares_outstanding'))
    eps_filed = col('eps_diluted').fillna(col('eps_basic'))
    shares_implied = col('net_income') / eps_filed.replace(0, np.nan)
    shares_implied = shares_implied.where(shares_implied > 0)
    shares_filed, unit_fixes = _repair_unit_scale(shares_raw, shares_implied)

    shares = shares_filed.copy()
    ref = shares_filed.rolling(7, center=True, min_periods=2).median()
    fill_ok = (shares_filed.isna() & shares_implied.notna()
               & (ref.isna() | (shares_implied / ref).between(0.67, 1.5)))
    shares[fill_ok] = shares_implied[fill_ok]

    p['shares_used'] = shares
    p['shares_filed'] = shares_filed
    p['shares_implied'] = shares_implied
    p.attrs['share_unit_fixes'] = unit_fixes
    p.attrs['shares_from_identity'] = int(fill_ok.sum())

    # Book value and net current assets are year-end balances, so they are
    # divided by the shares outstanding at the year end, not by the year's
    # weighted average - which lags a buyback programme and overstates the
    # count it divides by. The weighted count stands in where no year-end
    # count is filed, or where the two disagree by more than a quarter (a
    # count filed for one class, or at the wrong scale).
    outstanding, _ = _repair_unit_scale(count('shares_outstanding'), shares_implied)
    agree = (outstanding / shares.replace(0, np.nan)).between(0.8, 1.25)
    year_end = outstanding.where(agree, shares)
    p['shares_year_end'] = year_end
    p['bvps'] = p['common_equity'] / year_end.replace(0, np.nan)
    p['tangible_bvps'] = p['tangible_equity'] / year_end.replace(0, np.nan)
    p['ncav_per_share'] = p['ncav'] / year_end.replace(0, np.nan)
    p['fcf_per_share'] = p['fcf'] / shares.replace(0, np.nan)
    p['revenue_per_share'] = col('revenue') / shares.replace(0, np.nan)

    # EPS: prefer what the company reported. Computing it from net income and a
    # share count disagrees with the filed figure often enough to matter, and
    # the filed figure is the one Graham's tests refer to - except a filed zero
    # in a year with no shares, which is the same placeholder as the count
    # (Constellation Energy 2020-21: $589m of income, "EPS 0.00").
    eps_filed = eps_filed.mask((eps_filed == 0) & shares.isna())
    computed = col('net_income') / shares.replace(0, np.nan)
    p['eps'] = eps_filed.fillna(computed)
    p['eps_source'] = np.where(eps_filed.notna(), 'filed', 'computed')

    p['payout_ratio'] = col('dividends_per_share') / p['eps'].replace(0, np.nan)
    p['pays_dividend'] = (col('dividends_per_share').fillna(0) > 0) | (col('dividends_paid').fillna(0) > 0)
    return p


# ---------------------------------------------------------------------------
# Panel cache
# ---------------------------------------------------------------------------
#
# Reading and parsing a companyfacts document is most of the cost of a run
# once prices are cached - a large filer's is ~20 MB of JSON - and between
# two filings it produces the same panel every time. A built panel is kept
# with a key over everything it depends on: the document's hash, the split
# history, the filer's metadata, the config, and the source of this module
# and of config.py, so an edit to either invalidates the cache without
# anyone having to remember to.

_DATE_KEYS = ('shares_outstanding_date', 'cover_filed', 'first_fy', 'last_fy',
              'share_basis_break')
_code_hash_memo: list = []


def _code_hash() -> str:
    if not _code_hash_memo:
        h = hashlib.sha1()
        for f in (Path(__file__), Path(__file__).with_name('config.py')):
            h.update(f.read_bytes())
        _code_hash_memo.append(h.hexdigest())
    return _code_hash_memo[0]


def entry_key(facts_sha1, splits, meta: dict, cfg: ValuationConfig):
    """The cache key for one company's panel, or None when the document has
    no hash (it could not be written to disk), in which case nothing is cached."""
    if not facts_sha1:
        return None
    split_sig = ('unknown' if splits is None else
                 ';'.join(f'{d:%Y-%m-%d}:{v:g}' for d, v in splits.items()))
    # `build_panel` drops rows older than `history_years` + 1 before today;
    # the cut-off's year is in the key, so a reused panel is at most a year
    # behind it - immaterial while XBRL starts in 2009.
    cutoff = (pd.Timestamp.today() - pd.DateOffset(years=cfg.history_years + 1)).year
    parts = [_code_hash(), CONFIG_VERSION, cfg.history_years, cutoff, facts_sha1, split_sig,
             {k: meta.get(k) for k in ('name', 'sic', 'sicDescription', 'fiscalYearEnd',
                                       'exchanges')}]
    return hashlib.sha1(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def _entry_paths(t: str) -> tuple:
    stem = t.replace('/', '_')
    return FUND_DIR / f'{stem}.parquet', FUND_DIR / f'{stem}.json'


def _json_default(o):
    if isinstance(o, (pd.Timestamp, dt.date)):
        return o.isoformat()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def save_entry(t: str, key: str, panel: pd.DataFrame, info: dict) -> None:
    pq, js = _entry_paths(t)
    try:
        # The key goes last, so a panel whose write was interrupted is never
        # paired with a key that vouches for it.
        js.unlink(missing_ok=True)
        panel.to_parquet(pq)
        js.write_text(json.dumps({'key': key, 'info': info}, default=_json_default))
    except Exception as exc:  # noqa: BLE001
        print(f'  panel cache write failed for {t}: {exc}')


def cached_entry(t: str, key, cfg: ValuationConfig, source: str = 'cache'):
    """``{'panel', 'info'}`` built earlier from the same inputs, or None.

    The date-dependent parts of `info` are recomputed (`_dated_info`), and
    `facts_source` is this run's, so a reused panel served from an
    out-of-date document is still flagged as one.
    """
    if key is None:
        return None
    pq, js = _entry_paths(t)
    try:
        blob = json.loads(js.read_text())
        if blob.get('key') != key:
            return None
        panel = pd.read_parquet(pq)
        panel.index = pd.DatetimeIndex(panel.index, name='FYEnd')
    except Exception:  # noqa: BLE001 - missing, unreadable: rebuild
        return None
    info = blob['info']
    for k in _DATE_KEYS:
        if info.get(k) is not None:
            info[k] = pd.Timestamp(info[k])
    info['facts_source'] = source
    return {'panel': panel, 'info': _dated_info(panel, info, cfg)}


def build_entry(t: str, blob: dict, cfg: ValuationConfig, splits, key=None) -> dict:
    """Build one company's panel from its fetched document, caching it under
    `key` when one is given.

    Used as `sources.fetch_many(..., consume=...)` so each companyfacts
    document is released as soon as its panel exists. Returns
    ``{'panel', 'info'}`` or ``{'error'}``.
    """
    try:
        panel, info = build_panel(t, blob['facts'], blob.get('meta') or {}, cfg, splits=splits)
    except Exception as exc:  # noqa: BLE001
        return {'error': f'{type(exc).__name__}: {exc}'}
    if panel is None or panel.empty:
        return {'error': info.get('error', 'empty panel')}
    info['facts_source'] = blob.get('source')
    if key is not None:
        save_entry(t, key, panel, info)
    return {'panel': panel, 'info': info}


def collect_built(built: dict, verbose: bool = True) -> tuple:
    """Split `build_entry` results into panels and infos, and report."""
    panels, infos, errors = {}, {}, []
    for t, r in built.items():
        if 'error' in r:
            errors.append(f'{t}: {r["error"]}')
            continue
        panels[t], infos[t] = r['panel'], r['info']

    if verbose:
        if panels:
            yrs = [i['n_years'] for i in infos.values()]
            print(f'  panels: {len(panels)} built, median {int(np.median(yrs))} '
                  f'fiscal years (range {min(yrs)}-{max(yrs)})')
            groups = pd.Series([i['group'] for i in infos.values()]).value_counts()
            nonop = {k: int(v) for k, v in groups.items() if k != 'operating'}
            if nonop:
                print(f'  carve-outs by filer SIC: {nonop}')
            bad = [t for t, i in infos.items() if not i.get('per_share_reliable', True)]
            if bad:
                print(f'  breaks in the share count for {len(bad)}: '
                      f'{", ".join(bad[:6])}'
                      + (' ...' if len(bad) > 6 else '')
                      + ' (split-shaped ones are checked against the prices before any '
                        'history is cut)')
        if errors:
            print(f'  {len(errors)} panel failure(s): {"; ".join(errors[:5])}'
                  + (' ...' if len(errors) > 5 else ''))
    return panels, infos
