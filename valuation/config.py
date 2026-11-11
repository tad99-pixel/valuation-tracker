"""Thresholds and parameters for the Graham / Buffett valuation tracker.

Every number here is either quoted from a primary source or marked as a
derived heuristic. The distinction matters: Graham wrote specific numeric
tests, Buffett mostly did not, and a good deal of what circulates as
"Buffett's criteria" comes from third-party books rather than his letters.
Mislabelling the second kind as the first is how a screen acquires false
authority, so `PROVENANCE` below records the source of each gate and the
dashboard prints it.

Graham's dollar thresholds were written in 1973 and are inflation-adjusted at
runtime from FRED's CPIAUCSL rather than hard-coded, because "$100 million of
annual sales" in 1973 money is a materially different company today.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / 'data_cache'
FUND_DIR = CACHE_DIR / 'fundamentals'      # one parquet panel per ticker
FACTS_DIR = CACHE_DIR / 'sec_facts'        # raw companyfacts JSON, gzipped
MACRO_DIR = CACHE_DIR / 'macro'            # FRED series
STATE_DIR = ROOT / 'valuation_state'
REPORT_DIR = ROOT / 'valuation_reports'

TRACKER_PATH = Path(os.environ.get('VALUATION_TRACKER')
                    or (ROOT / 'valuation_tracker.xlsx'))
LOG_PATH = ROOT / 'valuation_daily.log'

# Stamped on every stored valuation. Bump whenever a threshold or a model
# changes, so a tracker history is never silently pooled across two different
# definitions of "margin of safety".
CONFIG_VERSION = '2026-11-10-r9'

for _d in (CACHE_DIR, FUND_DIR, FACTS_DIR, MACRO_DIR, STATE_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Graham
# ---------------------------------------------------------------------------

@dataclass
class GrahamConfig:
    """The Intelligent Investor, 4th revised edition (1973), chapters 14-15.

    Criteria 1-7 are the defensive investor's tests from chapter 14. The
    `enterprising_*` block is the ten-point screen Graham described in 1976,
    near the end of his life, which is a different and stricter instrument -
    it is reported separately rather than averaged in.
    """

    # --- 1. Adequate size (ch. 14) ---
    # "not less than $100 million of annual sales" for an industrial company,
    # and "$50 million of total assets" for a public utility. 1973 dollars;
    # scaled by CPI at runtime - see `inflate` below.
    min_revenue_1973: float = 100e6
    min_assets_1973_utility: float = 50e6
    # Graham's edition year, the base for the CPI adjustment.
    base_year: int = 1973

    # --- 2. Strong financial condition (ch. 14) ---
    min_current_ratio: float = 2.0
    # "long-term debt should not exceed the net current assets"
    max_ltdebt_to_working_capital: float = 1.0
    # Graham allowed utilities a different test: debt <= 2x equity.
    max_debt_to_equity_utility: float = 2.0

    # --- 3. Earnings stability (ch. 14) ---
    # "Some earnings for the common stock in each of the past ten years."
    earnings_stability_years: int = 10

    # --- 4. Dividend record (ch. 14) ---
    # "Uninterrupted payments for at least the past 20 years."
    # XBRL only reaches back to roughly 2009, so 20 years cannot be verified
    # from this data source for most companies. The test therefore reports the
    # longest streak it can *prove* and is marked N/A rather than FAIL when the
    # window is too short - see `graham.py`. A test that fails for lack of data
    # is not a failed test.
    dividend_years: int = 20

    # --- 5. Earnings growth (ch. 14) ---
    # "a minimum increase of at least one-third in per-share earnings in the
    # past ten years, using three-year averages at the beginning and end."
    growth_window_years: int = 10
    growth_average_years: int = 3
    min_eps_growth_total: float = 1 / 3

    # --- 6-7. Moderate price (ch. 14) ---
    # "current price should not be more than 15 times average earnings of the
    # past three years" and "not more than 1-1/2 times the book value".
    max_pe_3yr_avg: float = 15.0
    max_price_to_book: float = 1.5
    # "a multiplier of earnings of 15 and of 1.5 for book value... a current
    # ratio of price to assets-times-earnings of no more than 22.5"
    max_pe_times_pb: float = 22.5

    # --- Graham Number: sqrt(22.5 * EPS * BVPS) ---
    graham_number_constant: float = 22.5

    # --- Net current asset value (ch. 15, the classic deep-value screen) ---
    # Graham bought below two-thirds of NCAV.
    ncav_discount: float = 2 / 3

    # --- The 1976 ten-point screen ---
    enterprising_earnings_yield_x_aaa: float = 2.0      # E/P >= 2x AAA yield
    enterprising_pe_vs_5yr_high: float = 0.4            # P/E <= 40% of 5y high
    enterprising_div_yield_x_aaa: float = 2 / 3          # div yld >= 2/3 AAA
    enterprising_price_to_tangible_book: float = 2 / 3
    enterprising_price_to_ncav: float = 2 / 3
    enterprising_debt_to_tangible_book: float = 1.0
    enterprising_debt_to_ncav: float = 2.0
    enterprising_min_eps_cagr: float = 0.07
    enterprising_max_eps_declines: int = 2               # declines of >=5%
    enterprising_decline_threshold: float = 0.05

    # --- Graham's 1962 growth formula, V = EPS * (8.5 + 2g) ---
    # Included because it is widely quoted, and flagged because Graham himself
    # stopped defending it. `g` is capped: the formula is linear in growth and
    # produces absurd values on a high-growth input.
    growth_formula_base: float = 8.5
    growth_formula_multiplier: float = 2.0
    growth_formula_max_g: float = 15.0      # percentage points
    # The 1974 revision divides by the prevailing AAA yield, normalised to the
    # 4.4% yield that prevailed when the formula was written.
    growth_formula_norm_yield: float = 4.4

    def inflate(self, cpi_ratio: float) -> dict:
        """Graham's 1973 dollar thresholds, restated in today's money."""
        return {
            'min_revenue': self.min_revenue_1973 * cpi_ratio,
            'min_assets_utility': self.min_assets_1973_utility * cpi_ratio,
            'cpi_ratio': cpi_ratio,
        }


# ---------------------------------------------------------------------------
# Buffett
# ---------------------------------------------------------------------------

@dataclass
class BuffettConfig:
    """Quality gates in the spirit of the Berkshire letters.

    Buffett's own acquisition criteria are qualitative - "a business we can
    understand", "favourable long-term economics", "able and trustworthy
    management", "a sensible price tag". Only the last is directly computable.
    What follows are *proxies*, and `PROVENANCE` marks which have a direct
    textual basis and which are third-party operationalisations.
    """

    # Return on equity. Buffett returns to ~15% repeatedly as the mark of a
    # business worth owning, and criticises reported EPS growth that comes from
    # retained earnings rather than returns on them.
    min_roe: float = 0.15
    # Measured over this many years, not one - a single year's ROE is an
    # accident. `roe_consistency_years` of history must clear the bar.
    quality_window_years: int = 10
    min_years_above_roe: int = 8

    # Return on invested capital. Not a Buffett term, but the cleanest
    # available proxy for "favourable long-term economics" since it is
    # insensitive to leverage in a way ROE is not.
    min_roic: float = 0.12

    # Debt. Buffett's stated preference is for businesses with little of it.
    # The quantitative form - long-term debt repayable from a few years of
    # earnings - is a common operationalisation, not his phrasing.
    max_ltdebt_to_net_income: float = 4.0

    # Margins. These come from Mary Buffett's *Buffettology*, a third-party
    # book, and are included because they are widely used - not because
    # Buffett published them.
    min_gross_margin: float = 0.40
    min_net_margin: float = 0.10
    # Durability is the point, so dispersion matters more than the level: a
    # coefficient of variation on gross margin above this is evidence against
    # a durable advantage.
    max_gross_margin_cv: float = 0.15

    # Free cash flow conversion.
    min_fcf_margin: float = 0.05

    # The one-dollar premise (1984 letter): "unrestricted earnings should be
    # retained only where there is a reasonable prospect... that for every
    # dollar retained... at least one dollar of market value will be created".
    one_dollar_window_years: int = 10
    min_market_value_per_retained_dollar: float = 1.0
    # Our threshold, not Buffett's: below this share of earnings kept (after
    # dividends and buybacks), the ratio has a sliver for a denominator and
    # measures noise - Home Depot kept $0.8bn of $130bn and "created $245 per
    # dollar retained". The premise is then reported as having too little
    # retention to judge, like a company that kept nothing.
    min_retained_share: float = 0.10

    # Share count: dilution is a cost the income statement does not show.
    max_share_count_cagr: float = 0.01

    # --- owner earnings (1986 letter, appendix) ---
    # "reported earnings plus depreciation, depletion, amortization and certain
    # other non-cash charges, less the average annual amount of capitalized
    # expenditures for plant and equipment etc. that the business requires to
    # fully maintain its long-term competitive position and its unit volume."
    #
    # The maintenance portion of capex is not disclosed by anyone, which
    # Buffett acknowledged in the same appendix ("the maintenance portion...
    # is a guess"). Two estimators are computed and both are reported:
    #   'greenwald' - capex less growth capex, growth capex being
    #                 (PP&E / sales) * change in sales  (Greenwald's method)
    #   'da'        - treat D&A as the maintenance requirement
    maintenance_capex_method: str = 'greenwald'

    # --- discounted cash flow ---
    # A DCF's output is dominated by these three numbers, which is why the
    # dashboard shows a reverse DCF alongside it. The discount rate is a
    # required return, not a measured quantity: the return required of the
    # business before debt. Each company's equity rate (the DCF's) and
    # whole-company rate (EPV's) are derived from it and its leverage - see
    # `buffett.required_returns`.
    discount_rate: float = 0.09
    stage1_years: int = 10
    terminal_growth: float = 0.025
    # Stage-1 growth is faded linearly from the historical rate to terminal.
    # Unfaded growth over ten years is the single most common way a DCF is
    # talked into any answer you like.
    fade_growth: bool = True
    max_stage1_growth: float = 0.15
    min_stage1_growth: float = -0.05
    # Credit earnings growth only as far as revenue growth supports it. Earnings
    # can outgrow revenue only through margin expansion, which is finite, and
    # earnings growth is the input most distorted by base effects: Southern
    # Company's per-share owner earnings "grew 45.6% a year" off a near-zero
    # opening block, Exxon's 12.5% off the 2015-17 oil trough. Revenue is far
    # less prone to either. Taking the lower of the two is conservative by
    # construction, and both rates are reported so the choice is visible.
    cap_growth_at_revenue: bool = True

    # Earnings Power Value (Greenwald): normalised operating earnings
    # capitalised at the whole-company rate, with no growth credited at all.
    epv_normalisation_years: int = 7
    effective_tax_rate: float = 0.21


# ---------------------------------------------------------------------------
# Scoring / reporting
# ---------------------------------------------------------------------------

@dataclass
class ScoreConfig:
    """How the individual tests roll up, and when they refuse to.

    The weights are a presentation choice, not a finding. They are here so the
    dashboard can sort a watchlist; nothing in Graham or Buffett says a
    composite score is meaningful, and the per-criterion table is the part
    worth reading.
    """

    weight_graham_defensive: float = 0.30
    weight_buffett_quality: float = 0.40
    weight_valuation: float = 0.30

    # A score computed from a handful of available tests is not comparable with
    # one computed from all of them. Below this share of applicable tests
    # actually measurable, the verdict is INSUFFICIENT DATA rather than a
    # number - the same principle as the scanner's "a missing measurement must
    # never print as a measured zero".
    min_data_completeness: float = 0.60

    # Margin of safety bands, applied to the median of the applicable
    # intrinsic-value models.
    mos_strong: float = 0.33      # Graham's own rule of thumb
    mos_adequate: float = 0.15
    # Dispersion guard: half the range of the value models, relative to their
    # median. Above this the margin of safety is reported as unreliable rather
    # than as a number. 0.5 puts the highest estimate at about three times the
    # lowest - a tight margin of safety computed off models that span 3x is
    # arithmetic, not analysis. (0.60 had been set against a different
    # measure and allowed a 4x spread, contradicting this comment.)
    max_model_dispersion: float = 0.50

    # The coarse verdict labels. "Passes" means at least this share of the
    # applicable tests in a block passed; a company with no measured margin
    # but a composite at or above `watch_composite` is worth watching.
    quality_pass_rate: float = 0.60
    graham_pass_rate: float = 0.60
    watch_composite: float = 0.60


# ---------------------------------------------------------------------------
# Instrument classification
# ---------------------------------------------------------------------------

# Graham treated public utilities separately and warned that financial
# companies need different tests. Applying the current-ratio and
# net-current-asset tests to a bank is not a strict reading of Graham - it is
# a category error, and it marks every bank in the universe as a failure.
#
# SIC ranges come from the SEC submissions API, so the classification is the
# filer's own, not a guess from the ticker.
SIC_GROUPS = {
    'bank':      [(6020, 6036), (6060, 6062), (6120, 6120), (6199, 6199)],
    'insurance': [(6311, 6411)],
    'reit':      [(6798, 6798)],
    'utility':   [(4900, 4991)],
    'financial': [(6000, 6799)],     # catch-all, evaluated last
}

# Tests and models that do not transfer to a group, keyed by the names the
# criteria and models check. This table is the only place a carve-out is
# decided: a key here makes the test N/A for a *structural* reason, which
# does not count against data completeness. Every key is read somewhere -
# `tests/test_valuation.py` checks that none is dead.
_NO_CLASSIFIED_BALANCE_SHEET = {
    # Deposits, float and broker payables have no "current" in the operating
    # sense, and most such filers present an unclassified balance sheet.
    'current_ratio', 'ltdebt_vs_working_capital', 'ncav', 'price_to_ncav',
    'debt_to_ncav',
}
NA_CRITERIA_BY_GROUP = {
    # For a bank, borrowed money is the raw material, not a risk to the
    # business - JPMorgan "passed" a debt-repayable-from-earnings test at 1.1
    # years on the $65bn its tags happen to capture, against several hundred
    # billion of actual borrowings. A pass that meaningless is worse than N/A.
    # Owner earnings and EPV need operating cash flows separable from the
    # balance sheet; for a bank or an insurer they are not, and both are
    # valued on justified P/B instead.
    'bank': _NO_CLASSIFIED_BALANCE_SHEET | {
        'roic', 'gross_margin', 'fcf_margin', 'max_ltdebt_to_net_income',
        'debt_to_tangible_book', 'owner_earnings', 'epv'},
    'insurance': _NO_CLASSIFIED_BALANCE_SHEET | {
        'roic', 'gross_margin', 'owner_earnings', 'epv'},
    # Brokers, card lenders, asset managers: the catch-all for SIC 6000-6799.
    # Only the balance-sheet tests are carved - the earnings tests still mean
    # something for a fee business.
    'financial': _NO_CLASSIFIED_BALANCE_SHEET | {'gross_margin'},
    # GAAP EPS is distorted for a REIT (depreciation on property that usually
    # appreciates), so every test built on it is carved out - not only P/E.
    # Carving criterion 6 alone left R1, R2 and the EPS growth tests failing
    # Realty Income on the same refused number. `pe_times_pb` withdraws only
    # the P/E route to criterion 7; the P/B route still applies. EPV
    # capitalises the same depreciated operating income.
    'reit': {'ncav', 'price_to_ncav', 'debt_to_ncav', 'gross_margin',
             'max_pe_3yr_avg', 'graham_number', 'pe_times_pb', 'earnings_yield',
             'pe_vs_high', 'eps_growth', 'eps_cagr', 'eps_declines', 'epv'},
    # Graham's asset test replaces the sales test, and most utilities report
    # no cost of revenue, so there is no gross margin to judge.
    'utility': {'min_revenue', 'gross_margin'},
}

# Groups valued on justified P/B and capitalised earnings, at the required
# return without a leverage adjustment: financials, whose owner earnings cannot
# be separated from the balance sheet, and utilities, whose regulators set
# their return as a return on book equity. A utility's DCF and EPV still count
# where they compute; before utilities joined, none of 37 had a measurable
# margin - most had one model, or none, once net debt swamped their EPV.
BOOK_VALUED_GROUPS = ('bank', 'insurance', 'financial', 'utility')

# Companies that moved to a new SEC registration - a reorganisation into a new
# holding company, a change of state - file their history under the old CIK
# and everything after under the new one, and EDGAR does not link the two.
# Current CIK -> earlier CIKs, newest first; their facts are merged, so a
# company's ten-year record survives the change instead of restarting with one
# quarterly report. An earlier registration counts only for what it filed
# before the next one's first annual report (`sources.merge_facts`).
CIK_SUCCESSIONS = {
    # ExxonMobil Holdings Corp (Texas) took over the XOM listing in 2026; the
    # history is filed by Exxon Mobil Corp (New Jersey).
    2115436: (34088,),
    # Texas Pacific Land Corp replaced Texas Pacific Land Trust in January
    # 2021, one share for each sub-share certificate; the trust filed 1996-2020.
    1811074: (97517,),
    # New holding companies of 2023-25, each with a year or two of filings of
    # its own; the index table points the ticker at the new one.
    2012383: (1364742,),    # BlackRock (2024, with GIP) <- BlackRock, now BlackRock Finance
    1996862: (1144519,),    # Bunge Global SA (2023) <- Bunge Limited
    2011641: (1832433,),    # Ferguson Enterprises (2024) <- Ferguson plc
    2041610: (813828,),     # Paramount Skydance (2025) <- Paramount Global
}

# Companies that tag their EPS and share counts only per share class, which
# SEC's companyfacts feed drops - leaving them with no per-share history at
# all. Ticker -> the class member whose figures match the quoted price; see
# `valuation.classes`, which reads them from each annual report's own XBRL.
SHARE_CLASSES = {
    'V': 'CommonClassAMember',        # Visa: Class A trades; B and C convert into it
    'BRK-B': 'CommonClassBMember',    # Berkshire: per Class B share (1/1,500th of an A)
}


# ---------------------------------------------------------------------------
# Provenance - printed in the dashboard so a heuristic is never mistaken for
# a quotation.
# ---------------------------------------------------------------------------

PROVENANCE = {
    # key: (source tier, citation)
    'min_revenue': ('graham', 'Intelligent Investor ch.14, criterion 1 (1973 $, CPI-adjusted)'),
    'financial_condition': ('graham', 'Intelligent Investor ch.14, criterion 2 (utilities: debt vs equity)'),
    'earnings_stability': ('graham', 'Intelligent Investor ch.14, criterion 3'),
    'dividend_record': ('graham', 'Intelligent Investor ch.14, criterion 4'),
    'eps_growth': ('graham', 'Intelligent Investor ch.14, criterion 5'),
    'max_pe_3yr_avg': ('graham', 'Intelligent Investor ch.14, criterion 6'),
    'price_to_book': ('graham', 'Intelligent Investor ch.14, criterion 7, with its P/E x P/B <= 22.5 alternative'),
    'graham_number': ('graham-derived', 'sqrt(22.5 x EPS x BVPS), from the ch.14 combined rule'),
    'ncav': ('graham', 'Intelligent Investor ch.15 / Security Analysis - net current asset value'),
    'enterprising': ('graham', 'Graham 1976 ten-point screen (Medical Economics interview)'),
    'growth_formula': ('graham-disavowed', 'Intelligent Investor ch.11 (1962); Graham later cautioned against it'),

    'owner_earnings': ('buffett', 'Berkshire 1986 letter, appendix - definition quoted'),
    'maintenance_capex': ('derived', 'not disclosed by filers; Greenwald estimator, or D&A fallback'),
    'one_dollar_premise': ('buffett', 'Berkshire 1984 letter - retained earnings test'),
    'min_roe': ('buffett-paraphrase', 'repeated ~15% ROE standard across letters; not a published screen'),
    'min_roic': ('derived', 'leverage-insensitive proxy for "favourable long-term economics"'),
    'max_ltdebt_to_net_income': ('derived', 'common operationalisation of Buffett\'s low-debt preference'),
    'min_gross_margin': ('third-party', 'Mary Buffett, "Buffettology" - not Buffett\'s own text'),
    'min_net_margin': ('third-party', 'Mary Buffett, "Buffettology" - not Buffett\'s own text'),
    'max_gross_margin_cv': ('derived', 'margin stability as a moat proxy'),
    'share_count': ('buffett', 'repeated criticism of dilution; threshold is ours'),
    'dcf': ('derived', 'standard two-stage DCF; Buffett describes the concept, not these inputs'),
    'reverse_dcf': ('derived', 'growth implied by the current price - the DCF read backwards'),
    'epv': ('third-party', 'Greenwald, "Value Investing" - earnings power value'),
    'justified_pb': ('derived', 'residual-income model, single stage: P/B = (ROE - g)/(r - g); banks and insurers only'),
}

SOURCE_TIERS = {
    'graham': 'Quoted from Graham',
    'graham-derived': 'Arithmetic from a Graham rule',
    'graham-disavowed': 'Graham published it, then cautioned against it',
    'buffett': 'Quoted from a Berkshire letter',
    'buffett-paraphrase': 'Buffett\'s standard, our threshold',
    'third-party': 'From a third-party book, not the principals',
    'derived': 'Our own proxy',
}


@dataclass
class ValuationConfig:
    """One end-to-end configuration."""

    graham: GrahamConfig = field(default_factory=GrahamConfig)
    buffett: BuffettConfig = field(default_factory=BuffettConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)

    # How much annual history to request. Graham's longest test needs 10 years
    # of earnings and 20 of dividends; XBRL starts around 2009.
    history_years: int = 20
    # SEC serves no Last-Modified or ETag, so whether a filer has reported
    # since the last download is read from its filing index (the submissions
    # feed, ~1/20th the size of the facts) every `filings_check_days`, and the
    # companyfacts document is refetched only when a newer 10-K or 10-Q is
    # listed - or, as a backstop for amendments the index check cannot see,
    # once it is `facts_max_age_days` old.
    filings_check_days: int = 2
    facts_max_age_days: int = 90
    sec_pause: float = 0.12          # SEC asks for <= 10 requests/second

    # A company that has not filed in this long is probably delisted or
    # acquired, and its last balance sheet is not a current fact.
    max_filing_age_days: int = 450


DEFAULT = ValuationConfig()
