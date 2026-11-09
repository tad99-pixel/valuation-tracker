"""Graham / Buffett valuation tracker.

A fundamental counterpart to `scanner/`: where that one asks whether a price
moves over three days, this one asks what a business is worth and what it
costs. The two share only the index-membership loader; prices here are kept
in their own cache, split-adjusted but not dividend-adjusted.

    from valuation import config, sources, fundamentals, graham, buffett, score

Reading order for the output is in VALUATION_README.md; the short version is
that the per-criterion table is the analysis and the composite score is only a
way to sort a watchlist.
"""

from . import config  # noqa: F401

__all__ = ['config', 'sources', 'fundamentals', 'metrics', 'graham',
           'buffett', 'score', 'tracker', 'dashboard']
