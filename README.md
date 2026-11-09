# Valuation tracker

Benjamin Graham's tests from *The Intelligent Investor* and Warren Buffett's
quality standards, applied to companies' own SEC filings and re-run after the
US close every trading day.

**Dashboard: https://tad99-pixel.github.io/valuation-tracker/**

- **How it works and how to read it:** [VALUATION_README.md](VALUATION_README.md)
- **What is tracked:** [valuation_watchlist.txt](valuation_watchlist.txt) - 49
  names chosen by hand and, through its `@sp500` line, the rest of the S&P 500.
  Edit it and push, and the tracker re-runs with the new list.
- **Alerts:** a run that finds a company moving - its verdict, its margin of
  safety, its price crossing the estimate - comments on the open issue
  labelled `tracker-alerts`, so GitHub notifies whoever watches the repository.
- **History:** every run is kept on the `data` branch
  (`valuation_tracker_store/`), and the site links to a workbook of the recent
  runs.

Research tooling. Every figure applies a stated rule to public filings; none
of it is investment advice.
