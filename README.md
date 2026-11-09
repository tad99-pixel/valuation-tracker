# Tracker history

This branch is the valuation tracker's record. `valuation_tracker_store/`
holds every run: `history`, `runlog` and `changes` are append-only, and
`latest` is the most recent scorecard. The GitHub Actions workflow on `main`
checks this branch out, adds the day's run, and commits it back.

The code lives on `main`. The dashboard is at
https://tad99-pixel.github.io/valuation-tracker/, with a workbook of the full
history linked from its footer.
