# Valuation Tracker

A fundamental counterpart to the scanner. Where `scanner/` asks whether a price
moves over three days, `valuation/` asks what a business is worth and what it
costs, by the tests Benjamin Graham wrote down in *The Intelligent Investor*
and the standards Warren Buffett describes in the Berkshire letters.

```bash
python run_valuation.py                         # the watchlist file
python run_valuation.py --tickers KO,AAPL,JPM   # just these
python run_valuation.py --universe sp500        # every constituent (~5 min first run)
python run_valuation.py --open                  # open the dashboard when done
python run_valuation.py --force                 # refetch filings and prices, rebuild every panel
python tests/test_valuation.py                  # regression tests, offline
```

Each run appends to the history store in `valuation_tracker_store/`, refreshes
`valuation_tracker.xlsx` (a view of the last 400 days), and writes
`valuation_dashboard.html` (one self-contained file; also archived in
`valuation_reports/`, every day for a month and monthly after that). Edit
`valuation_watchlist.txt` to change what is tracked.

Research tooling, as in the scanner. An intrinsic value is an opinion with
arithmetic attached; nothing here is a recommendation to buy or sell anything.

---

## What it measures

| Instrument | Source | What it asks |
|---|---|---|
| **Graham's seven defensive tests** | *The Intelligent Investor*, ch. 14 | size, financial condition (current ratio and debt vs working capital; for utilities, debt vs equity), 10 years of earnings, 20 years of dividends, 10-year EPS growth, P/E <= 15, P/B <= 1.5 (or P/E x P/B <= 22.5) - seven tests, each counted once |
| **Graham's 1976 ten-point screen** | his last published screen | five reward tests (earnings yield vs AAA, P/E vs its 5-year high, dividend yield, price vs tangible book, price vs NCAV) and five risk tests |
| **Buffett quality proxies** | the Berkshire letters, plus labelled third-party proxies | ROE across the cycle, ROIC, debt repayable from earnings, margin level and stability, cash conversion, dilution, the 1984 one-dollar premise |
| **Intrinsic value** | four earnings-based models | two-stage DCF on owner earnings, earnings power value, 15 x 3-year average EPS, justified P/B for banks and insurers |
| **What the price assumes** | reverse DCF | the growth today's price needs, beside the growth the record supports |

Every test reports **Pass, Fail or N/A**, and every threshold carries its
provenance - *quoted from Graham*, *quoted from a Berkshire letter*,
*Buffett's standard, our threshold*, *third-party book*, or *our own proxy*.
"Gross margin above 40%" comes from Mary Buffett's *Buffettology*, not from
Warren Buffett, and the dashboard says so.

Graham's 1973 dollar thresholds are inflated by CPI at runtime (x7.6 today), and
his bond-relative tests use the live AAA corporate yield from FRED.

---

## Reading the output honestly

**Read "what the price assumes" first.** A forward DCF converts an assumption
into a valuation; the reverse DCF converts today's price into the assumption it
embeds, which can be checked against what the business has actually done. It
needs no growth forecast - but it is not free of inputs. Coca-Cola's price
implies 20% growth a year at an 8% discount rate and 29% at 10%, so the rate
one point either side is reported with it. Coca-Cola's conclusion holds
across that range - its revenue has grown 1.5% a year and its owner earnings
7% - so it is a finding. Exxon's does not: its price implies 2% growth at 9%,
below the 4.8% its record supports, but 5.6% at 10%.

**N/A is not Fail.** A test is N/A when its data does not exist, or when it does
not apply - a current ratio for a bank is a category error. Collapsing the two
into Fail is the commonest way a Graham screen is built wrong: it rejects every
bank, insurer and REIT for reasons Graham never gave.

**The estimate is a range.** The median of the earnings-based models is
reported with its spread. When the highest estimate is more than three times
the lowest ((highest - lowest) / (highest + lowest) > 0.5), or only one model
applies, the margin of safety is **withheld**, and it scores zero - the same
as a measured premium. A margin you cannot demonstrate is not one you have.

**The Graham Number and NCAV are price limits, not values.** They are the
prices at which Graham's rules would be satisfied. On an asset-light business
at 40x book the Graham Number does not say what the company is worth; it says
Graham would never have bought it. They are shown, never averaged in.

**The composite only sorts the list.** Neither author weighted tests into one
number. The verdict is deliberately coarse:

| Verdict | Means |
|---|---|
| Candidate | quality passes, Graham's defensive tests mostly pass, measured margin >= 15% |
| Cheap, quality passes | as above without the Graham side |
| Good business, fair price | quality passes, priced within 15% of value - Buffett's "wonderful company at a fair price" |
| Good business, dear | quality passes, priced above the estimate |
| Good business, value unclear | quality passes, the models disagree too much to call a price |
| Statistically cheap only | measured margin, quality fails - Graham's territory, and the value-trap risk |
| Watch / No | neither |
| Insufficient data / No price | too few tests measurable; or delisted / acquired |

---

## Running on GitHub

`.github/workflows/valuation.yml` runs the tracker at 22:30 UTC on weekdays
(after the US close), on any push that changes the code or the watchlist, and
on demand from the Actions tab ("Run workflow"). Each run:

1. runs the regression tests, and stops if any fail;
2. restores the previous run's downloads (filings, prices, splits) from the
   Actions cache, so a normal day fetches only what changed;
3. runs the tracker against the history store checked out from the `data`
   branch, and commits the new run back to it;
4. publishes the dashboard as the Pages site, with the workbook beside it.

The `data` branch is the record. The cache is only a speed-up: if GitHub
evicts it, the next run downloads everything again and nothing is lost.

**FRED needs an API key on GitHub.** Its keyless CSV download answers a home
connection but leaves requests from GitHub's servers hanging, so without a
key the run falls back to the last FRED data it has (a copy kept on the
`data` branch, dated on the dashboard). Get a free key at
https://fredaccount.stlouisfed.org/apikeys and add it as a repository secret
named `FRED_API_KEY`:

```bash
gh secret set FRED_API_KEY --repo tad99-pixel/valuation-tracker
```

SEC's ticker list (www.sec.gov) refuses GitHub's servers too; the run uses
the S&P 500 and 600 index tables instead, which cover any constituent. For a
ticker outside them, or if SEC's data endpoints start refusing requests
(403s in the run log), add a secret named `VALUATION_USER_AGENT` with a
contact, for example `ValuationTracker/1.0 you@example.com`; SEC asks
automated clients to identify themselves.

GitHub disables scheduled workflows in a public repository after 60 days with
no activity; if that happens, the Actions tab offers a button to re-enable it.

---

## Why the data layer is most of the code

Fundamentals come from the SEC's XBRL `companyfacts` API rather than a vendor
feed, because Graham's tests need ten years of earnings and twenty of dividends
and the free alternatives serve four. Raw XBRL produces confidently wrong
numbers in at least ten distinct ways, each found on a real company and each
now pinned by a test:

| Trap | Seen on | Handling |
|---|---|---|
| The same period filed many times with different values | Apple FY2012 EPS: 44.15 as filed, 6.31 restated | keep the latest-filed value |
| `form=10-K, fp=FY` rows that are quarters | Apple: 90-370-day periods under the same labels | select annual periods by measured length |
| `fy` is the filing's year, not the period's | every company | key on the period end date |
| Per-share history on several share bases at once | Apple pre-2014, post-7:1, post-4:1 | divide each fact by the splits after **its filing date** |
| Share counts filed in thousands or millions | NVIDIA FY08-09, Hershey 2009, Berkshire 08-09 | repair exact orders of magnitude when the global and local medians agree |
| Issued shares mistaken for outstanding | Exxon: 8.0bn issued vs 4.7bn outstanding | outstanding only |
| A stale, pre-split cover-page share count | Mastercard: 122.5m vs 906m - a 7.4x overvaluation | cover count vetted for age and agreement |
| Debt filed as overlapping slices | Prologis, Realty Income, Coca-Cola | compose by rule; unknown debt stays unknown |
| D&A filed as parts | AbbVie: $0.76bn depreciation + $7.38bn amortisation | total tag, else depreciation + amortisation |
| Tags that change mid-history | Verizon capex (2019), Kroger PP&E (2021) | extended tag chains |
| A per-share basis that differs from the traded class | Berkshire (Class A filings, Class B ticker) | price x shares must describe a plausible company, or every price comparison is withheld |

Prices are split-adjusted but **not** dividend-adjusted. The scanner's cache is
total-return adjusted - right for a return model, wrong here: Exxon's November
2016 close reads $54.31 there against an actual $83.65, which flattered every
test that needs a historical market value.

---

## Windows are positional

"The last three years" means the three most recent fiscal years in the panel.
An earlier version dropped missing values before cutting the window, which
reached back across gaps: Berkshire's "3-year average EPS" was its 2012-14
figures. Growth rates are annualised over elapsed time, and the ten-year EPS
growth test needs two three-year blocks whose centres are ten years apart.

Every growth rate that feeds a valuation is the **lower** of the company's
total growth and its growth per share. Per share where it issued stock:
Realty Income's owner earnings compounded at ~15% while its share count
compounded at ~14%, and crediting today's holders with the total rate valued
them on growth they had been diluted out of. Total where it bought stock
back: the DCF already counts every dollar of owner earnings as value to
today's shares, including the dollars later spent on buybacks, so crediting
the per-share growth those buybacks produce counted them twice. An earlier
version did, and it overstated Apple's DCF by a fifth.

---

## What accumulates, and where

The record is the store, `valuation_tracker_store/`: `history`, `runlog` and
`changes` are append-only parquet files, and `latest` is replaced each run. A
run appends a few kilobytes and rewrites nothing already there. The workbook
is a view regenerated from the store, with `History` and `Changes` cut to the
last 400 days, so its cost no longer grows with every day the tracker runs.

| Sheet | Contents |
|---|---|
| `RunLog` | one row per run: universe, verdict counts, AAA yield, CPI, config version |
| `Latest` | the current scorecard, one row per company |
| `Changes` | what moved since the last run: verdict and margin-band changes, and the price crossing the estimate, the Graham Number, or two-thirds of NCAV |
| `History` | price, value range and margin per company per run |
| `Criteria` | every test, its verdict, value, threshold, reasoning and source tier |
| `Models` | every intrinsic-value model, its role and how it was computed |
| `Provenance` | where each threshold comes from |

`CONFIG_VERSION` is stamped on every row; bump it in `valuation/config.py`
whenever a threshold or model changes, so history is never pooled across two
definitions of "margin of safety". A workbook open in Excel no longer matters:
the run is already in the store, and the view is rebuilt next time. Only a
store that cannot be written stashes the run in `valuation_pending/`, to be
merged by the next run. The first run after upgrading seeds the store from the
existing workbook; if that workbook is open at that moment, the run stops
rather than start an empty history beside it.

---

## What a run fetches

SEC serves no `Last-Modified` or `ETag`, so nothing can be revalidated
cheaply. Instead:

| What | When |
|---|---|
| Filing index (`submissions`: SIC, entity type, latest 10-K/10-Q date) | every 2 days |
| `companyfacts` document | only when the index lists a report newer than the cached copy, or after 90 days |
| Built panel | reused when the document, the split history, the filer's metadata, the config and the source of `fundamentals.py` / `config.py` are all unchanged |
| Prices | the last few sessions, appended to the cache; if the overlap disagrees (a split restated the history) the ticker is downloaded in full |
| Split histories | weekly, forty tickers per request, with a direct retry for any ticker missing from its batch |

A cached run of the 49-name watchlist takes about 20 seconds, down from 50.
Funds and trusts are declined on SEC's own entity type before any facts are
fetched; matching "ETF" or "TRUST" in a name both missed funds and caught
REITs.

---

## What the first run said (2026-11-07, config r3)

49 companies, all with filings; margin of safety measurable for 33. The median
measured company trades at **1.46x** its estimated value, and only 3 of the 33
are below their estimate. No company is a *Candidate*.

| Verdict | Count |
|---|---|
| Good business, dear | 21 |
| No | 16 |
| Good business, value unclear | 8 |
| Good business, fair price | 1 (Nike, 0.97x) |
| Statistically cheap only | 1 (Verizon, 0.83x: cheap, but ROIC 11.7% and debt 9.2 years of earnings) |
| Insufficient data | 2 (Berkshire and Visa - multi-class share counts) |

That is what Graham's tests say about large US quality franchises in late 2026:
the businesses pass, the prices do not. It is a statement about these
thresholds at these prices, not a forecast. Re-run under config r4 on
2026-11-08, after the clean-up below, every verdict count came out the same.

Under config r5, after the logic review below: *Good business, dear* 25,
*No* 14, *value unclear* 6, *insufficient data* 2, Nike still the one *fair
price* and Verizon the one *statistically cheap*; margin measurable for 35,
median 1.52x. What moved, and why: Bank of America and Wells Fargo went from
*No* to *dear* (the one-dollar test had failed them on $140bn of buybacks
apiece); Johnson & Johnson and Mastercard from *value unclear* to *dear*
(their DCFs had been inflated by buyback growth until the models were 3x
apart); and U.S. Bancorp's 3% "margin" disappeared once its $7bn of
preferred stock stopped counting as common equity.

## How it was checked

Besides the regression tests, the package went through a maximum-effort code
review - nine independent reviewers, each taking one angle (line-by-line,
documented promises, cross-module contracts, language pitfalls, caches and
fallbacks, reuse, simplification, efficiency, design altitude). Of about 35
distinct issues, the 15 most severe were fixed and are pinned by tests,
including a retry loop that could delete the tracker workbook, failed split
fetches cached as "never split", dividend tagging gaps read as cut dividends,
fiscal years duplicated by one-day date shifts, and a memory footprint that
would not have fit an S&P 500 run.

The lower-severity items followed (config r4), each pinned by a test:

- **Speed:** the append-only store above; panels cached between runs; facts
  refetched on a new report rather than weekly; incremental prices; batched
  splits; the per-fact loops (split factors, unit-scale repair, balance-sheet
  alignment) vectorised and fuzzed against the loops they replaced, with no
  difference except that, of two balance-sheet dates equally far from a
  year end, the later is now taken.
- **One source of truth:** one verdict list feeds the RunLog columns, the
  dashboard and the run summary; one N/A table decides every carve-out (a
  test checks that none of its keys is dead); one list of per-share columns;
  verdict thresholds in `ScoreConfig`.
- **Criterion boilerplate:** each test now declares its name, number and
  threshold once. That fixed drift the repetition had caused: a test reported
  under different names depending on which branch ran, and K1 printed
  "< 1.00x" over a <= comparison. The 1976 tests now follow Graham's wording
  ("less than" is strict). A loss now fails R2 as it fails R1, and a negative
  book value or average EPS fails the combined 6-7 test as it fails 6 or 7.
- **Smaller fixes:** a partial CIK map can no longer overwrite the complete
  one, and a ticker missing from it triggers a refresh; a cover-page share
  count with no current fiscal-year count to agree with must be within 2x of
  the last filed one; other financials (brokers, card lenders, asset
  managers) get the balance-sheet carve-outs banks get; the P/E-high note
  and large non-split share-count changes now reach the caveats; the payload
  dropped the fields the page never read (27% smaller); the daily dashboard
  archive is pruned.

Deliberately not done: reusing the scanner's trade journal and its price
downloader. The valuation prices must not be dividend-adjusted, which the
scanner's are, and the two designs have diverged since; sharing that code
would couple this tool to the scanner's production path for a few dozen
lines. The 4% band for recognising an uncorrected split is also unchanged:
widening it would start reading mergers as splits.

A logic review followed (config r5), looking for reasoning errors rather than
code errors. Each fix was simulated on the watchlist first, and the run
matched the simulation:

- **Buybacks were counted twice in the DCF** - as cash paid out, and again as
  the per-share growth they produce. Growth credited is now the lower of the
  total and the per-share rate (see *Windows are positional*).
- **The one-dollar premise counted buyback cash as retained**, while the
  buybacks shrank the market value it was set against - failing Kimberly-
  Clark, Altria, Kroger, 3M, Nike, Wells Fargo and Bank of America on
  earnings they had paid out. Retained now means net income less dividends
  *and* buybacks. A company that kept less than 10% of what it earned is
  reported as having too little retention to judge: Home Depot kept $0.8bn
  of $130bn, and the ratio read "$245 of market value per dollar retained".
- **Graham's criteria 2 and 7 were each scored as two tests.** Criterion 7's
  P/E x P/B <= 22.5 rule is a second way to meet it, and criterion 2's two
  conditions must both hold; scored separately, a company was charged twice
  for one failure, or half-credited for a failed criterion. Each now counts
  once, so the defensive tally is out of seven, as Graham's is.
- **Utilities were failed on the industrial current ratio.** Graham's test for
  utilities is debt against equity alone.
- **Preferred stock counted as common equity** in book value per share and in
  net current assets; Graham deducts it from both. Bank book values were
  overstated by 6-12% (JPMorgan 6%, U.S. Bancorp 12%).
- **The reverse DCF was described as untunable.** It needs no growth forecast,
  but it moves with the discount rate, so the rate one point either side is
  now reported with it.
- **A rule change would have been logged as company news.** Changes across a
  `CONFIG_VERSION` are now labelled with the rule change.

## Known limits

- **The earnings-multiple model is Graham's criterion-6 price ceiling.** "15 x
  three-year EPS" is the highest price criterion 6 allows, and one of the two
  factors inside the Graham Number, which is excluded from the estimate for
  being a price limit. It is kept as a value model, and it sets the median for
  most companies, so the estimate leans towards no growth: two of the three
  models credit none. Simply treating it as a price limit would leave margins
  measurable for 25 companies instead of 35 and move the median to 2.3x, and
  banks and insurers would be left with one model each unless something (an
  EPS / 9% earnings value, say) took its place. That is a choice about what
  "value" means here, and it has not been made.

- **Twenty years of dividends** cannot be verified from XBRL (it starts around
  2009). An unbroken record is reported as unverifiable, not failed; an
  interruption inside the record is a failure.
- **Multi-class filers** - Visa, Berkshire - tag their share counts only with
  class dimensions this source does not carry, so they get quality tests but
  no per-share valuation. The tracker says so rather than guessing.
- **REITs** are valued on an FFO proxy (net income + D&A), which overstates
  owner earnings for REITs with heavy recurring capex.
- **Maintenance capex** is not disclosed by anyone. Buffett called his own
  figure a guess; the Greenwald estimate and a D&A alternative are both kept.
- **The discount rate (9%) is a required return, not a measurement.** The
  reverse DCF exists so that no conclusion depends on it alone.
- **Cyclicals** defeat any ten-year growth rate. Growth is smoothed with
  three-year medians and capped at revenue growth and 15%, but a business at a
  cyclical peak will still look better than it is.
