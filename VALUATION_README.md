# Valuation Tracker

A fundamental counterpart to the scanner. Where `scanner/` asks whether a price
moves over three days, `valuation/` asks what a business is worth and what it
costs, by the tests Benjamin Graham wrote down in *The Intelligent Investor*
and the standards Warren Buffett describes in the Berkshire letters.

```bash
python run_valuation.py                         # the watchlist: 49 names, then the rest of the S&P 500
python run_valuation.py --tickers KO,AAPL,JPM   # just these
python run_valuation.py --universe sp600        # every S&P 600 constituent instead
python run_valuation.py --open                  # open the dashboard when done
python run_valuation.py --force                 # refetch filings and prices, rebuild every panel
python tests/test_valuation.py                  # regression tests, offline
```

Each run appends to the history store in `valuation_tracker_store/`, refreshes
`valuation_tracker.xlsx` (a view of recent history), and writes
`valuation_dashboard.html`. Edit `valuation_watchlist.txt` to change what is
tracked: a line reading `@sp500` (or `@sp600`) adds every constituent of that
index, and the dashboard's *List* filter separates the names you chose from
the ones the index brought in.

Up to 150 companies, the dashboard is one self-contained file, also archived
in `valuation_reports/` (every day for a month and monthly after that). Above
that, each company's tests and models go into a small file in
`valuation_details/`, loaded when its row is opened, so the page stays quick
with the whole S&P 500 on it. Keep that folder beside the page.

Research tooling, as in the scanner. An intrinsic value is an opinion with
arithmetic attached; nothing here is a recommendation to buy or sell anything.

---

## What it measures

| Instrument | Source | What it asks |
|---|---|---|
| **Graham's seven defensive tests** | *The Intelligent Investor*, ch. 14 | size, financial condition (current ratio and debt vs working capital; for utilities, debt vs equity), 10 years of earnings, 20 years of dividends, 10-year EPS growth, P/E <= 15, P/B <= 1.5 (or P/E x P/B <= 22.5) - seven tests, each counted once |
| **Graham's 1976 ten-point screen** | his last published screen | five reward tests (earnings yield vs AAA, P/E vs its 5-year high, dividend yield, price vs tangible book, price vs NCAV) and five risk tests |
| **Buffett quality proxies** | the Berkshire letters, plus labelled third-party proxies | ROE across the cycle, ROIC, debt repayable from earnings, margin level and stability, cash conversion, dilution, the 1984 one-dollar premise |
| **Intrinsic value** | earnings-based models | two-stage DCF on owner earnings and earnings power value; for banks, insurers and utilities, justified P/B and earnings capitalised at the required return; for REITs, the DCF on funds from operations and the same FFO capitalised without growth |
| **Graham's price limits** | his rules, restated as prices | 15 x 3-year average EPS (criterion 6), the Graham Number, net current assets and two-thirds of them - the most he would pay, shown beside the estimate |
| **What the price assumes** | reverse DCF | the growth today's price needs, beside the growth the record supports |

Every test reports **Pass, Fail or N/A**, and every threshold carries its
provenance - *quoted from Graham*, *quoted from a Berkshire letter*,
*Buffett's standard, our threshold*, *third-party book*, or *our own proxy*.
"Gross margin above 40%" comes from Mary Buffett's *Buffettology*, not from
Warren Buffett, and the dashboard says so.

Graham's 1973 dollar thresholds are inflated by CPI at runtime (x7.6 today), and
his bond-relative tests use the live AAA corporate yield from FRED. His twenty
years of dividends are read from the filings and, before XBRL begins, from
the dividend record in the price history.

---

## Reading the output honestly

**Read "what the price assumes" first.** A forward DCF converts an assumption
into a valuation; the reverse DCF converts today's price into the assumption it
embeds, which can be checked against what the business has actually done. It
needs no growth forecast - but it is not free of inputs. Coca-Cola's price
implies 25% growth a year for ten years at its 9.3% discount rate, 21% at
8.3% and 30% at 10.3%, so the rate one point either side is reported with
it. Coca-Cola's conclusion holds across that range - its revenue has grown
1.5% a year and its owner earnings 7% - so it is a finding. Exxon's does not:
its price implies 2.3% growth at 9.1%, below the 4.8% its record supports,
but 5.7% at 10.1%.

**The discount rate follows the debt.** Everything starts from one required
return on the business, 9% before debt. Owner earnings are struck after
interest, so the DCF discounts them at the return the equity needs, which
rises with leverage; operating income is struck before interest, so the
earnings power value capitalises it at the return the whole company needs,
which tax-deductible debt lowers (Modigliani-Miller, at market values).
Verizon, with debt at 0.84x its market value, is discounted at 10.9% and
capitalised at 8.1%; Apple, with almost none, at 9.0% for both. A flat 9%
treated a heavily indebted company's shares as no riskier than its business.
Banks and insurers borrow as their business, and a utility's regulator sets
its return on its book equity, debt and all, so their rate is not adjusted.

**N/A is not Fail.** A test is N/A when its data does not exist, or when it does
not apply - a current ratio for a bank is a category error. Collapsing the two
into Fail is the commonest way a Graham screen is built wrong: it rejects every
bank, insurer and REIT for reasons Graham never gave.

**The estimate is a range.** The median of the earnings-based models is
reported with its spread. When the highest estimate is more than three times
the lowest ((highest - lowest) / (highest + lowest) > 0.5), or only one model
applies, the margin of safety is **withheld**, and it scores zero - the same
as a measured premium. A margin you cannot demonstrate is not one you have.

**Graham's price limits are not values.** 15 x three-year earnings, the
Graham Number and NCAV are the prices at which his rules would be satisfied.
On an asset-light business at 40x book the Graham Number does not say what
the company is worth; it says Graham would never have bought it. 15 x
earnings is criterion 6 restated as a price, and it credits no growth at
all: while it counted as a value (until config r6), two of the three models
assumed the business would never grow, and it set the estimate for most
companies. They are shown, never averaged in.

**The composite only sorts the list.** Neither author weighted tests into one
number. The verdict is deliberately coarse:

| Verdict | Means |
|---|---|
| Candidate | quality passes, Graham's defensive tests mostly pass, measured margin >= 15% |
| Cheap, quality passes | as above without the Graham side |
| Good business, fair price | quality passes, priced at or up to 15% below the estimate - Buffett's "wonderful company at a fair price" |
| Good business, dear | quality passes, priced above the estimate |
| Good business, value unclear | quality passes, the models disagree too much to call a price |
| Statistically cheap only | measured margin, quality fails - Graham's territory, and the value-trap risk |
| Watch / No | neither |
| Insufficient data / No price | too few tests measurable; or delisted / acquired |

---

## Running on GitHub

`.github/workflows/valuation.yml` runs the tracker at 22:30 UTC on weekdays
(after the US close), on any push that changes the code or the watchlist, and
on demand from the Actions tab ("Run workflow"). GitHub drops and delays
scheduled runs when it is busy - the first evening's never started, and the
first catch-up began 44 minutes late - so a second schedule at 03:30 UTC
catches up. A scheduled run does nothing when the history already holds the
latest close; pushes and manual runs always run in full. Each run:

1. runs the regression tests, and stops if any fail;
2. restores the previous run's downloads (filings, prices, splits and
   dividends) from the Actions cache, so a normal day fetches only what
   changed;
3. runs the tracker against the history store checked out from the `data`
   branch, and commits the new run back to it;
4. posts what moved as a comment on an issue labelled `tracker-alerts`,
   which GitHub notifies whoever watches the repository of (its owner, by
   default). For the names in the watchlist that is any change - a verdict,
   a margin band, a price crossing its estimate or one of Graham's limits;
   for the rest of the index, only a verdict entering or leaving the four
   that price a company at or below its estimate. Companies seen for the
   first time, and changes that a new `CONFIG_VERSION` caused, are left out;
5. publishes the dashboard as the Pages site, with the workbook beside it.

The `data` branch is the record. The cache is only a speed-up: if GitHub
evicts it, the next run downloads everything again and nothing is lost.

The dashboard's *What changed* card lists the last 30 days of changes, rule
changes included and marked as such. A crossing or a margin band move counts
only once it clears 2%, measured from the state last reported: on any day
about twenty companies sit within 2% of a threshold, and ordinary price moves
would report them crossing back and forth. If the figures on the page are
four or more days old - the schedule stopped, or runs are failing - a banner
at the top says so.

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
the S&P 500 and 600 index tables instead, which cover any constituent. SEC's
archive of filed documents refuses any request without a contact, from
anywhere. Visa's and Berkshire's per-class figures are read from it (see
*Known limits*), so until a secret named `VALUATION_USER_AGENT` holds a
contact - for example `ValuationTracker/1.0 you@example.com`; SEC asks
automated clients to identify themselves - those two stay *Insufficient
data*. The same secret covers a ticker outside both indexes, or 403s from
SEC's data endpoints in the run log.

GitHub disables scheduled workflows in a public repository after 60 days with
no activity; if that happens, the Actions tab offers a button to re-enable it.

---

## Why the data layer is most of the code

Fundamentals come from the SEC's XBRL `companyfacts` API rather than a vendor
feed, because Graham's tests need ten years of earnings and twenty of dividends
and the free alternatives serve four. Raw XBRL produces confidently wrong
numbers in at least a dozen distinct ways, each found on a real company and
each now pinned by a test:

| Trap | Seen on | Handling |
|---|---|---|
| The same period filed many times with different values | Apple FY2012 EPS: 44.15 as filed, 6.31 restated | keep the latest-filed value |
| `form=10-K, fp=FY` rows that are quarters | Apple: 90-370-day periods under the same labels | select annual periods by measured length |
| `fy` is the filing's year, not the period's | every company | key on the period end date |
| Per-share history on several share bases at once | Apple pre-2014, post-7:1, post-4:1 | divide each fact by the splits after **its filing date** |
| A spin-off booked as a split | Yahoo's split history: AT&T 1.324 (Warner Bros. Discovery), GE 1.281 and 1.253, 3M 1.196 - AT&T's FY2019 count read 9,729m against 7,348m filed | restate a "split" only if the filer's own restated share counts confirm it, or, with no filing since, if only a split produces its ratio (k-for-1, 1-for-k, 3-for-2); a historical price set against the filings is put back as traded |
| Share counts filed in thousands or millions | NVIDIA FY08-09, Hershey 2009, Berkshire 08-09, McDonald's 2021-25, ConocoPhillips 2010-19, Garmin 2011-16 | repair exact orders of magnitude when two of three references agree: the whole history, the neighbouring years, and the year's own net income / EPS. Where the net income / EPS series runs smoothly and the filed one jumps a thousandfold, it decides alone: a long run of bad years carries both medians (ConocoPhillips: ten of nineteen years) |
| A share count of zero | Constellation Energy before its 2022 spin-off; Ares before it became a corporation | read as no count; kept, it was a jump of "infx" that cut the per-share history |
| A merger or listing shaped like a split | Cigna doubling its shares to buy Express Scripts; United and Continental; Datadog's IPO | a split-shaped jump stands as a split only with a split on record nearby or its one-day gap in the prices; revenue that grew with the shares, or prices that begin that year, make it an event, and the history is kept |
| Revenue that is only part of the total | American Tower: $0.8bn of contract revenue beside $10.1bn of Revenues; Charter's $0.9bn of about $55bn; banks' fees alone | the largest revenue filed for the year - ASC 606 contract revenue leaves out leases, premiums and interest - but never the figure gross of excise taxes; for a bank, net interest income plus fees |
| A total filed a thousandfold too large | Coherent FY2023 D&A: $681.7bn, against $267.6m + $414.1m of parts | the same vote, with depreciation + amortisation as the year's own figure |
| Preferred stock filed at par | American Express: $1.6bn of preferred as 1,600 shares of $1m, and a value of zero | shares x liquidation price |
| A new registrant for an old company | ExxonMobil's 2026 Texas holding company, with one quarterly report; Texas Pacific Land, a trust until 2021; BlackRock, Bunge, Ferguson and Paramount, 2023-25 | the predecessor's filings merged in, for the companies `CIK_SUCCESSIONS` names - EDGAR does not link the two - up to the successor's first annual report, so a predecessor still filing as a subsidiary (Apache, Baker Hughes Holdings) cannot outvote it |
| A ticker that changed hands | SEC's ticker.txt, unchanged since September 2022: BNY is a BlackRock municipal bond fund there, DOC is Physicians Realty, and GE HealthCare is missing | the index table's CIK for its own constituents |
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
`changes` are append-only parquet files, and `latest` is replaced each run.
A run is dated by the market close it priced, not the machine's calendar, so
a second run for the same close replaces the first one's rows. A
run appends a few kilobytes and rewrites nothing already there. The workbook
is a view regenerated from the store, with `History` and `Changes` cut to
recent runs - 400 days for the 49-name watchlist, 100 for the S&P 500, to
stay under 50,000 history rows - so its cost no longer grows with every day
the tracker runs.

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
| Split and dividend histories | weekly, forty tickers per request, with a direct retry for any ticker missing from its batch; dividends back to 1995 |
| Index constituents (`@sp500`) | weekly, from the index tables on Wikipedia |
| Ticker to CIK map | every 14 days: SEC's ticker.txt, with the index tables' CIK winning for their own constituents |
| Per-class figures (Visa, Berkshire) | each annual report once, from its XBRL in SEC's archive - which needs a contact (above) |

A cached run of the 49-name watchlist takes about 20 seconds, down from 50.
On GitHub's runners the first S&P 500 run took 7 minutes, downloading every
filing; with the filings cached, a run that rebuilt every panel took under 2.
Funds and trusts are declined on SEC's own entity type before any facts are
fetched; matching "ETF" or "TRUST" in a name both missed funds and caught
REITs.

---

## What the runs said

### The watchlist, from 2026-11-07

The first run (config r3): 49 companies, all with filings; margin of safety
measurable for 33. The median measured company trades at **1.46x** its
estimated value, and only 3 of the 33 are below their estimate. No company
is a *Candidate*.

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

Under config r6 (2026-11-09) - 15 x earnings moved to the price limits,
discount rates that follow leverage, and the dividend record: *Good
business, dear* 28, *No* 15, *value unclear* 4, *insufficient data* 2;
margin measurable for 36, median 1.78x, and none of the 36 below its
estimate. What moved, and why:

- Nike (*fair price* to *dear*) and Verizon (*statistically cheap* to *No*)
  had both been valued at 15 x earnings, which set their medians. Without
  it, Nike's estimate is $38.49 against a price of $38.59. Verizon's DCF
  also discounts at 10.9% now, its debt being 0.84x its market value.
- Merck, 3M and Caterpillar went from *value unclear* to *dear*: the new
  rates lowered their DCFs and raised their EPVs enough to bring the two
  within 3x of each other. Deere, Southern and Duke went the other way, each
  left with one model that can value it, and NextEra with none - its capital
  spending leaves no owner earnings to discount, and its net debt exceeds
  its earnings power value.
- Graham's twenty-year dividend test can now pass, and 40 of the 49 did.
  Apple, Alphabet, NVIDIA, Cisco and Visa have paid for less than twenty
  years, Intel paid nothing in 2025 and Berkshire pays nothing; AbbVie, a
  2013 spin-off, cannot be checked.

Config r7 changed no watchlist verdict; Mastercard's dividend record now
passes at twenty years, the 41st.

### The S&P 500, from 2026-11-09 (config r7)

503 tickers, 501 evaluated: HONA, newly listed, has filed no annual report
yet, and VYLR is declined on SEC's entity type. The margin of safety is
measurable for 260. The median measured company trades at **1.84x** its
estimate; 22 are below it, and 7 below two-thirds of it - Graham's margin.

| Verdict | Count |
|---|---|
| Candidate | 6 (Lennar, D.R. Horton, T. Rowe Price, Synchrony, Cognizant, Deckers) |
| Cheap, quality passes | 3 (CF Industries, HP, PayPal) |
| Good business, fair price | 4 (General Mills, Arch Capital, Everest, Skyworks) |
| Statistically cheap only | 5 (MGM Resorts, Comcast, Aptiv, Charter, LyondellBasell) |
| Good business, dear | 163 |
| Good business, value unclear | 77 |
| No | 231 |
| Insufficient data | 12 (new listings and spin-offs with a few years on file; multi-class filers) |

Read the candidates with *Known limits* in mind. Two are homebuilders,
valued on earnings from a housing boom - the cyclical case a ten-year record
flatters most - and each is a statement about these thresholds at these
prices, not a recommendation.

The margin is withheld for 208 companies: 108 have one model that can value
them, and for 96 the DCF and the EPV are more than 3x apart. 261 of the 501
have a verified record of twenty years of dividends or more.

Under config r8, on Monday's closes, the margin is measurable for **313**
(from 260) - all 27 REITs and 26 of the 37 utilities for the first time -
and the median is still 1.84x, with 28 below their estimate and 8 below
two-thirds of it. *Candidate* 5 (Deckers slipped to *fair price* on the
day's prices), *cheap, quality passes* 3, *fair price* 6 (Boston
Properties joins), *statistically cheap only* 7 (Alexandria and Camden
join), *dear* 162, *value unclear* 73, *No* 235, *insufficient data* 10.
Of the 166 margins withheld, 60 have one model and 106 have models more
than 3x apart.

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

The review's suggested improvements followed (config r6), with the open
question from r5 decided:

- **15 x earnings is a price limit**, beside the Graham Number (see *Reading
  the output honestly*). Banks and insurers, which would have been left with
  one model, gained an earnings value: three-year average EPS divided by the
  required return, crediting no growth.
- **Discount rates follow leverage** - higher for the equity, lower for the
  whole company - instead of one flat 9%.
- **Graham's twenty years of dividends are measured**, from the filings and
  the dividend record in the price history before them. The test could not
  pass before; on the watchlist, 41 of 49 now do.
- **The one-dollar test compares like with like.** The rise in market value
  ran ten calendar years back from today, up to a year adrift of the ten
  fiscal years of retained earnings it was set against; it now runs over
  the same fiscal years, at year-end prices and year-end share counts.
- **Book value per share uses the year-end share count**, not the year's
  weighted average, which lags a buyback programme and overstates the count.
- **Preferred stock filed as shares and a liquidation price** - American
  Express files $1.6bn of it at a par value of zero - is now deducted.
- **Visa's and Berkshire's per-class figures** can be read from each annual
  report's XBRL, when SEC's archive allows it.
- **The whole S&P 500** is tracked, with each company's details loaded on
  demand; the page lists what changed, GitHub posts it as a notification,
  and a banner says when the figures are old.

Config r7 followed the same day, after the first S&P 500 run surfaced data
errors no watchlist company had, and a gap in the dividend test:

- **Share counts mis-scaled across a run of years.** McDonald's filed its
  2021-25 counts in millions; W. R. Berkley (2017-21), Garmin (2011-16) and
  ConocoPhillips (2010-19) filed theirs in thousands. The repair compared
  each year with its neighbours, which were wrong too, so it fixed the edges
  of each run at most - and for ConocoPhillips, whose bad years were the
  majority, "repaired" the one correct year beside them. McDonald's and
  W. R. Berkley went to *Insufficient data*. Net income / EPS now has a vote,
  and the deciding one where its own series is smooth (see the table above).
- **A share count of zero** - filed for the years before a spin-off, an IPO
  or a conversion to a corporation - was read as a count, and cut the
  per-share history of Constellation Energy, Ares and Diamondback.
- **Texas Pacific Land** restarted its history in 2021, when the trust
  became a corporation under a new registration; the trust's filings are
  now merged in, as ExxonMobil's are.
- **A D&A total filed a thousandfold too large** gave Coherent $136bn a year
  of owner earnings and a DCF of $15,434 a share, against a price of $193.
  The dispersion guard withheld the margin, but the estimate was still on the
  page.
- **The dividend record skipped the year a company listed**, even when it
  paid in it: Mastercard listed in May 2006 and paid that August, and was
  reported as unverifiable at 19 years instead of passing at 20.
- Two notes said the opposite of their verdict: Microsoft's dividend test
  passed (24 years) with a note calling it refuted, and a value lost to a
  rule change was blamed on "the latest filings".

Config r8 (2026-11-10) followed a look at where the S&P 500 results fell
short. Each change was checked on all ~500 companies before it shipped:

- **Revenue was a fraction of the total for 22 companies.** The chain read
  ASC 606 contract revenue first, which leaves out leases, premiums and
  interest: American Tower, Crown Castle and the apartment REITs, MetLife,
  Humana and Hartford, Capital One and the regional banks, United Rentals,
  ADM and Bunge, and Charter - one of the five *statistically cheap* - with
  $0.9bn of revenue against $12.9bn of operating income. Another figure now
  takes over only at twice the first; simply taking the largest read
  Hasbro's and Republic Services' totals before eliminations.
- **REITs and utilities can have a margin of safety.** None of 27 REITs or
  37 utilities could before; all 27 REITs and 26 of the utilities now do,
  REITs at a median 1.32x their estimate and utilities at 2.10x. Most still
  score *No* on Buffett's quality proxies, as before.
- **Mergers and listings are no longer read as missed splits.** Of the
  split-shaped breaks, Cigna, Global Payments, Evergy, United, Viatris,
  Republic Services, Keurig Dr Pepper, Workday, Palantir and eight listings
  (Datadog, Robinhood, Moderna, Coinbase, Bloom Energy, Twilio, GoDaddy,
  Dell) keep their per-share history, with the move reported as an event.
  Ten are still cut: T-Mobile and WTW have a split on record beside the
  jump, and for Abbott, Blackstone, Carvana, Pentair, Kinder Morgan, Live
  Nation, Johnson Controls and Targa nothing else explains it.
- **The daily run** has a catch-up schedule, and runs are dated by the
  close they priced (see *Running on GitHub*).
- **Alerts** cover the watchlist in full and the rest of the index only
  when a verdict enters or leaves the four priced at or below value; and a
  crossing or band move must clear 2% from the state last reported.

Config r9 (2026-11-10) fixed two data errors that the point-in-time backtest
found in the live tracker:

- **Spin-offs were restated as splits.** Yahoo books a spin-off as a
  "split" of the price ratio across it - ninety-odd in the S&P 500's
  histories since 2008, four of them folded into a reverse split (Hilton,
  DuPont, Leidos, Motorola) - and every share count and EPS filed before one
  was restated as if the shares had split: AT&T's FY2019 count read 9,729m
  against 7,348m filed, GE's FY2014-21 counts were 1.25-1.6x too high, 3M's
  FY2021 EPS was 8.46 against 10.12. A "split" is now restated only if the
  filer's own restated share counts confirm it, or, with no report since,
  if only a split produces its ratio. Every split with a split's ratio still
  is, and so are Brown-Forman's 5-for-4 and Willis's 2.649-for-1
  consolidation. A historical price set against the filings - the five-year
  P/E high, the one-dollar test's market values - is taken as traded. Of
  the 67 S&P 500 companies with a spin-off on record, ten-year EPS growth
  moved for 36 and the share-count trend for 31; Abbott, Pentair and
  Johnson Controls, three of r8's breaks that nothing explained, keep their
  per-share history. No verdict moved.
- **Tickers that changed hands were read from the wrong filer.** SEC's
  ticker.txt has not changed since September 2022: locally, BNY was a
  BlackRock municipal bond fund, DOC was Physicians Realty, BlackRock,
  Bunge and Ferguson were their old registrations, and 24 newer
  constituents (GE HealthCare, Solventum, Veralto, Cencora...) had no CIK.
  The index tables' CIK now wins for their constituents - GitHub's runs,
  which cannot reach SEC's file, already used them. BlackRock, Bunge,
  Ferguson and Paramount Skydance carry their predecessors' history, each
  predecessor cut at its successor's first annual report so that one still
  filing as a subsidiary cannot outvote it.

On the cached S&P 500 filings and the same day's prices, run locally, 478
companies were evaluated before and 501 after. Of the 478, one verdict moved:
Ferguson, from *insufficient data* to *good business, dear*, with six years
on file instead of four. Healthpeak, valued on Physicians Realty's filings
before, has a measured margin (a premium); BlackRock and Bunge are valued on
filings to 2025 instead of 2023 and 2022. Of the 23 newly evaluated, five
are *good business, dear*, one *fair price* (Everest), three *value
unclear*, three *insufficient data* and eleven *No*; HONA (no annual report)
and VYLR (declined) remain.

## Known limits

- **Two value models, not three.** With 15 x earnings moved to the price
  limits, an operating company's estimate is the median of a DCF and an EPV,
  and the margin is withheld when they are more than 3x apart - which, for
  the S&P 500, is often (see above). That is the honest result: the models
  disagree, and averaging in a third that assumes no growth would only hide
  it. Banks, insurers and utilities have justified P/B and an earnings value
  as well; REITs, capitalised FFO beside the DCF.
- **Utilities at 9%.** A utility's earnings and book are valued at the same
  required return as any business's. Investors have long accepted less from
  regulated returns, so at 9% most utilities read as dear - a median 2.1x
  their estimate; the reverse DCF shows the growth their prices assume at
  that rate.
- **Twenty years of dividends** come from the filings and, before them, the
  dividend record in Yahoo's price history, which begins at listing. A year
  the record covers only in part counts as paid if a dividend went ex in it,
  and as unknown otherwise, so a company listed less than twenty years ago
  that has paid ever since is reported as unverifiable, not failed. A year
  with no dividend inside the record is a failure.
- **Multi-class filers** - Visa, Berkshire, Constellation Brands - tag their
  share counts and EPS only with class dimensions that `companyfacts` does
  not carry. For Visa and Berkshire, `valuation/classes.py` reads the
  per-class figures from each annual report's own XBRL instead; that needs
  SEC's archive, and so a contact (see *Running on GitHub*). Without one,
  they get quality tests but no per-share valuation, and the tracker says so
  rather than guessing.
- **REITs** are valued on an FFO proxy (net income + D&A), which overstates
  owner earnings for REITs with heavy recurring capex.
- **Maintenance capex** is not disclosed by anyone. Buffett called his own
  figure a guess; the Greenwald estimate and a D&A alternative are both kept.
- **The discount rate (9% before debt) is a required return, not a
  measurement**, and the leverage adjustment rests on book interest rates
  and today's market values. The reverse DCF exists so that no conclusion
  depends on it alone.
- **Cyclicals** defeat any ten-year growth rate. Growth is smoothed with
  three-year medians and capped at revenue growth and 15%, but a business at a
  cyclical peak will still look better than it is.
- **Tickers outside the two indexes** are resolved from SEC's ticker.txt,
  which stopped changing in September 2022: a watchlist name that changed
  ticker or listed since then is missing or points at the old holder, until
  it is written into `valuation_state/cik_overrides.json`.
- **A "split" newer than the latest report** is judged by its ratio alone
  until the next report's share counts settle it: a spin-off with a ratio
  only a split produces (k-for-1, 1-for-k, 3-for-2) would be restated, and a
  5-for-4 split would not. None of the ninety-odd spin-offs in the S&P 500's
  split histories since 2008 has such a ratio.
