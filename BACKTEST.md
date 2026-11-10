# Backtest: did the verdicts predict anything?

The tracker's verdicts and price-to-value ratios, as it would have given them
at each year-end from 2014 to 2023, set against the total returns that
followed. Run on 2026-11-10 with config r8; returns to 2026-11-09.

> **Survivorship bias - read this first.** The universe is *today's* S&P 500.
> Every company in it survived to 2026, and many joined the index only after
> the year they are tested in, because they had done well. At the end of
> 2014, 142 of the 455 companies priced were not yet in the index, and 186 of
> that day's 499 members are not in today's list. Today's constituents
> returned **17.1% a year** over the one-year windows that followed; the
> index as it actually was (the S&P 500 Equal Weight ETF, RSP) returned
> **10.8%**. Every absolute return below is overstated by about six points a
> year, and the relative results can be bent too: the companies that went
> bankrupt, were taken over or shrank out of the index are missing, and they
> were not a random sample. See [Survivorship](#survivorship).

## What it found

- **The tracker's buy list lagged.** *Candidate* and *Cheap, quality
  passes* together - about nine companies a year - trailed the
  equal-weighted universe by 7.6% a year over one-year windows (ahead in 3
  of 10 years; t = -2.0), by 9.1% a year over three years and 7.6% over
  five, behind in every one of those 16 overlapping windows. Among companies
  that were in the index at the time, *Candidate* trailed by 2.6% a year.
- **The worst verdict was *Cheap, quality passes*:** -19.1% a year (t =
  -2.6) - but it held fewer than two companies a year, and eight of its 19
  company-years were Gilead and Biogen, as their franchises eroded.
- **Price-to-value ranked nothing.** The cheapest fifth by measured
  price-to-value did no better than the dearest (-1.4% a year over one year,
  -0.7% over three, -1.6% over five); the rank correlation with the next
  year's return was -0.02. Including the margins the tracker withholds makes
  no difference.
- **This was a bad decade for any value sort.** Ranking the same companies
  by plain earnings yield did about as badly over one year (-1.9% a year,
  cheapest minus dearest) and worse over five (-4.2%); across the whole
  universe, -11.9%. The tracker's measure lost less than a P/E sort over the
  longer windows, but nothing here shows it finding anything.
- **The best verdict was *Good business, value unclear*** - the businesses
  whose models disagree by more than 3x, mostly growth companies whose DCF
  runs far above their earnings power - at +3.0% a year (t = 1.7), and
  ahead in all seven five-year windows.

The buy list's members explain most of it. Asset managers in secular decline
(Franklin Resources in seven of ten years, Invesco, T. Rowe Price),
biotechs past their peak (Gilead, Biogen), refiners and chemical and
fertiliser makers - the cyclicals a ten-year record flatters - airlines and
a cruise line, Paramount through its streaming losses, legacy hardware (IBM,
HP, Western Digital): businesses with ten good years behind them, priced
for decline, and mostly declining. The quality tests read the trailing
decade, and a business whose economics are deteriorating passes them until
well after its price has fallen. Meta in 2022 (+168% the next year) and
Intel in 2022 were the exceptions.

Ten annual observations are few. With nine degrees of freedom a t-statistic
needs to pass 2.26 for the usual 5% test; only *Cheap, quality passes*, on
19 company-years, and *Insufficient data* - a survivorship artefact, below -
do. Three- and five-year windows overlap and carry no t-statistic. This is
one decade, in which growth beat value; it is evidence that the verdicts did
not help over it, not proof that they cannot.

## How it was run

`python backtest_valuation.py` - the tracker's own code, at today's
configuration, at each year-end from 2014 to 2023:

- **Filings as they stood that day.** `fundamentals.build_panel(as_of=day)`
  drops every XBRL fact filed after the day before anything is parsed -
  later restatements, later fiscal years, later cover-page share counts - and
  judges the filing's age and the current share count from the day. Apple's
  FY2012 EPS is 44.15 as of end-2013 and the restated 6.31 (on the split
  basis of the time) a year later. `tests/test_valuation.py` rewrites every
  fact filed after the day, adds later years and tags, and checks that the
  panel does not move; it runs on every tracker run.
- **A company that has since moved to a new registration** - Alphabet
  (Google Inc. until 2015), Disney (2019), Cigna (2018), Linde (Praxair),
  Medtronic (2015), Broadcom (Avago), Paramount Skydance (CBS) and a dozen
  more - is read from the registration that was filing then. Each
  predecessor counts until its successor files its first annual report;
  several kept filing as subsidiaries for years.
- **Prices as of the day:** the year-end close, with the twelve years of
  closes before it that the tracker downloads, and the dividend record cut
  at the day. Config r8's check on a split-shaped break in the share count
  reads that price record and the splits on record, and both stop at the
  day. **Macro as of the day:** the AAA yield of the last month completed,
  the CPI of the last month published (November's, both, at a December
  year-end).
- **Returns:** total returns from the year-end close over one, three and
  five years, from Yahoo's closes and dividends, each dividend reinvested at
  its ex-date close (the CRSP convention). They match Yahoo's adjusted
  closes to within 0.03% a year for the median company. Yahoo books a
  spin-off as a fractional split, which keeps the return whole; for
  Danaher's 2016 Fortive spin-off it also booked a dividend, counted once
  here (Yahoo's adjusted close counts it twice).
- **Comparisons:** each year, each verdict group is an equal-weighted
  portfolio, set against the equal-weighted universe - every company priced
  that day, whatever its verdict. Excess returns are annualised, then
  averaged across years; *ahead* counts the years the group beat the
  universe. Alphabet's, Fox's and News Corp's two share classes count once:
  503 tickers, 500 companies.

## Results

### Verdicts per year

| Year-end | Priced | Verdict given | Median years on file | Margin measured | Median P/V | Candidate or cheap + quality | Fair price | Statistically cheap | Dear | Value unclear | No / watch | No verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2014 | 455 | 384 | 7 | 290 | 1.94x | 4 | 5 | 3 | 136 | 32 | 204 | 71 |
| 2015 | 462 | 397 | 8 | 305 | 1.77x | 18 | 5 | 3 | 132 | 34 | 205 | 65 |
| 2016 | 467 | 414 | 9 | 301 | 1.84x | 7 | 5 | 3 | 138 | 36 | 225 | 53 |
| 2017 | 471 | 428 | 10 | 297 | 2.05x | 4 | 3 | 1 | 144 | 36 | 240 | 43 |
| 2018 | 475 | 443 | 11 | 296 | 1.71x | 10 | 8 | 10 | 144 | 46 | 225 | 32 |
| 2019 | 480 | 454 | 12 | 294 | 1.94x | 6 | 5 | 6 | 151 | 53 | 233 | 26 |
| 2020 | 485 | 456 | 13 | 306 | 2.07x | 16 | 3 | 7 | 156 | 51 | 223 | 29 |
| 2021 | 489 | 463 | 14 | 305 | 2.40x | 3 | 5 | 2 | 166 | 52 | 235 | 26 |
| 2022 | 491 | 470 | 15 | 290 | 1.79x | 15 | 9 | 9 | 155 | 60 | 222 | 21 |
| 2023 | 492 | 477 | 16 | 292 | 1.75x | 9 | 8 | 13 | 147 | 60 | 240 | 15 |

XBRL begins in 2009, so in 2014 the median company had seven fiscal years on
file and Graham's ten-year tests were mostly unmeasurable; *No verdict*
(insufficient data, or no annual report yet) falls from 71 to 15 as the
record lengthens. The market looked dearest to the tracker at the end of
2021 (median 2.40x its estimate).

### Verdict groups against the equal-weighted universe

| Verdict | Company-years | Per year | 1y return | 1y excess | 1y ahead | t (1y) | 3y excess | 3y ahead | 5y excess | 5y ahead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Candidate | 73 | 7.3 | 11.1% | -6.1% | 3/10 | -1.7 | -8.0% | 0/9 | -6.6% | 1/7 |
| Cheap, quality passes | 19 | 1.9 | -2.0% | -19.1% | 1/10 | -2.6 | -13.9% | 0/9 | -10.3% | 0/7 |
| Good business, fair price | 56 | 5.6 | 16.4% | -0.8% | 4/10 | -0.1 | -2.7% | 4/9 | -1.8% | 3/7 |
| Statistically cheap only | 57 | 5.7 | 18.8% | +1.7% | 6/10 | +0.4 | -1.9% | 4/9 | -2.5% | 2/7 |
| Good business, dear | 1469 | 146.9 | 15.1% | -2.0% | 4/10 | -1.4 | -1.9% | 3/9 | -0.3% | 4/7 |
| Good business, value unclear | 460 | 46.0 | 20.2% | +3.0% | 6/10 | +1.7 | +2.8% | 7/9 | +4.1% | 7/7 |
| No | 2251 | 225.1 | 16.7% | -0.5% | 5/10 | -0.4 | -0.8% | 3/9 | -1.2% | 1/7 |
| Insufficient data | 302 | 30.2 | 32.0% | +14.9% | 9/10 | +2.3 | +11.9% | 9/9 | +3.8% | 7/7 |
| No annual report yet | 79 | 7.9 | 22.1% | +5.0% | 6/10 | +0.5 | +9.4% | 8/9 | +5.2% | 5/7 |

The universe: 477 companies a year; 17.1% a year over one-year windows (10
of them), 17.4% over three-year (9), 16.8% over five-year (7). One *Watch*
is left out. *Insufficient data*'s lead is survivorship, not signal: it is
mostly young companies with short filing records that later grew into the
index, and among companies in the index at the time it is +1.2% a year.

### The tracker's buy list

| Group | Company-years | Per year | 1y return | 1y excess | 1y ahead | t (1y) | 3y excess | 3y ahead | 5y excess | 5y ahead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Candidate or cheap, quality passes | 92 | 9.2 | 9.5% | -7.6% | 3/10 | -2.0 | -9.1% | 0/9 | -7.6% | 0/7 |
| Statistically cheap only | 57 | 5.7 | 18.8% | +1.7% | 6/10 | +0.4 | -1.9% | 4/9 | -2.5% | 2/7 |
| Everything else | 4618 | 461.8 | 17.3% | +0.1% | 6/10 | +1.0 | +0.1% | 8/9 | +0.1% | 7/7 |

### Price-to-value

Within each year-end, the companies whose margin of safety the tracker
measured (about 300 a year: the models agree within 3x) are ranked into
fifths by price-to-value.

| Quintile | Median P/V | Per year | 1y return | 1y excess | 1y ahead | t (1y) | 3y excess | 3y ahead | 5y excess | 5y ahead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (cheapest) | 1.03x | 59.9 | 14.3% | -2.9% | 3/10 | -1.1 | -3.0% | 1/9 | -2.4% | 1/7 |
| 2 | 1.49x | 59.4 | 13.2% | -3.9% | 3/10 | -1.8 | -2.7% | 3/9 | -2.3% | 1/7 |
| 3 | 1.91x | 59.3 | 13.9% | -3.2% | 3/10 | -2.1 | -3.3% | 1/9 | -1.1% | 2/7 |
| 4 | 2.41x | 59.4 | 13.2% | -4.0% | 2/10 | -2.5 | -2.6% | 1/9 | -1.4% | 2/7 |
| 5 (dearest) | 3.61x | 59.6 | 15.7% | -1.5% | 5/10 | -0.8 | -2.3% | 5/9 | -0.7% | 3/7 |

Every quintile trails the universe because the companies the tracker can
measure (14.0% a year) trail the ones it cannot - young and fast-growing
companies, whose models disagree. The question is the slope across the
quintiles, and there is none - or a slightly wrong-way one:

| Ranking | Ranked per year | 1y cheapest - dearest | t | 1y rank corr. | t | 3y cheapest - dearest | 3y rank corr. | 5y cheapest - dearest | 5y rank corr. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Price-to-value, margin measured | 298 | -1.4% | -0.4 | -0.016 | -0.3 | -0.7% | -0.046 | -1.6% | -0.068 |
| Price-to-value, every estimate | 440 | -1.9% | -0.5 | -0.016 | -0.3 | -1.0% | -0.039 | -2.6% | -0.056 |
| Earnings yield, the same companies | 297 | -1.9% | -0.5 | -0.020 | -0.3 | -2.5% | -0.054 | -4.2% | -0.101 |
| Earnings yield, every company with 3 years of EPS | 459 | -11.9% | -1.9 | -0.054 | -0.9 | -10.6% | -0.092 | -6.9% | -0.136 |

The rank correlation is oriented so that a positive number means the
cheaper did better. Earnings yield is three-year average EPS over the price.

### Survivorship

Who was in the index at each year-end comes from a point-in-time list of
its members ([fja05680/sp500](https://github.com/fja05680/sp500)), and when
each of today's constituents last joined it from the index table's *date
added*, which follows a company through a change of ticker (Facebook to
Meta).

| Year-end | Index members then | Today's constituents priced | ... in the index then | ... joined later | Members since gone | Next year: universe | ... in index then | ... joined later | RSP | SPY |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2014 | 499 | 455 | 313 | 142 | 186 | 3.9% | 2.2% | 7.6% | -2.7% | 1.3% |
| 2015 | 502 | 462 | 326 | 136 | 176 | 18.5% | 16.6% | 23.1% | 14.5% | 12.0% |
| 2016 | 506 | 467 | 347 | 120 | 159 | 26.8% | 24.4% | 33.7% | 18.5% | 21.7% |
| 2017 | 505 | 471 | 369 | 102 | 136 | -1.4% | -4.5% | 9.9% | -7.8% | -4.6% |
| 2018 | 505 | 475 | 382 | 93 | 123 | 34.0% | 31.7% | 43.2% | 28.9% | 31.2% |
| 2019 | 505 | 480 | 401 | 79 | 104 | 20.8% | 12.0% | 65.0% | 12.7% | 18.4% |
| 2020 | 505 | 485 | 411 | 74 | 94 | 31.1% | 30.8% | 32.5% | 29.4% | 28.7% |
| 2021 | 505 | 489 | 421 | 68 | 84 | -9.5% | -8.8% | -13.4% | -11.6% | -18.2% |
| 2022 | 503 | 491 | 435 | 56 | 68 | 25.7% | 18.0% | 84.8% | 13.7% | 26.2% |
| 2023 | 503 | 492 | 448 | 44 | 55 | 21.5% | 14.8% | 90.1% | 12.8% | 24.9% |

| Horizon | Universe | In index then | Joined later | RSP | SPY | Universe - RSP | In index then - RSP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1y (10 windows) | 17.1% | 13.7% | 37.6% | 10.8% | 14.2% | +6.3% | +2.9% |
| 3y (9 windows) | 17.4% | 13.7% | 32.1% | 10.9% | 13.9% | +6.5% | +2.8% |
| 5y (7 windows) | 16.8% | 14.1% | 24.7% | 11.6% | 14.1% | +5.3% | +2.5% |

So the universe overstates the equal-weighted index by about 6.3 points a
year. About 3.4 points come from companies that joined the index later -
which returned 37.6% a year in the windows before they did - and about 2.9
from the rest: today's survivors among that day's members, against RSP,
which held the ones that later left too (and charges 0.2% a year, and
rebalances quarterly).

The first part can be taken out, and the tables below do. The second
cannot: Yahoo keeps no history for a delisted ticker, so the members that
were acquired, merged away or went bankrupt cannot be priced. Chesapeake, a
member until 2018 whose equity was wiped out in 2020, appears here only as
Expand Energy, from 2021. Their absence most likely flatters the cheap
verdicts, where value traps and bankruptcies concentrate - though takeovers
at a premium cut the other way.

### Companies in the index at the time only

| Verdict | Company-years | Per year | 1y return | 1y excess | 1y ahead | t (1y) | 3y excess | 3y ahead | 5y excess | 5y ahead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Candidate | 69 | 6.9 | 11.2% | -2.6% | 3/10 | -0.7 | -4.8% | 1/9 | -4.7% | 1/7 |
| Cheap, quality passes | 18 | 1.8 | -1.6% | -15.3% | 1/10 | -2.3 | -9.9% | 1/9 | -7.5% | 1/7 |
| Good business, fair price | 53 | 5.3 | 15.8% | +2.1% | 3/10 | +0.4 | +0.2% | 5/9 | -0.1% | 4/7 |
| Statistically cheap only | 54 | 5.4 | 18.6% | +4.8% | 6/10 | +1.2 | +0.6% | 4/9 | -2.4% | 4/7 |
| Good business, dear | 1316 | 131.6 | 14.2% | +0.5% | 5/10 | +0.4 | +0.9% | 6/9 | +1.2% | 5/7 |
| Good business, value unclear | 377 | 37.7 | 17.0% | +3.3% | 7/10 | +1.4 | +3.5% | 8/9 | +3.9% | 7/7 |
| No | 1807 | 180.7 | 12.6% | -1.2% | 4/10 | -1.3 | -1.5% | 2/9 | -1.8% | 0/7 |
| Insufficient data | 151 | 15.1 | 15.0% | +1.2% | 5/10 | +0.6 | +0.4% | 5/9 | +2.4% | 5/7 |

Universe: 385 companies a year, 13.7% a year over one year. Price-to-value
quintiles on this universe: cheapest minus dearest -0.9% a year over one
year (t = -0.2), -1.1% over three, -2.3% over five; rank correlation -0.01.
The conclusions do not change.

## What else is not point in time

- **The configuration.** Config r8 was written in 2026. Its thresholds were
  set by reading Graham and Buffett and its data repairs by finding XBRL
  traps, not by fitting returns - but a 2014 version of the tracker would not
  have known about the traps the later ones fixed.
- **Yahoo's histories.** Prices exist only for tickers that still trade;
  dividends and splits are as Yahoo records them. Filer metadata - SIC code,
  entity type - is today's.
- **Spin-offs restate the tracker's share counts.** The tracker passes
  Yahoo's fractional "splits" for spin-offs to its per-share restatement of
  XBRL facts, so before a spin-off GE's, 3M's and AT&T's share counts read
  1.2-1.6x too high (and EPS that much too low). The backtest reproduces the
  tracker as it is; the bug is reported separately.
- **DuPont** (a merger of equals in 2017) has no single predecessor whose
  filings match its price history, and has no verdict before 2019.

## Reproduce

```bash
python backtest_valuation.py                   # ~15 minutes; downloads prices and dividends once
python backtest_valuation.py --analyse-only    # redo the tables from the saved rows
python tests/test_backtest.py                  # its tests, offline
```

Everything it downloads and writes stays in `data_cache/backtest/`, which is
not published: `backtest_rows.csv` holds every company's verdict, price,
price-to-value and forward returns at every year-end, and
`backtest_tables.md` the tables above.
