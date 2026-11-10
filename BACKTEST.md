# Backtest: did the verdicts predict anything?

The tracker's verdicts and price-to-value ratios, as it would have given them
at each year-end from 2014 to 2023, set against the total returns that
followed. Run on 2026-11-10 with config r7; returns to 2026-11-09.

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
  equal-weighted universe by 7.8% a year over one-year windows (ahead in 3
  of 10 years; t = -2.0), by 9.1% a year over three years and 7.9% over
  five, behind in every one of those 16 overlapping windows. Among companies
  that were in the index at the time, *Candidate* trailed by 2.6% a year.
- **The worst verdict was *Cheap, quality passes*:** -19.1% a year (t =
  -2.6) - but it held fewer than two companies a year, and eight of its 19
  company-years were Gilead and Biogen, as their franchises eroded.
- **Price-to-value ranked nothing.** The cheapest fifth by measured
  price-to-value did no better than the dearest (-2.6% a year over one year,
  -1.3% over three, -1.5% over five); the rank correlation with the next
  year's return was -0.04. Including the margins the tracker withholds makes
  no difference.
- **This was a bad decade for any value sort.** Ranking the same companies
  by plain earnings yield did worse (-4.6% a year, cheapest minus dearest),
  and across the whole universe -10.8%. The tracker's measure lost less than
  a P/E sort, but nothing here shows it finding anything.
- **The best verdict was *Good business, value unclear*** - the businesses
  whose models disagree by more than 3x, mostly growth companies whose DCF
  runs far above their earnings power - at +2.9% a year (t = 1.9), and
  ahead in all seven five-year windows.

The buy list's members explain most of it. Asset managers in secular decline
(Franklin Resources in seven of ten years, Invesco, T. Rowe Price),
biotechs past their peak (Gilead, Biogen), refiners and chemical and
fertiliser makers - the cyclicals a ten-year record flatters - airlines and
a cruise line, Paramount through its streaming losses, legacy hardware (IBM,
HP, Western Digital): businesses with ten good years behind them, priced
for decline, and mostly declining.
The quality tests read the trailing decade, and a business whose economics
are deteriorating passes them until well after its price has fallen. Meta in
2022 (+168% the next year) and Intel in 2022 were the exceptions.

Ten annual observations are few. With nine degrees of freedom a t-statistic
needs to pass 2.26 for the usual 5% test; only *Cheap, quality passes*
does, on 19 company-years. Three- and five-year windows overlap and carry no
t-statistic. This is one decade, in which growth beat value; it is evidence
that the verdicts did not help over it, not proof that they cannot.

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
  at the day. **Macro as of the day:** the AAA yield of the last month
  completed, the CPI of the last month published (November's, both, at a
  December year-end).
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
| 2014 | 455 | 382 | 7 | 266 | 1.92x | 4 | 5 | 3 | 132 | 33 | 205 | 73 |
| 2015 | 462 | 395 | 8 | 283 | 1.77x | 18 | 5 | 3 | 129 | 35 | 205 | 67 |
| 2016 | 467 | 414 | 9 | 279 | 1.86x | 7 | 5 | 3 | 136 | 37 | 226 | 53 |
| 2017 | 471 | 427 | 10 | 269 | 2.12x | 4 | 3 | 0 | 142 | 37 | 241 | 44 |
| 2018 | 475 | 440 | 11 | 265 | 1.73x | 10 | 8 | 6 | 143 | 47 | 226 | 35 |
| 2019 | 480 | 453 | 12 | 258 | 1.95x | 5 | 5 | 5 | 150 | 54 | 234 | 27 |
| 2020 | 485 | 453 | 13 | 261 | 2.11x | 15 | 2 | 5 | 155 | 53 | 223 | 32 |
| 2021 | 489 | 460 | 14 | 259 | 2.50x | 3 | 4 | 0 | 165 | 54 | 234 | 29 |
| 2022 | 491 | 469 | 15 | 246 | 1.81x | 14 | 9 | 8 | 154 | 64 | 220 | 22 |
| 2023 | 492 | 475 | 16 | 246 | 1.87x | 9 | 8 | 12 | 146 | 65 | 235 | 17 |

XBRL begins in 2009, so in 2014 the median company had seven fiscal years on
file and Graham's ten-year tests were mostly unmeasurable; *No verdict*
(insufficient data, or no annual report yet) falls from 73 to 17 as the
record lengthens. The market looked dearest to the tracker at the end of
2021 (median 2.50x its estimate).

### Verdict groups against the equal-weighted universe

| Verdict | Company-years | Per year | 1y return | 1y excess | 1y ahead | t (1y) | 3y excess | 3y ahead | 5y excess | 5y ahead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Candidate | 70 | 7.0 | 11.0% | -6.1% | 3/10 | -1.6 | -7.9% | 1/9 | -6.9% | 0/7 |
| Cheap, quality passes | 19 | 1.9 | -2.0% | -19.1% | 1/10 | -2.6 | -13.9% | 0/9 | -10.3% | 0/7 |
| Good business, fair price | 54 | 5.4 | 14.6% | -2.6% | 3/10 | -0.4 | -3.6% | 3/9 | -2.6% | 3/7 |
| Statistically cheap only | 45 | 5.6 | 25.1% | +2.3% | 5/8 | +0.4 | -1.7% | 2/7 | +0.0% | 3/6 |
| Good business, dear | 1452 | 145.2 | 15.2% | -1.9% | 4/10 | -1.3 | -1.8% | 3/9 | -0.2% | 4/7 |
| Good business, value unclear | 479 | 47.9 | 20.0% | +2.9% | 7/10 | +1.9 | +2.4% | 7/9 | +3.8% | 7/7 |
| No | 2248 | 224.8 | 16.5% | -0.6% | 4/10 | -0.6 | -1.0% | 3/9 | -1.2% | 1/7 |
| Insufficient data | 320 | 32.0 | 34.0% | +16.9% | 9/10 | +2.3 | +12.4% | 8/9 | +3.4% | 7/7 |
| No annual report yet | 79 | 7.9 | 22.1% | +5.0% | 6/10 | +0.5 | +9.4% | 8/9 | +5.2% | 5/7 |

The universe: 477 companies a year; 17.1% a year over one-year windows (10
of them), 17.4% over three-year (9), 16.8% over five-year (7). One *Watch*
is left out. *Insufficient data*'s lead is survivorship, not signal: it is
mostly young companies with short filing records that later grew into the
index, and among companies in the index at the time it is +1.1% a year.

### The tracker's buy list

| Group | Company-years | Per year | 1y return | 1y excess | 1y ahead | t (1y) | 3y excess | 3y ahead | 5y excess | 5y ahead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Candidate or cheap, quality passes | 89 | 8.9 | 9.4% | -7.8% | 3/10 | -2.0 | -9.1% | 0/9 | -7.9% | 0/7 |
| Statistically cheap only | 45 | 5.6 | 25.1% | +2.3% | 5/8 | +0.4 | -1.7% | 2/7 | +0.0% | 3/6 |
| Everything else | 4633 | 463.3 | 17.3% | +0.1% | 6/10 | +1.0 | +0.1% | 8/9 | +0.1% | 7/7 |

### Price-to-value

Within each year-end, the companies whose margin of safety the tracker
measured (about 260 a year: the models agree within 3x) are ranked into
fifths by price-to-value.

| Quintile | Median P/V | Per year | 1y return | 1y excess | 1y ahead | t (1y) | 3y excess | 3y ahead | 5y excess | 5y ahead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (cheapest) | 1.02x | 53.1 | 14.6% | -2.5% | 4/10 | -1.0 | -2.9% | 1/9 | -1.9% | 3/7 |
| 2 | 1.50x | 52.5 | 13.0% | -4.1% | 3/10 | -2.1 | -1.6% | 3/9 | -1.3% | 1/7 |
| 3 | 1.93x | 52.4 | 15.0% | -2.1% | 3/10 | -1.9 | -2.0% | 2/9 | -0.4% | 2/7 |
| 4 | 2.45x | 52.5 | 13.7% | -3.4% | 2/10 | -1.9 | -1.8% | 2/9 | -0.8% | 3/7 |
| 5 (dearest) | 3.82x | 52.7 | 17.3% | +0.1% | 5/10 | +0.1 | -1.6% | 5/9 | -0.4% | 3/7 |

Every quintile trails the universe because the companies the tracker can
measure (14.7% a year) trail the ones it cannot - young and fast-growing
companies, whose models disagree. The question is the slope across the
quintiles, and there is none - or a slightly wrong-way one:

| Ranking | Ranked per year | 1y cheapest - dearest | t | 1y rank corr. | t | 3y cheapest - dearest | 3y rank corr. | 5y cheapest - dearest | 5y rank corr. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Price-to-value, margin measured | 263 | -2.6% | -0.6 | -0.036 | -0.6 | -1.3% | -0.057 | -1.5% | -0.062 |
| Price-to-value, every estimate | 434 | -0.4% | -0.1 | -0.004 | -0.1 | +0.6% | -0.022 | -1.8% | -0.037 |
| Earnings yield, the same 263 | 263 | -4.6% | -1.0 | -0.053 | -0.8 | -4.6% | -0.105 | -5.6% | -0.157 |
| Earnings yield, every company with 3 years of EPS | 456 | -10.8% | -1.9 | -0.052 | -0.9 | -9.5% | -0.089 | -6.8% | -0.134 |

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
| Candidate | 66 | 6.6 | 11.1% | -2.6% | 3/10 | -0.7 | -4.6% | 3/9 | -5.0% | 1/7 |
| Cheap, quality passes | 18 | 1.8 | -1.6% | -15.3% | 1/10 | -2.3 | -9.9% | 1/9 | -7.5% | 1/7 |
| Good business, fair price | 51 | 5.1 | 14.0% | +0.3% | 2/10 | +0.1 | -0.8% | 4/9 | -0.9% | 3/7 |
| Statistically cheap only | 43 | 5.4 | 23.6% | +4.8% | 5/8 | +0.9 | -0.3% | 4/7 | -0.4% | 4/6 |
| Good business, dear | 1298 | 129.8 | 14.3% | +0.6% | 5/10 | +0.5 | +0.9% | 6/9 | +1.2% | 5/7 |
| Good business, value unclear | 395 | 39.5 | 16.8% | +3.0% | 7/10 | +1.4 | +3.2% | 8/9 | +3.6% | 6/7 |
| No | 1812 | 181.2 | 12.7% | -1.1% | 4/10 | -1.2 | -1.3% | 2/9 | -1.7% | 0/7 |
| Insufficient data | 162 | 16.2 | 14.8% | +1.1% | 4/10 | +0.6 | -0.4% | 4/9 | +2.0% | 5/7 |

Universe: 385 companies a year, 13.7% a year over one year. Price-to-value
quintiles on this universe: cheapest minus dearest -2.0% a year over one
year (t = -0.5), -1.3% over three, -1.8% over five; rank correlation -0.03.
The conclusions do not change.

## What else is not point in time

- **The configuration.** Config r7 was written in 2026. Its thresholds were
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
