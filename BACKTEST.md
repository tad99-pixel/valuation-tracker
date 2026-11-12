# Backtest: did the verdicts predict anything?

The tracker's verdicts and price-to-value ratios, as it would have given them
at each year-end from 2014 to 2023, set against the total returns that
followed. Run on 2026-11-12 with config r9; returns to 2026-11-09.

It has been run twice. The first run, under config r8, found that the
tracker's buy list had lagged the market by 7.6% a year, and config r9's two
changes to the value models were made because of it. This page reports r9
with r8 beside it. **r9's numbers are in-sample**: its rules were written
after these ten years had been seen. They show whether the corrections did
what they were meant to do. They are not evidence of what the tracker will
do next.

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

- **The tracker's buy list lagged, under both configs.** *Candidate* and
  *Cheap, quality passes* together - about nine companies a year - trailed
  the equal-weighted universe by 6.3% a year over one-year windows (ahead in
  3 of 10 years; t = -1.7), by 6.6% a year over three years and 3.7% over
  five. Under r8 it was 7.6%, 9.1% and 7.6%. Among companies that were in
  the index at the time, *Candidate* trailed by 3.4% a year.
- **Among cheap companies, passing the quality tests did not help.**
  *Statistically cheap only* - cheap, quality fails - did better than the
  buy list at every horizon: -2.0% a year over one year, +0.9% over three,
  +2.8% over five.
- **Price-to-value ranked nothing.** The cheapest fifth by measured
  price-to-value did no better than the dearest (-0.8% a year over one year,
  -0.1% over three, -1.7% over five; r8: -1.4%, -0.7%, -1.6%); the rank
  correlation with the next year's return was -0.01. Including the margins
  the tracker withholds makes no difference.
- **This was a bad decade for any value sort.** Ranking the same companies
  by plain earnings yield did worse (-1.7% a year over one year, cheapest
  minus dearest, and -4.2% over five); across the whole universe, -12.0%.
  The tracker's measure lost less than a P/E sort, but nothing here shows it
  finding anything.
- **The best verdict was *Good business, value unclear*** - the businesses
  whose models disagree by more than 3x, mostly growth companies whose DCF
  runs far above their earnings power - at +3.2% a year (t = 1.9), and
  ahead in all seven five-year windows.

Who was on the buy list explains most of it. Aflac in seven years of ten
(about level with the market). Franklin Resources and Gilead in five each,
Paramount in four - an asset manager losing its assets, a drug maker past
its peak, a broadcaster through its streaming losses: -19%, -17% and -42% a
year against the market. Refiners, chemical and fertiliser makers
(LyondellBasell, Valero, CF Industries, Mosaic) - the cyclicals a ten-year
record flatters. Airlines and a cruise line at the end of 2020, valued on
2019's annual reports. Businesses with ten good years behind them, priced
for decline, and mostly declining: the quality tests read the trailing
decade, and a business whose economics are deteriorating passes them until
well after its price has fallen. The nine bank picks did well (14 points a
year ahead: Synchrony four times, Capital One twice, Citigroup, Fifth Third
and Wells Fargo); Meta in 2022 (168 points ahead of the market the next
year), Lam Research in 2018 and Intel in 2022 were the exceptions among the
rest.

Ten annual observations are few. With nine degrees of freedom a t-statistic
needs to pass 2.26 for the usual 5% test; only *Insufficient data* - a
survivorship artefact, below - does. Three- and five-year windows overlap
and carry no t-statistic. This is one decade, in which growth beat value; it
is evidence that the verdicts did not help over it, not proof that they
cannot.

## What config r9 changed, and what each change did

The first run's buy list pointed at something mechanical. Both value models
capitalised a mean of past *dollars* - five years of owner earnings, seven
of operating income. For a company growing 10% a year that mean sits a sixth
to a quarter below what it earns now; for one shrinking 8% a year, a fifth
to a quarter above. The models leaned by that much towards whatever was in
decline.

Config r9 made two corrections, each argued from the models rather than
from these returns, and tried a third:

1. **Earnings base.** The company's average margin on the revenue it holds
   now (the lower of last year's and the three-year mean), instead of a mean
   of dollars.
2. **Growth.** Above the terminal rate, growth is credited only as far as
   the latest annual report's three years still show it. At the end of 2017
   and 2018 Gilead's ten-year record still earned the DCF's 15% cap under
   r8, with revenue falling; under r9 it is credited 10.5% and then 2.5%.
3. **Graham's earnings-growth test** (ch. 15: last year's earnings above
   those of four years before), required of the buy-side verdicts. **Tried,
   and not adopted** - it is shown beside the quality tests and decides
   nothing.

The buy list, one change at a time (excess return a year against the
equal-weighted universe; each row is a full run of the backtest):

| Models | Company-years | 1y | 3y | 5y |
|---|---:|---:|---:|---:|
| Config r8, as first run | 92 | -7.6% | -9.1% | -7.6% |
| r8's models on r9's data fixes (spin-offs, ticker map) | 92 | -7.6% | -9.1% | -7.6% |
| ... with the earnings base only | 100 | -5.7% | -6.2% | -3.8% |
| ... with the growth rule only | 86 | -7.2% | -8.6% | -7.1% |
| ... with Graham's test required, only | 67 | -6.4% | -9.7% | -7.6% |
| **Config r9: earnings base and growth rule** | **92** | **-6.3%** | **-6.6%** | **-3.7%** |
| r9 with Graham's test required as well (not adopted) | 72 | -5.3% | -6.3% | -3.1% |

The earnings base does most of what was done. The lag is smaller and still
there, and price-to-value ranks no better than before. None of the
differences between rows is distinguishable from noise on ninety
company-years; the corrections are in because the old models were wrong in a
way that can be shown on paper, not because of this table.

**Why Graham's test is not required.** With it, the list lags a little less
by the table's measure. But the 20 picks it removes are not clearly worse
than the 72 it keeps: pooled, they trailed the market by 8.9% over the next
year against 2.1% for the rest, and by *less* over three years and five
(1.8% a year against 4.4%, and 2.3% against 3.3%). Across all companies,
those that failed the test - a quarter of them in any year - went on to beat
the rest at every horizon (+0.8% a year against -1.2% over one year, +0.5%
against -0.6% over five). Depressed earnings recover more often than not,
which is the premise of judging a business on ten years. A rule whose
evidence changes sign with the horizon is one more thing to be wrong about.

**One version of the growth rule was dropped for what it did, though it
scored better.** It let two years of falling revenue impose a decline, where
r9 only withholds the growth premium. Over one year it took the buy list's
lag to 4.2% with Graham's test required. It also read Deere, at the bottom
of a farm cycle, as shrinking for a decade: 1.30x and 1.14x its estimate at
the end of 2015 and 2016, where r8 had it at 0.81x and 0.66x - and Deere
returned 39% and 55% over the following years. Two years show that growth
has stopped, not that a business will shrink for ten.

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
  at the day. The tracker's check on a split-shaped break in the share count
  reads that price record and the splits on record, and both stop at the
  day. A close before a later spin-off is put back as it traded: Yahoo has
  since divided it by the spin-off's ratio, and the day's filings were never
  restated for it. **Macro as of the day:** the AAA yield of the last month
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
| 2014 | 455 | 384 | 7 | 289 | 1.80x | 4 | 6 | 3 | 137 | 30 | 204 | 71 |
| 2015 | 462 | 397 | 8 | 307 | 1.71x | 20 | 6 | 5 | 131 | 32 | 203 | 65 |
| 2016 | 467 | 414 | 9 | 320 | 1.83x | 8 | 3 | 1 | 144 | 31 | 227 | 53 |
| 2017 | 471 | 428 | 10 | 308 | 2.07x | 3 | 3 | 2 | 149 | 32 | 239 | 43 |
| 2018 | 475 | 443 | 11 | 308 | 1.70x | 10 | 10 | 10 | 149 | 39 | 225 | 32 |
| 2019 | 480 | 454 | 12 | 312 | 1.91x | 6 | 5 | 6 | 160 | 44 | 233 | 26 |
| 2020 | 485 | 456 | 13 | 322 | 2.05x | 15 | 5 | 9 | 161 | 44 | 222 | 29 |
| 2021 | 489 | 463 | 14 | 317 | 2.45x | 3 | 5 | 1 | 169 | 49 | 236 | 26 |
| 2022 | 491 | 470 | 15 | 301 | 1.73x | 14 | 10 | 10 | 157 | 56 | 223 | 21 |
| 2023 | 492 | 477 | 16 | 305 | 1.73x | 9 | 9 | 12 | 151 | 55 | 241 | 15 |

XBRL begins in 2009, so in 2014 the median company had seven fiscal years on
file and Graham's ten-year tests were mostly unmeasurable; *No verdict*
(insufficient data, or no annual report yet) falls from 71 to 15 as the
record lengthens. The market looked dearest to the tracker at the end of
2021 (median 2.45x its estimate). With the models normalised on revenue, a
few more margins are measurable each year than under r8 (about 310 against
300).

### Verdict groups against the equal-weighted universe

| Verdict | Company-years | Per year | 1y return | 1y excess | 1y ahead | t (1y) | 3y excess | 3y ahead | 5y excess | 5y ahead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Candidate | 72 | 7.2 | 10.7% | -6.4% | 3/10 | -1.8 | -6.5% | 0/9 | -3.7% | 2/7 |
| Cheap, quality passes | 20 | 2.2 | 4.6% | -12.0% | 2/9 | -1.9 | -11.4% | 1/9 | -7.1% | 2/7 |
| Good business, fair price | 62 | 6.2 | 15.6% | -1.5% | 4/10 | -0.4 | -3.5% | 4/9 | -4.6% | 3/7 |
| Statistically cheap only | 59 | 5.9 | 15.1% | -2.0% | 5/10 | -0.6 | +0.9% | 4/9 | +2.8% | 5/7 |
| Good business, dear | 1508 | 150.8 | 15.2% | -2.0% | 4/10 | -1.5 | -1.9% | 3/9 | -0.3% | 4/7 |
| Good business, value unclear | 412 | 41.2 | 20.3% | +3.2% | 7/10 | +1.9 | +3.1% | 8/9 | +4.7% | 7/7 |
| No | 2253 | 225.3 | 16.7% | -0.5% | 5/10 | -0.4 | -0.8% | 3/9 | -1.3% | 1/7 |
| Insufficient data | 302 | 30.2 | 32.0% | +14.9% | 9/10 | +2.3 | +11.9% | 9/9 | +3.8% | 7/7 |
| No annual report yet | 79 | 7.9 | 22.1% | +5.0% | 6/10 | +0.5 | +9.4% | 8/9 | +5.2% | 5/7 |

The universe: 477 companies a year; 17.1% a year over one-year windows (10
of them), 17.4% over three-year (9), 16.8% over five-year (7).
*Insufficient data*'s lead is survivorship, not signal: it is mostly young
companies with short filing records that later grew into the index, and
among companies in the index at the time it is +1.2% a year. *Cheap, quality
passes* held about two companies a year; six of its 20 company-years were
Gilead and Biogen.

### The tracker's buy list

| Group | Company-years | Per year | 1y return | 1y excess | 1y ahead | t (1y) | 3y excess | 3y ahead | 5y excess | 5y ahead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Candidate or cheap, quality passes | 92 | 9.2 | 10.9% | -6.3% | 3/10 | -1.7 | -6.6% | 0/9 | -3.7% | 2/7 |
| Statistically cheap only | 59 | 5.9 | 15.1% | -2.0% | 5/10 | -0.6 | +0.9% | 4/9 | +2.8% | 5/7 |
| Everything else | 4616 | 461.6 | 17.2% | +0.1% | 7/10 | +0.7 | +0.1% | 8/9 | +0.0% | 4/7 |

Under r8: 92 company-years, -7.6% (t = -2.0), -9.1% and -7.6%, ahead in 3
of 10, 0 of 9 and 0 of 7 windows.

### Price-to-value

Within each year-end, the companies whose margin of safety the tracker
measured (about 310 a year: the models agree within 3x) are ranked into
fifths by price-to-value.

| Quintile | Median P/V | Per year | 1y return | 1y excess | 1y ahead | t (1y) | 3y excess | 3y ahead | 5y excess | 5y ahead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (cheapest) | 1.02x | 62.2 | 14.5% | -2.6% | 4/10 | -1.0 | -2.4% | 1/9 | -1.9% | 1/7 |
| 2 | 1.48x | 61.5 | 14.0% | -3.2% | 4/10 | -1.8 | -2.4% | 2/9 | -2.2% | 1/7 |
| 3 | 1.86x | 61.6 | 13.0% | -4.1% | 4/10 | -2.2 | -3.7% | 1/9 | -1.8% | 1/7 |
| 4 | 2.33x | 61.5 | 14.6% | -2.5% | 2/10 | -2.9 | -2.7% | 1/9 | -1.7% | 0/7 |
| 5 (dearest) | 3.48x | 62.1 | 15.3% | -1.8% | 4/10 | -0.9 | -2.4% | 5/9 | -0.1% | 3/7 |

Every quintile trails the universe because the companies the tracker can
measure (14.3% a year) trail the ones it cannot - young and fast-growing
companies, whose models disagree. The question is the slope across the
quintiles, and there is none - or a slightly wrong-way one:

| Ranking | Ranked per year | 1y cheapest - dearest | t | 1y rank corr. | t | 3y cheapest - dearest | 3y rank corr. | 5y cheapest - dearest | 5y rank corr. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Price-to-value, margin measured | 309 | -0.8% | -0.2 | -0.014 | -0.3 | -0.1% | -0.046 | -1.7% | -0.085 |
| Price-to-value, every estimate | 436 | -2.9% | -0.8 | -0.015 | -0.3 | -0.5% | -0.032 | -1.8% | -0.046 |
| Earnings yield, the same companies | 308 | -1.7% | -0.4 | -0.024 | -0.4 | -2.0% | -0.053 | -4.2% | -0.115 |
| Earnings yield, every company with 3 years of EPS | 459 | -12.0% | -1.9 | -0.055 | -0.9 | -10.5% | -0.091 | -6.8% | -0.136 |

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
| Candidate | 67 | 6.7 | 10.4% | -3.4% | 3/10 | -1.0 | -4.1% | 1/9 | -3.2% | 2/7 |
| Cheap, quality passes | 18 | 2.0 | 5.3% | -8.3% | 2/9 | -1.3 | -7.7% | 2/9 | -4.5% | 2/7 |
| Good business, fair price | 61 | 6.1 | 15.4% | +1.6% | 5/10 | +0.5 | +0.2% | 5/9 | -2.1% | 3/7 |
| Statistically cheap only | 53 | 5.3 | 13.9% | +0.2% | 5/10 | +0.1 | +1.3% | 4/9 | -4.3% | 3/7 |
| Good business, dear | 1345 | 134.5 | 14.2% | +0.4% | 5/10 | +0.4 | +0.8% | 6/9 | +1.2% | 5/7 |
| Good business, value unclear | 339 | 33.9 | 17.1% | +3.3% | 7/10 | +1.5 | +3.7% | 8/9 | +4.2% | 6/7 |
| No | 1812 | 181.2 | 12.7% | -1.1% | 4/10 | -1.2 | -1.4% | 2/9 | -1.7% | 0/7 |
| Insufficient data | 151 | 15.1 | 15.0% | +1.2% | 5/10 | +0.6 | +0.4% | 5/9 | +2.4% | 5/7 |

Universe: 385 companies a year, 13.7% a year over one year. Price-to-value
quintiles on this universe: cheapest minus dearest -0.8% a year over one
year (t = -0.2), -0.4% over three, -2.2% over five; rank correlation -0.01.
The conclusions do not change.

## What else is not point in time

- **The configuration.** Config r9 was written in 2026, and its two changes
  to the value models after the first run of this backtest: see the top of
  this page. Its thresholds were set by reading Graham and Buffett and its
  data repairs by finding XBRL traps, not by fitting returns - but a 2014
  version of the tracker would not have known about the traps the later
  ones fixed.
- **Yahoo's histories.** Prices exist only for tickers that still trade;
  dividends and splits are as Yahoo records them. Filer metadata - SIC code,
  entity type - is today's.
- **Annual reports only.** The tracker reads each company's latest annual
  report, so at a December year-end most figures are eleven months old, and
  in the live tracker they can be fourteen. At the end of 2020 the airlines
  and Carnival were valued on 2019.
- **DuPont** (a merger of equals in 2017) has no single predecessor whose
  filings match its price history, and has no verdict before 2019.

The first run noted that the tracker restated Yahoo's spin-off "splits" as
share splits, so GE's, 3M's and AT&T's share counts before a spin-off read
1.2-1.6x too high. Config r9 fixed that in the tracker; it changed nothing
in these tables (the second row of the table above).

## Reproduce

```bash
python backtest_valuation.py                   # ~15 minutes; downloads prices and dividends once
python backtest_valuation.py --analyse-only    # redo the tables from the saved rows
python tests/test_backtest.py                  # its tests, offline
```

Everything it downloads and writes stays in `data_cache/backtest/`, which is
not published: `backtest_rows.csv` holds every company's verdict, price,
price-to-value and forward returns at every year-end, and
`backtest_tables.md` the tables above. The single-change rows come from the
switches in `valuation/config.py` - `normalise_on_revenue`,
`cap_growth_at_recent` and `require_earnings_growth`.
