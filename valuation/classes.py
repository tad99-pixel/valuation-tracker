"""Per-share figures for companies that tag them only by share class.

SEC's companyfacts feed carries only facts without dimensions. A company with
several share classes may tag its EPS and share counts only per class -
Visa (Class A, B and C) since 2010, Berkshire Hathaway (Class A and B) since
2014 - and to that feed it then has no per-share history at all: Visa and
Berkshire sat at INSUFFICIENT DATA for that reason alone.

The figures are in each annual report's own XBRL document, on SEC's archive
site, where every fact keeps its class. This reads them for the class that
trades under the ticker (`config.SHARE_CLASSES`) and adds them to the
company's facts as if filed without a class, so the rest of the pipeline -
split adjustment by filing date, latest-filed-wins, the panel - treats them
like any other filing.

SEC's archive site refuses automated requests that do not name a contact
(set VALUATION_USER_AGENT, e.g. "ValuationTracker/1.0 you@example.com"). What
has been read is kept in valuation_state/share_classes.json - annual reports
do not change once filed - so a run that cannot reach the archive still has
every year read before, and says so.
"""

from __future__ import annotations

import hashlib
import json
import re
import xml.etree.ElementTree as ET
from io import BytesIO

from .config import SHARE_CLASSES, STATE_DIR
from .sources import _read_json, _sec_get, _write_json

STORE_PATH = STATE_DIR / 'share_classes.json'
SUBMISSIONS = 'https://data.sec.gov/submissions/{name}'
ARCHIVE = 'https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/'
# The concepts read, and the companyfacts unit each is filed under.
CONCEPTS = {
    'EarningsPerShareDiluted': 'USD/shares',
    'EarningsPerShareBasic': 'USD/shares',
    'WeightedAverageNumberOfDilutedSharesOutstanding': 'shares',
    'WeightedAverageNumberOfSharesOutstandingBasic': 'shares',
}
# Each 10-K restates two earlier years, so this many reports cover the
# twenty-odd years the tests can use.
MAX_REPORTS = 16
_LINKBASE = re.compile(r'(_cal|_def|_lab|_pre)\.xml$|^FilingSummary\.xml$|^R\d+\.xml$')


def parse_instance(xml_bytes: bytes, member: str) -> list:
    """Facts for `member` of the share-class axis in one XBRL instance.

    Only facts whose context carries exactly that one dimension: a figure
    split by class *and* by segment is a different number.
    """
    contexts, facts = {}, []
    for _, el in ET.iterparse(BytesIO(xml_bytes)):
        tag = el.tag.rsplit('}', 1)[-1]
        if tag == 'context':
            dims = {(m.get('dimension') or '').rsplit(':', 1)[-1]:
                    (m.text or '').strip().rsplit(':', 1)[-1]
                    for m in el.iter() if m.tag.endswith('explicitMember')}
            period = {c.tag.rsplit('}', 1)[-1]: (c.text or '').strip() for c in el.iter()
                      if c.tag.rsplit('}', 1)[-1] in ('startDate', 'endDate', 'instant')}
            contexts[el.get('id')] = (dims, period.get('startDate'), period.get('endDate'))
            el.clear()
        elif tag in CONCEPTS:
            facts.append((tag, el.get('contextRef'), (el.text or '').strip()))
            el.clear()
    out = []
    for concept, ref, text in facts:
        dims, start, end = contexts.get(ref, ({}, None, None))
        if dims != {'StatementClassOfStockAxis': member} or not start or not end:
            continue
        try:
            out.append({'concept': concept, 'start': start, 'end': end, 'val': float(text)})
        except ValueError:
            continue
    return out


def _annual_reports(cik: int, pause: float) -> list:
    """``[(accession, filing date)]`` of the company's 10-Ks, newest first.

    From the submissions feed (data.sec.gov), following its older pages only
    until enough reports are found.
    """
    j = _sec_get(SUBMISSIONS.format(name=f'CIK{cik:010d}.json'), timeout=45, pause=pause).json()
    pages, reports = [j['filings']['recent']], []
    pages += [None] * len(j['filings'].get('files', []))
    for i, page in enumerate(pages):
        if page is None:
            name = j['filings']['files'][i - 1]['name']
            page = _sec_get(SUBMISSIONS.format(name=name), timeout=45, pause=pause).json()
        reports += [(a, d) for f, a, d in zip(page['form'], page['accessionNumber'],
                                              page['filingDate']) if f == '10-K']
        if len(reports) >= MAX_REPORTS:
            break
    return sorted(reports, key=lambda r: r[1], reverse=True)[:MAX_REPORTS]


def _instance(cik: int, accession: str, pause: float):
    """The XBRL instance of one filing, or None if it has none.

    An inline-XBRL filing (2019 on) has an extracted instance, `*_htm.xml`;
    an older one lists a single `.xml` that is not a linkbase or the summary.
    """
    folder = ARCHIVE.format(cik=cik, folder=accession.replace('-', ''))
    r = _sec_get(folder + 'index.json', timeout=45, pause=pause)
    r.raise_for_status()
    names = [i['name'] for i in r.json()['directory']['item']]
    name = next((n for n in names if n.endswith('_htm.xml')), None) or next(
        (n for n in names if n.endswith('.xml') and not _LINKBASE.search(n)), None)
    if name is None:
        return None
    r = _sec_get(folder + name, timeout=120, pause=pause)
    r.raise_for_status()
    return r.content


def refresh(ciks: dict, pause: float = 0.12, verbose: bool = True) -> None:
    """Read every annual report not yet read, for the tickers in `ciks` that
    `SHARE_CLASSES` lists. A refused or failed request stops that ticker for
    this run and keeps everything read before."""
    store = _read_json(STORE_PATH, {})
    for t, cik in ciks.items():
        member = SHARE_CLASSES.get(t)
        if not member:
            continue
        entry = store.get(t)
        if not entry or entry.get('member') != member:
            entry = store[t] = {'member': member, 'read': [], 'facts': []}
        try:
            todo = [r for r in _annual_reports(cik, pause) if r[0] not in entry['read']]
        except Exception as exc:  # noqa: BLE001
            todo, problem = [], f'filing list unavailable ({type(exc).__name__})'
        else:
            problem = None
        for accession, filed in todo:
            try:
                xml = _instance(cik, accession, pause)
            except Exception as exc:  # noqa: BLE001
                status = getattr(getattr(exc, 'response', None), 'status_code', None)
                problem = ('SEC\'s archive refused the request (403) - it needs a contact in '
                           'VALUATION_USER_AGENT' if status == 403 else f'{type(exc).__name__}')
                break
            rows = parse_instance(xml, member) if xml else []
            entry['facts'] += [dict(r, filed=filed, accn=accession) for r in rows]
            entry['read'].append(accession)
        if verbose:
            years = len({f['end'] for f in entry['facts'] if f['concept'].startswith('Earnings')})
            print(f'  share classes: {t} {member}, {years} years of figures from '
                  f'{len(entry["read"])} annual reports' + (f'; {problem}' if problem else ''))
    _write_json(STORE_PATH, store)


def _facts(ticker: str) -> list:
    return (_read_json(STORE_PATH, {}).get(ticker) or {}).get('facts') or []


def signature(ticker: str):
    """A hash of the class figures held for `ticker`, for the panel cache key;
    None when there are none."""
    facts = _facts(ticker)
    if not facts:
        return None
    return hashlib.sha1(json.dumps(sorted(facts, key=lambda f: json.dumps(f, sort_keys=True)),
                                   sort_keys=True).encode()).hexdigest()[:16]


def inject(blob: dict, ticker: str) -> dict:
    """Add the class figures held for `ticker` to its companyfacts document,
    as annual 10-K facts without a dimension. Returns `blob`."""
    gaap = blob['facts'].setdefault('facts', {}).setdefault('us-gaap', {})
    for f in _facts(ticker):
        units = gaap.setdefault(f['concept'], {'units': {}})['units']
        units.setdefault(CONCEPTS[f['concept']], []).append({
            'start': f['start'], 'end': f['end'], 'val': f['val'], 'filed': f['filed'],
            'accn': f['accn'], 'form': '10-K', 'fp': 'FY'})
    return blob
