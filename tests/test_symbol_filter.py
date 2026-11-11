#!/usr/bin/env python3
"""Regression: exact NIFTY symbol filter (BANKNIFTY leak corrupted futures CSV, 6 Nov 2026)."""
import csv, sys

def test_fetcher_filter_source():
    src = open('/home/sandbox/nifty-analyst/iv_history/fetch_futures.py').read()
    assert "r.get('TckrSymb') == 'NIFTY'" in src, 'futures fetcher must use exact == NIFTY'
    assert "'NIFTY' in str(" not in src, 'substring match leaked BANKNIFTY (root cause 6 Nov)'
    opt = open('/home/sandbox/nifty-analyst/iv_history/fetch.py').read()
    assert "row['TckrSymb'] == 'NIFTY'" in opt

def test_futures_data_ranges():
    with open('/home/sandbox/nifty-analyst/iv_history/nifty_fut_daily.csv') as f:
        rows = list(csv.reader(f))
    assert rows, 'no futures data'
    for r in rows:
        px = float(r[3] or r[2])
        und = float(r[4])
        assert 15000 < px < 35000, f'futures price out of NIFTY range: {r}'
        assert 15000 < und < 35000, f'underlying out of NIFTY range: {r}'
        assert abs(px - und) / und < 0.06, f'basis >6% implausible: {r}'

def test_options_underlying_range():
    with open('/home/sandbox/nifty-analyst/iv_history/nifty_ido_daily.csv') as f:
        for r in csv.DictReader(f):
            if r['UndrlygPric']:
                u = float(r['UndrlygPric'])
                assert 15000 < u < 35000, f'options underlying out of range: {u}'

if __name__ == '__main__':
    test_fetcher_filter_source(); test_futures_data_ranges(); test_options_underlying_range()
    print('symbol filter regression tests PASS')