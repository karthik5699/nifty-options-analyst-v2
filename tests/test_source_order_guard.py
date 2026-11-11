"""Source-order regressions. Offline only, no broker or quantity-policy changes."""
import copy
from nifty_analyst.adapters.request_safety import interpret_depth_book


def book(buy=None, sell=None):
    return {'mktLot':65,'depth':{'buy':buy if buy is not None else [{'price':100,'quantity':65},{'price':99,'quantity':130}], 'sell':sell if sell is not None else [{'price':101,'quantity':65},{'price':102,'quantity':130}]}}


def test_reject_inconsistent_each_side_and_no_first_row_fallback():
    for side in ('buy','sell'):
        payload=book();payload['depth'][side].reverse(); before=copy.deepcopy(payload)
        result=interpret_depth_book(payload,65)
        assert not result.is_executable
        assert result.rejection_reason=='DEPTH_LEVEL_ORDER_INCONSISTENT'
        assert result.best_bid is None and result.best_ask is None
        assert result.raw_payload==before and payload==before


def test_ordered_and_duplicates_keep_values():
    payload=book();payload['depth']['buy'].insert(1,{'price':100,'quantity':65})
    result=interpret_depth_book(payload,65)
    assert result.is_executable
    assert (result.best_bid,result.best_ask)==(100,101)
    assert (result.bid_depth_lots,result.ask_depth_lots)==(4,3)


def test_invalid_rows_fail_closed_without_numeric_crash():
    for row in [None,{}, {'price':float('inf'),'quantity':65}, {'price':100,'quantity':float('inf')}, {'price':float('nan'),'quantity':65}, {'price':100,'quantity':0.5}, {'price':True,'quantity':65}, {'price':100,'quantity':-1}, {'price':0,'quantity':65}]:
        payload=book();payload['depth']['buy'].append(row)
        result=interpret_depth_book(payload,65)
        assert not result.is_executable
        assert result.rejection_reason=='DEPTH_ROW_INVALID'
        assert result.best_bid is None and result.best_ask is None


def test_zero_zero_padding_and_alias_keys():
    payload=book();payload['depth']['buy'].append({'price':0,'quantity':0})
    payload['depth']['sell']=[{'prc':101,'qty':65}]
    assert interpret_depth_book(payload,65).is_executable


def test_ordered_crossed_locked_empty_and_schema_rejected():
    for payload,reason in [(book(sell=[{'price':100,'quantity':65}]),'LOCKED_BOOK'),(book(sell=[{'price':99,'quantity':65}]),'CROSSED_BOOK'),(book(buy=[]),'EMPTY_BUY_BOOK'),(book(buy='bad'),'DEPTH_SCHEMA_INVALID')]:
        result=interpret_depth_book(payload,65)
        assert not result.is_executable and reason in result.rejection_reason


def test_quantity_unchanged_small_touch_is_not_quantity_rejected():
    payload=book(buy=[{'price':100,'quantity':1},{'price':90,'quantity':649}],sell=[{'price':101,'quantity':1},{'price':110,'quantity':649}])
    result=interpret_depth_book(payload,65)
    assert result.is_executable and result.best_bid==100 and result.best_ask==101
    # Structural use only. This does not certify a full lot at the touch prices.
