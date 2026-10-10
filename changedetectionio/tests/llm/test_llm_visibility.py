"""Focused V1 tests for replay isolation, bounded metrics, and cost visibility."""

from copy import deepcopy
from threading import RLock
from unittest.mock import MagicMock, patch

from changedetectionio.llm.client import CompletionResult
from changedetectionio.llm.evaluator import (
    _get_month_key,
    evaluate_change,
    estimate_cost_usd,
    normalise_llm_metrics,
)


def _datastore(**llm):
    datastore = MagicMock()
    datastore.lock = RLock()
    datastore.data = {
        'settings': {'application': {'llm': {
            'enabled': True,
            'model': 'fixture/model',
            **llm,
        }}},
    }
    return datastore


def _watch():
    return {
        'uuid': 'visibility-test',
        'url': 'https://example.test/item',
        'page_title': 'Fixture item',
        'llm_intent': 'alert on price changes',
        'llm_evaluation_cache': {},
    }


def test_replay_uses_global_metrics_only_and_does_not_cache_or_mutate_watch():
    datastore = _datastore()
    watch = _watch()
    before = deepcopy(watch)
    response = CompletionResult(
        '{"important": false, "summary": "<script>alert(1)</script>"}',
        'stop', 18, 12, 6, True,
    )

    with patch('changedetectionio.llm.evaluator.llm_client.completion', return_value=response):
        result = evaluate_change(
            watch, datastore, '- price: 10\n+ price: 11', 'price: 11',
            intent_override='alert when price drops', persist=False, operation='replay',
        )

    assert result['important'] is False
    assert result['explanation'] == '<script>alert(1)</script>'
    assert watch == before
    replay_metrics = datastore.data['settings']['application']['llm']['operation_metrics']['replay']
    assert replay_metrics['dispatched_calls'] == 1
    assert replay_metrics['input_tokens'] == 12
    assert replay_metrics['output_tokens'] == 6
    assert replay_metrics['tokens'] == 18
    assert replay_metrics['unknown_cost'] == 1


def test_replay_global_budget_skip_never_calls_provider_or_watch_budget():
    datastore = _datastore(
        token_budget_month=1,
        tokens_month_key=_get_month_key(),
        tokens_this_month=1,
    )
    watch = _watch()
    with patch('changedetectionio.llm.evaluator.llm_client.completion') as completion:
        result = evaluate_change(
            watch, datastore, '- a\n+ b', 'b', intent_override='new intent',
            persist=False, operation='replay',
        )
    assert result['status'] == 'budget_exceeded'
    completion.assert_not_called()
    assert 'llm_tokens_this_period' not in watch
    assert datastore.data['settings']['application']['llm']['operation_metrics']['replay']['budget_skips'] == 1


def test_unknown_and_known_costs_are_distinguishable():
    assert estimate_cost_usd('fixture/model', 10, 5) is None
    assert estimate_cost_usd('gpt-4o-mini', 10, 5) is not None
    metrics = normalise_llm_metrics({'replay': {'unknown_cost': 2}})
    assert set(metrics) >= {'evaluation', 'summary', 'replay'}
    assert metrics['replay']['unknown_cost'] == 2


def test_live_evaluation_counts_cache_lookups_separately_from_hits():
    datastore = _datastore()
    watch = _watch()
    response = CompletionResult(
        '{"important": true, "summary": "Fixture alert"}',
        'stop', 18, 12, 6, True,
    )

    with patch('changedetectionio.llm.evaluator.llm_client.completion', return_value=response):
        first = evaluate_change(watch, datastore, '- price: 10\n+ price: 11', 'price: 11')
        second = evaluate_change(watch, datastore, '- price: 10\n+ price: 11', 'price: 11')

    assert first['status'] == 'ok'
    assert second['status'] == 'cached'
    metrics = datastore.data['settings']['application']['llm']['operation_metrics']['evaluation']
    assert metrics['cache_lookups'] == 2
    assert metrics['cache_hits'] == 1
    assert metrics['dispatched_calls'] == 1
    assert watch['llm_metrics']['evaluation']['cache_lookups'] == 2
    assert watch['llm_metrics']['evaluation']['cache_hits'] == 1
