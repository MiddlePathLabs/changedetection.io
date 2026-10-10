"""Focused V1 tests for replay isolation, bounded metrics, and cost visibility."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier, RLock
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import litellm
import pytest

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


def _provider_response(text='{"important": true, "summary": "Price changed"}', finish='stop'):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text), finish_reason=finish,
        )],
        usage=SimpleNamespace(total_tokens=18, prompt_tokens=12, completion_tokens=6),
    )


def test_execution_health_uses_measured_calls_not_historical_dispatches_or_cache_hits():
    datastore = _datastore(operation_metrics={
        'evaluation': {'dispatched_calls': 50, 'tokens': 900},
    })
    watch = _watch()
    transient = litellm.InternalServerError(
        message='busy', llm_provider='openai', model='fixture/model',
    )
    with (
        patch('litellm.completion', side_effect=[transient, _provider_response(), _provider_response()]),
        patch('changedetectionio.llm.client.time.sleep'),
        patch('changedetectionio.llm.evaluator.monotonic', side_effect=[100, 107, 200, 202]),
    ):
        evaluate_change(watch, datastore, '- price: 10\n+ price: 11', 'price: 11')
        cached = evaluate_change(watch, datastore, '- price: 10\n+ price: 11', 'price: 11')
        evaluate_change(watch, datastore, '- price: 11\n+ price: 12', 'price: 12')

    assert cached['status'] == 'cached'
    global_metrics = datastore.data['settings']['application']['llm']['operation_metrics']['evaluation']
    assert global_metrics['dispatched_calls'] == 52
    for metrics in (global_metrics, watch['llm_metrics']['evaluation']):
        assert metrics['execution_calls'] == 2
        assert metrics['execution_seconds'] == 9
        assert metrics['execution_max_seconds'] == 7
        assert metrics['provider_attempts'] == 3
        assert metrics['retries'] == 1
        assert metrics['retried_calls'] == 1


@pytest.mark.parametrize('failure, attempts', [
    (litellm.InternalServerError(message='busy', llm_provider='openai', model='fixture/model'), 3),
    (litellm.Timeout(message='timed out', llm_provider='openai', model='fixture/model'), 1),
])
def test_failed_evaluations_contribute_duration_and_actual_retry_counts(failure, attempts):
    datastore = _datastore()
    watch = _watch()
    with (
        patch('litellm.completion', side_effect=failure),
        patch('changedetectionio.llm.client.time.sleep'),
        patch('changedetectionio.llm.evaluator.monotonic', side_effect=[10, 22.5]),
    ):
        result = evaluate_change(watch, datastore, '- price: 10\n+ price: 11', 'price: 11')
    assert result['important'] is True
    assert result['status'] == 'fail_open'
    metrics = watch['llm_metrics']['evaluation']
    assert metrics['execution_calls'] == 1
    assert metrics['execution_seconds'] == 12.5
    assert metrics['execution_max_seconds'] == 12.5
    assert metrics['provider_attempts'] == attempts
    assert metrics['retries'] == attempts - 1
    assert metrics['retried_calls'] == int(attempts > 1)
    assert watch['llm_evaluation_cache'] == {}


def test_empty_output_and_parameter_retries_are_counted_even_when_attempt_budget_is_reset():
    datastore = _datastore()
    watch = _watch()
    rejected = litellm.BadRequestError(
        message='temperature unsupported', llm_provider='openai', model='fixture/model',
    )
    with (
        patch('litellm.completion', side_effect=[
            rejected, _provider_response('', finish='length'), _provider_response(),
        ]),
        patch('changedetectionio.llm.evaluator.monotonic', side_effect=[10, 14]),
    ):
        result = evaluate_change(watch, datastore, '- price: 10\n+ price: 11', 'price: 11')
    assert result['status'] == 'ok'
    metrics = watch['llm_metrics']['evaluation']
    assert metrics['provider_attempts'] == 3
    assert metrics['retries'] == 2
    assert metrics['retried_calls'] == 1
    assert metrics['execution_calls'] == 1
    assert metrics['execution_seconds'] == 4


def test_replay_execution_health_is_global_only():
    datastore = _datastore()
    watch = _watch()
    before = deepcopy(watch)
    with (
        patch('litellm.completion', return_value=_provider_response()),
        patch('changedetectionio.llm.evaluator.monotonic', side_effect=[10, 13]),
    ):
        result = evaluate_change(
            watch, datastore, '- price: 10\n+ price: 11', 'price: 11',
            intent_override='alert on price increases', persist=False, operation='replay',
        )
    assert result['status'] == 'ok'
    assert watch == before
    metrics = datastore.data['settings']['application']['llm']['operation_metrics']['replay']
    assert metrics['execution_calls'] == 1
    assert metrics['execution_seconds'] == 3
    assert metrics['provider_attempts'] == 1


def test_budget_skips_do_not_create_execution_samples():
    datastore = _datastore(
        token_budget_month=1, tokens_month_key=_get_month_key(), tokens_this_month=1,
    )
    watch = _watch()
    result = evaluate_change(watch, datastore, '- price: 10\n+ price: 11', 'price: 11')
    assert result['status'] == 'budget_exceeded'
    metrics = watch['llm_metrics']['evaluation']
    assert metrics['execution_calls'] == 0
    assert metrics['execution_seconds'] == 0
    assert metrics['provider_attempts'] == 0


def test_watch_execution_and_usage_metrics_survive_datastore_reload(tmp_path):
    from changedetectionio.store import ChangeDetectionStore

    datastore = ChangeDetectionStore(datastore_path=str(tmp_path), include_default_watches=False)
    datastore.data['settings']['application']['llm'] = {
        'enabled': True, 'model': 'fixture/model',
    }
    uuid = datastore.add_watch('https://example.com/health', extras={
        'paused': True, 'llm_intent': 'Alert on price changes',
    })
    watch = datastore.data['watching'][uuid]
    with (
        patch('litellm.completion', return_value=_provider_response()),
        patch('changedetectionio.llm.evaluator.monotonic', side_effect=[10, 13]),
    ):
        result = evaluate_change(watch, datastore, '- price: 10\n+ price: 11', 'price: 11')
    assert result['status'] == 'ok'

    restored = ChangeDetectionStore(datastore_path=str(tmp_path), include_default_watches=False)
    watch_metrics = restored.data['watching'][uuid]['llm_metrics']['evaluation']
    global_metrics = restored.data['settings']['application']['llm']['operation_metrics']['evaluation']
    for metrics in (watch_metrics, global_metrics):
        assert metrics['execution_calls'] == 1
        assert metrics['execution_seconds'] == 3
        assert metrics['execution_max_seconds'] == 3
        assert metrics['provider_attempts'] == 1
        assert metrics['retries'] == 0
        assert metrics['tokens'] == 18


def test_concurrent_evaluations_preserve_global_and_watch_health_on_disk(tmp_path):
    from changedetectionio.store import ChangeDetectionStore

    datastore = ChangeDetectionStore(datastore_path=str(tmp_path), include_default_watches=False)
    datastore.data['settings']['application']['llm'] = {
        'enabled': True, 'model': 'fixture/model',
    }
    uuid = datastore.add_watch('https://example.com/concurrent-health', extras={
        'paused': True, 'llm_intent': 'Alert on price changes',
    })
    watch = datastore.data['watching'][uuid]
    barrier = Barrier(4)

    def provider(**kwargs):
        barrier.wait(timeout=10)
        return _provider_response()

    with patch('litellm.completion', side_effect=provider), ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(evaluate_change, watch, datastore, f'- price: {price}\n+ price: {price + 1}')
            for price in range(4)
        ]
        assert all(future.result(timeout=20)['status'] == 'ok' for future in futures)

    restored = ChangeDetectionStore(datastore_path=str(tmp_path), include_default_watches=False)
    watch_metrics = restored.data['watching'][uuid]['llm_metrics']['evaluation']
    global_metrics = restored.data['settings']['application']['llm']['operation_metrics']['evaluation']
    for metrics in (watch_metrics, global_metrics):
        assert metrics['execution_calls'] == 4
        assert metrics['provider_attempts'] == 4
        assert metrics['retries'] == 0
        assert metrics['tokens'] == 72
