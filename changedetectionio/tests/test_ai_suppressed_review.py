import json
import asyncio
import queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from flask import url_for

from changedetectionio.llm.diff_text import build_llm_diff
from changedetectionio.worker import (
    ai_suppresses_change,
    async_update_worker,
    should_save_history_after_ai,
)
from changedetectionio.queuedWatchMetaData import PrioritizedItem


def _decision(important, diff, explanation='Only a cosmetic change'):
    return {
        'important': important,
        'status': 'ok',
        'coverage': 'complete',
        'explanation': explanation,
        'evaluated_diff': diff,
        'model': 'fixture/model',
        'prompt_version': 'b1-test',
    }


def test_worker_baseline_saves_two_suppressions_then_alert_with_own_diffs(client):
    datastore = client.application.config['DATASTORE']
    uuid = datastore.add_watch(url='https://example.com/b1-worker', extras={'paused': True})
    watch = datastore.data['watching'][uuid]

    watch.save_history_blob('price: 10\n', 1700000000, 'one')
    transitions = [
        ('price: 10\nprice: 11\n', 1700000001, False),
        ('price: 10\nprice: 11\nprice: 12\n', 1700000002, False),
        ('price: 10\nprice: 11\nprice: 12\nprice: 13\n', 1700000003, True),
    ]

    for content, timestamp, important in transitions:
        from_version = list(watch.history.keys())[-1]
        diff = build_llm_diff(watch.get_history_snapshot(timestamp=from_version), content)
        result = _decision(important, diff)
        suppressed = ai_suppresses_change(result)
        assert should_save_history_after_ai(important, suppressed, watch.history_n)
        watch.save_history_blob(content, timestamp, str(timestamp))
        watch.save_ai_decision(
            version=str(timestamp), decision=result, intent='flag price changes',
            from_version=from_version, suppressed_by_ai=suppressed,
        )

    versions = list(watch.history.keys())
    assert versions == ['1700000000', '1700000001', '1700000002', '1700000003']
    assert build_llm_diff(
        watch.get_history_snapshot(timestamp=versions[0]),
        watch.get_history_snapshot(timestamp=versions[1]),
    ).count('+price: 11') == 1
    assert build_llm_diff(
        watch.get_history_snapshot(timestamp=versions[1]),
        watch.get_history_snapshot(timestamp=versions[2]),
    ).count('+price: 12') == 1
    assert build_llm_diff(
        watch.get_history_snapshot(timestamp=versions[2]),
        watch.get_history_snapshot(timestamp=versions[3]),
    ).count('+price: 13') == 1
    assert [
        watch.get_ai_decision(version)['suppressed_by_ai'] for version in versions[1:]
    ] == [True, True, False]

    watch['llm_ai_decision_retention'] = 1
    watch.save_history_blob('price: 14\n', 1700000004, '1700000004')
    watch.save_ai_decision(
        version='1700000004',
        decision=_decision(True, '- price: 13\n+ price: 14', 'A price change'),
        intent='flag price changes',
        from_version='1700000003',
        suppressed_by_ai=False,
    )
    assert watch.get_ai_decision('1700000003') is None
    assert watch.get_ai_history_markers()['1700000003']['has_evidence'] is False
    watch.history_trim(newest_n_items=2)
    markers = watch.get_ai_history_markers()
    assert markers['1700000003']['has_evidence'] is False
    assert markers['1700000004']['has_evidence'] is True
    assert all(version not in markers for version in ('1700000000', '1700000001', '1700000002'))


def test_actual_worker_progression_saves_suppressed_baselines_without_notifications(client, monkeypatch):
    datastore = client.application.config['DATASTORE']
    app = client.application
    app.config.exit.clear()
    monkeypatch.setenv('WORKER_MAX_JOBS', '20')
    monkeypatch.setenv('WORKER_MAX_RUNTIME', '30')
    datastore.data['settings']['application']['llm'] = {
        'enabled': True, 'model': 'fixture/model', 'api_key': 'not-used',
    }
    uuid = datastore.add_watch(url='https://example.com/b1-worker-real', extras={
        'paused': True, 'llm_intent': 'flag price changes',
    })

    contents = [
        'price: 10\n',
        'price: 11\n',
        'price: 12\n',
        'price: 13\n',
    ]
    results = iter([
        _decision(False, '- price: 10\n+ price: 11'),
        _decision(False, '- price: 11\n+ price: 12'),
        _decision(True, '- price: 12\n+ price: 13', 'A price change'),
    ])

    class FixtureFetcher:
        def __init__(self, content):
            self.content = content
            self.favicon_blob = None

        def get_all_headers(self):
            return {'content-type': 'text/plain'}

        def clear_content(self):
            self.content = ''

        async def quit(self, watch=None):
            return None

    class FixtureHandler:
        def __init__(self, content):
            self.content = content
            self.fetcher = FixtureFetcher(content)
            self.screenshot = None
            self.xpath_data = None

        async def call_browser(self):
            return None

        def run_changedetection(self, watch):
            return True, {'previous_md5': f'fixture-{self.content.strip()[-2:]}'}, self.content

    class FixtureQueue:
        def __init__(self):
            self.items = [PrioritizedItem(priority=1, item={'uuid': uuid}) for _ in contents]

        async def async_get(self, executor=None, timeout=1.0):
            if self.items:
                return self.items.pop(0)
            app.config.exit.set()
            raise asyncio.TimeoutError

    handlers = iter(FixtureHandler(content) for content in contents)
    processor_module = SimpleNamespace(perform_site_check=lambda **kwargs: next(handlers))
    notification = AsyncMock()
    with patch('changedetectionio.processors.get_processor_module', return_value=processor_module), \
            patch('changedetectionio.worker.apply_update_handler_alter', side_effect=lambda handler, watch, datastore: handler), \
            patch('changedetectionio.worker.apply_update_finalize'), \
            patch('changedetectionio.worker_pool.claim_uuid_for_processing', return_value=True), \
            patch('changedetectionio.worker_pool.release_uuid_from_processing'), \
            patch('changedetectionio.llm.evaluator.evaluate_change', side_effect=lambda *args, **kwargs: next(results)), \
            patch('changedetectionio.llm.evaluator.summarise_change') as summary, \
            patch('changedetectionio.notification_service.watch_will_send_content_changed_notification', return_value=False), \
            patch('changedetectionio.worker.send_content_changed_notification', notification):
        asyncio.run(asyncio.wait_for(
            async_update_worker(
                worker_id=991, q=FixtureQueue(), notification_q=queue.Queue(), app=app,
                datastore=datastore, executor=None,
            ),
            timeout=10,
        ))
    app.config.exit.clear()

    watch = datastore.data['watching'][uuid]
    versions = list(watch.history.keys())
    assert len(versions) == 4
    assert [watch.get_ai_decision(version)['suppressed_by_ai'] for version in versions[1:]] == [True, True, False]
    assert notification.await_count == 1
    summary.assert_not_called()
    assert watch.has_unviewed is True
    assert datastore.unread_changes_count == 1


def test_suppressed_baselines_follow_unread_alert_semantics_across_reload_and_trim(client):
    from changedetectionio.blueprint.watchlist import filters as watchlist_filters
    from changedetectionio.store import ChangeDetectionStore

    datastore = client.application.config['DATASTORE']
    uuid = datastore.add_watch(url='https://example.com/b1-unread', extras={'paused': True})
    watch = datastore.data['watching'][uuid]

    def save_transition(content, timestamp, important):
        from_version = list(watch.history.keys())[-1]
        result = _decision(important, build_llm_diff(watch.get_history_snapshot(from_version), content))
        watch.save_history_blob(content, timestamp, str(timestamp))
        watch.save_ai_decision(
            version=str(timestamp), decision=result, intent='flag price changes',
            from_version=from_version, suppressed_by_ai=ai_suppresses_change(result),
        )

    watch.save_history_blob('price: 10\n', 1900000000, 'baseline')
    save_transition('price: 11\n', 1900000001, False)
    save_transition('price: 12\n', 1900000002, False)

    unread_filter = watchlist_filters.list_filters_from_args(datastore, {'unread': '1'})
    assert watch['last_viewed'] == 0
    assert watch.viewed is True
    assert watch.has_unviewed is False
    assert datastore.unread_changes_count == 0
    assert watchlist_filters.watch_passes_status(watch, unread_filter) is False

    save_transition('price: 13\n', 1900000003, True)
    assert watch.viewed is False
    assert watch.has_unviewed is True
    assert datastore.unread_changes_count == 1
    assert watchlist_filters.watch_passes_status(watch, unread_filter) is True

    save_transition('price: 14\n', 1900000004, False)
    assert watch['last_viewed'] == 0
    assert watch.viewed is False
    assert watch.has_unviewed is True
    assert datastore.unread_changes_count == 1

    datastore.set_last_viewed(uuid, 1900000003, send_signal=False)
    assert watch['last_viewed'] == 1900000003
    assert watch.viewed is True
    assert watch.has_unviewed is False
    assert datastore.unread_changes_count == 0

    save_transition('price: 15\n', 1900000005, True)
    assert watch.has_unviewed is True
    assert datastore.unread_changes_count == 1

    datastore2 = ChangeDetectionStore(datastore_path=datastore.datastore_path)
    datastore2.reload_state(
        datastore_path=datastore.datastore_path,
        include_default_watches=False,
        version_tag='test',
    )
    reloaded_watch = datastore2.data['watching'][uuid]
    assert reloaded_watch['last_viewed'] == 1900000003
    assert reloaded_watch.has_unviewed is True
    assert datastore2.unread_changes_count == 1

    reloaded_watch['llm_ai_decision_retention'] = 1
    reloaded_watch.commit()
    reloaded_watch.history_trim(newest_n_items=2)
    assert reloaded_watch.has_unviewed is True
    assert datastore2.unread_changes_count == 1
    retained_markers = reloaded_watch.get_ai_history_markers()
    assert retained_markers['1900000004']['suppressed_by_ai'] is True
    assert retained_markers['1900000005']['suppressed_by_ai'] is False
    assert retained_markers['1900000005']['has_evidence'] is True


def test_retention_keeps_real_alert_unread_after_initial_baseline_is_trimmed(client):
    from changedetectionio.store import ChangeDetectionStore

    datastore = client.application.config['DATASTORE']
    uuid = datastore.add_watch(url='https://example.com/r3-retained-alert', extras={'paused': True})
    watch = datastore.data['watching'][uuid]

    watch.save_history_blob('initial\n', 100, 'initial')
    assert watch.history_initial_version == '100'
    watch['last_viewed'] = 100
    watch.save_history_blob('real alert\n', 200, 'alert')
    watch.save_history_blob('suppressed baseline\n', 300, 'suppressed')
    watch.save_ai_decision(
        version='300',
        decision=_decision(False, '- real alert\n+ suppressed baseline'),
        intent='flag real alerts',
        from_version='200',
        suppressed_by_ai=True,
    )

    assert list(watch.history.keys()) == ['100', '200', '300']
    assert watch.has_unviewed is True
    assert datastore.unread_changes_count == 1

    watch.commit()
    reloaded_store = ChangeDetectionStore(datastore_path=datastore.datastore_path)
    reloaded_store.reload_state(
        datastore_path=datastore.datastore_path,
        include_default_watches=False,
        version_tag='r3',
    )
    reloaded_watch = reloaded_store.data['watching'][uuid]
    assert reloaded_watch.history_initial_version == '100'

    reloaded_watch.history_trim(newest_n_items=2)
    assert list(reloaded_watch.history.keys()) == ['200', '300']
    assert reloaded_watch.has_unviewed is True
    assert reloaded_store.unread_changes_count == 1

    reloaded_watch.history_trim(newest_n_items=1)
    assert list(reloaded_watch.history.keys()) == ['300']
    assert reloaded_watch.has_unviewed is False
    assert reloaded_store.unread_changes_count == 0


def test_unmarked_retained_alert_is_unread_even_when_only_one_snapshot_remains(client):
    datastore = client.application.config['DATASTORE']
    uuid = datastore.add_watch(url='https://example.com/r3-trim-to-one', extras={'paused': True})
    watch = datastore.data['watching'][uuid]

    watch.save_history_blob('initial\n', 400, 'initial')
    watch['last_viewed'] = 400
    watch.save_history_blob('unmarked pre-ai alert\n', 500, 'alert')
    watch.history_trim(newest_n_items=1)

    assert watch.history_initial_version == '400'
    assert list(watch.history.keys()) == ['500']
    assert watch.has_unviewed is True
    assert datastore.unread_changes_count == 1


def test_unknown_legacy_retained_version_stays_alert_worthy(client):
    import os

    datastore = client.application.config['DATASTORE']
    uuid = datastore.add_watch(url='https://example.com/r3-legacy', extras={'paused': True})
    watch = datastore.data['watching'][uuid]

    watch.save_history_blob('legacy initial\n', 600, 'initial')
    watch.save_history_blob('legacy retained alert\n', 700, 'alert')
    os.unlink(watch.history_initial_version_path)
    watch.history_trim(newest_n_items=1)

    assert watch.history_initial_version is None
    assert list(watch.history.keys()) == ['700']
    assert watch.has_unviewed is True
    assert datastore.unread_changes_count == 1


def test_clear_history_resets_initial_identity_for_the_next_baseline(client):
    datastore = client.application.config['DATASTORE']
    uuid = datastore.add_watch(url='https://example.com/r3-reset', extras={'paused': True})
    watch = datastore.data['watching'][uuid]

    watch.save_history_blob('old initial\n', 800, 'initial')
    watch.save_history_blob('old alert\n', 900, 'alert')
    watch.clear_watch()
    assert watch.history_initial_version is None
    assert watch.has_unviewed is False

    watch.save_history_blob('new initial\n', 1000, 'new-initial')
    assert watch.history_initial_version == '1000'
    assert watch.has_unviewed is False

    watch.save_history_blob('new suppressed\n', 1100, 'new-suppressed')
    watch.save_ai_decision(
        version='1100',
        decision=_decision(False, '- new initial\n+ new suppressed'),
        intent='ignore cosmetic changes',
        from_version='1000',
        suppressed_by_ai=True,
    )
    assert watch.has_unviewed is False
    assert datastore.unread_changes_count == 0


def test_ai_review_feedback_export_retention_and_safe_rendering(client):
    datastore = client.application.config['DATASTORE']
    datastore.data['settings']['application']['llm'] = {
        'enabled': True, 'model': 'fixture/model', 'api_key': 'not-used',
    }
    uuid = datastore.add_watch(url='https://example.com/b1-review', extras={'paused': True})
    watch = datastore.data['watching'][uuid]
    watch.save_history_blob('before\n', 1800000000, 'before')
    watch.save_history_blob('after\n', 1800000001, 'after')
    watch.save_ai_decision(
        version='1800000001',
        decision=_decision(False, '- before\n+ after', '<script>alert(1)</script>'),
        intent='find <unsafe> price changes',
        from_version='1800000000',
        suppressed_by_ai=True,
    )

    diff_page = client.get(url_for('ui.ui_diff.diff_history_page', uuid=uuid))
    assert diff_page.status_code == 200
    assert b'AI suppressed' in diff_page.data
    assert b'Review AI decisions and suppressed changes' in diff_page.data

    review = client.get(url_for('ui.ui_diff.ai_review', uuid=uuid))
    assert review.status_code == 200
    assert b'&lt;script&gt;alert(1)&lt;/script&gt;' in review.data
    assert b'<script>alert(1)</script>' not in review.data
    assert b'- before' in review.data and b'+ after' in review.data

    feedback = client.post(
        url_for('ui.ui_diff.ai_review_feedback', uuid=uuid, version='1800000001'),
        data={'feedback': 'should_alert'},
    )
    assert feedback.status_code == 302
    assert watch.get_ai_decision('1800000001')['feedback'] == 'should_alert'

    exported = client.get(url_for('ui.ui_diff.ai_review_export', uuid=uuid))
    assert exported.status_code == 200
    fixtures = json.loads(exported.data)
    assert fixtures == [{
        'fixture_version': 1,
        'source_version': '1800000001',
        'from_version': '1800000000',
        'intent': 'find <unsafe> price changes',
        'evaluated_diff': '- before\n+ after',
        'expected_important': True,
    }]

    retained = client.post(
        url_for('ui.ui_diff.ai_review_retention', uuid=uuid),
        data={'retention_count': '1'},
    )
    assert retained.status_code == 302
    assert watch.get_ai_decision_retention() == 1

    missing_version = client.get(url_for('ui.ui_diff.ai_review_export', uuid=uuid),
                                 query_string={'version': 'not-a-history-version'})
    assert missing_version.status_code == 404
