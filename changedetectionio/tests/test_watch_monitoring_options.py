"""Integration checks for saved monitoring settings and real processor/worker paths."""

import asyncio
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from flask import url_for

from changedetectionio.processors import save_processor_config
from changedetectionio.processors.restock_diff import Restock
from changedetectionio.processors.restock_diff import processor as restock_processor


def add_watch(client, **extras):
    datastore = client.application.config['DATASTORE']
    uuid = datastore.add_watch(url='https://example.com/product', extras={'paused': True, **extras})
    return datastore, uuid, datastore.data['watching'][uuid]


def test_edit_form_saves_stock_phrases_and_health_options(client):
    datastore, uuid, watch = add_watch(client, processor='restock_diff')
    response = client.post(url_for('ui.ui_edit.edit_page', uuid=uuid), data={
        'url': watch['url'], 'processor': 'restock_diff',
        'time_between_check_use_default': 'on',
        'check_failure_notification_send': 'on',
        'check_recovery_notification_send': 'on',
        'processor_config_restock_diff-custom_instock_strings': 'Ready to ship\nDisponible ahora',
        'processor_config_restock_diff-custom_outofstock_strings': 'Sold out',
        'processor_config_restock_diff-in_stock_processing': 'all_changes',
    }, follow_redirects=True)
    assert response.status_code == 200
    from bs4 import BeautifulSoup
    assert b'Updated watch.' in response.data, BeautifulSoup(response.data, 'html.parser').get_text(' ', strip=True)
    assert watch['check_failure_notification_send']
    assert watch['check_recovery_notification_send']
    handler = restock_processor.perform_site_check(datastore, uuid)
    config = handler.get_extra_watch_config('restock_diff.json')['restock_diff']
    assert config['custom_instock_strings'] == ['Ready to ship', 'Disponible ahora']
    assert config['custom_outofstock_strings'] == ['Sold out']
    response = client.get(url_for('ui.ui_edit.edit_page', uuid=uuid))
    assert b'Disponible ahora' in response.data


def test_settings_save_the_failure_threshold_and_api_hides_health_state(client):
    datastore, uuid, watch = add_watch(client)
    response = client.post(url_for('settings.settings_page'), data={
        'application-check_failure_notification_threshold_attempts': '3',
        'application-fetch_backend': 'html_requests',
        'requests-time_between_check-minutes': '180',
    }, follow_redirects=True)
    assert response.status_code == 200
    assert datastore.data['settings']['application']['check_failure_notification_threshold_attempts'] == 3
    watch.reset_watch_edited_flag()
    from changedetectionio.watch_health import record_check_outcome
    record_check_outcome(datastore, Queue(), uuid, False)
    assert not watch.was_edited
    token = datastore.data['settings']['application']['api_access_token']
    response = client.get(url_for('watch', uuid=uuid), headers={'x-api-key': token})
    assert response.status_code == 200
    assert 'consecutive_check_failures' not in response.json
    assert 'check_failure_alerted' not in response.json
    assert 'check_failure_notification_send' in response.json


def test_failure_notification_formats_error_as_data_without_a_snapshot(client):
    from changedetectionio.notification_service import create_notification_service
    from changedetectionio.notification.handler import process_notification

    datastore, uuid, watch = add_watch(client, notification_urls=['null://'], notification_format='htmlcolor')
    watch['last_error'] = '<img src=x onerror=alert(1)> {{ 7 * 7 }}'
    watch['consecutive_check_failures'] = 6
    notifications = Queue()
    assert create_notification_service(datastore, notifications).send_check_health_notification(uuid)
    rendered = process_notification(notifications.get(), datastore)[0]['body']
    assert '6 consecutive times' in rendered
    assert '&lt;img' in rendered, rendered
    assert '<img' not in rendered
    assert '{{ 7 * 7 }}' in rendered
    assert watch['url'] in rendered


@pytest.mark.parametrize('metadata,html,phrases,expected', [
    ({'price': 100, 'availability': 'instock'}, '<p>Sold out</p>', {'custom_outofstock_strings': ['sold out']}, False),
    ({'price': 100, 'availability': 'outofstock'}, '<p>Ready to ship</p>', {'custom_instock_strings': ['ready to ship']}, True),
    ({}, '<p>Ready to ship</p>', {'custom_instock_strings': ['ready to ship']}, True),
])
def test_custom_phrases_override_metadata_in_real_processor(client, monkeypatch, metadata, html, phrases, expected):
    datastore, uuid, watch = add_watch(client, processor='restock_diff')
    save_processor_config(datastore, uuid, {'restock_diff': {'in_stock_processing': 'all_changes', **phrases}})
    handler = restock_processor.perform_site_check(datastore, uuid)
    handler.fetcher = MagicMock(content=html, headers={'Content-Type': 'text/html'},
                               instock_data=None, backend_name=None, screenshot=None, xpath_data=None)
    handler.fetcher.get_last_status_code.return_value = 200
    monkeypatch.setattr(restock_processor, 'extract_itemprop_availability_safe', lambda html: Restock(metadata))
    changed, updates, content = handler.run_changedetection(watch)
    assert updates['restock']['in_stock'] is expected
    assert updates['restock'].get('price') == metadata.get('price')
    assert f'In Stock: {expected}' in content


def test_worker_records_navigation_failures_and_unchanged_recovery(client, monkeypatch):
    from changedetectionio import worker
    from changedetectionio.content_fetchers.exceptions import PageUnloadable, checksumFromPreviousCheckWasTheSame
    from changedetectionio import processors

    datastore, uuid, watch = add_watch(client, check_failure_notification_send=True,
                                     notification_urls=['json://example.com/alerts'])
    datastore.data['settings']['application']['check_failure_notification_threshold_attempts'] = 2
    notifications = Queue()
    app = SimpleNamespace(config=SimpleNamespace(exit=Event()))
    pool = MagicMock()
    pool.claim_uuid_for_processing.return_value = True
    pool.release_uuid_from_processing.side_effect = lambda *args, **kwargs: app.config.exit.set()
    monkeypatch.setattr(worker, 'worker_pool', pool)
    monkeypatch.setattr(worker, 'apply_update_handler_alter', lambda handler, *args: handler)
    monkeypatch.setattr(worker, 'apply_update_finalize', lambda **kwargs: None)
    monkeypatch.setattr(worker, 'cleanup_error_artifacts', lambda *args: None)

    for outcome in ('fail', 'fail', 'fail', 'unchanged', 'unchanged'):
        app.config.exit.clear()
        handler = MagicMock(xpath_data=None)
        handler.fetcher.favicon_blob = None
        handler.fetcher.headers = {}
        handler.fetcher.status_code = 200
        handler.fetcher.content = ''
        handler.fetcher.quit = AsyncMock()
        if outcome == 'fail':
            handler.call_browser = AsyncMock(side_effect=PageUnloadable(url=watch['url'], status_code=None,
                                            message='net::ERR_HTTP2_PROTOCOL_ERROR'))
        else:
            handler.call_browser = AsyncMock()
            handler.run_changedetection.side_effect = checksumFromPreviousCheckWasTheSame()
        monkeypatch.setattr(processors, 'get_processor_module', lambda name: SimpleNamespace(perform_site_check=lambda **kwargs: handler))
        queue = SimpleNamespace(async_get=AsyncMock(return_value=SimpleNamespace(item={'uuid': uuid}, priority=1)))
        asyncio.run(worker.async_update_worker(99, queue, notifications, app, datastore))
    assert notifications.qsize() == 2  # one failure, then one recovery; no duplicates
    assert 'failing' in notifications.get()['notification_title']
    assert 'recovered' in notifications.get()['notification_title']
    assert watch['consecutive_check_failures'] == 0
    assert not watch['check_failure_alerted']
    assert not watch.was_edited
