#!/usr/bin/env python3
"""
Integration tests for the AI setup assistant routes: POST /edit/<uuid>/ai-assist/<kind>.

Snapshots are written straight into the watch (no fetch), and the LLM call is mocked, so this
exercises the route, the watch storage accessors and the validation - and checks the helper
never changes the watch itself.
"""

import json
from unittest.mock import patch

from flask import url_for

from changedetectionio.tests.util import delete_all_watches

PAGE = """<html><body>
<main id="product"><h1 itemprop="name">Widget</h1>
<span itemprop="price">$149.99</span><div class="viewers">3 people are viewing this</div></main>
</body></html>"""


def _setup(client, llm=True):
    datastore = client.application.config.get('DATASTORE')
    if llm:
        datastore.data['settings']['application']['llm'] = {'model': 'gpt-4o-mini', 'api_key': 'sk-test-fake'}
    uuid = datastore.add_watch(url='https://example.com/widget')
    watch = datastore.data['watching'][uuid]
    for i in range(5):
        ts = 1_700_000_000 + i * 60
        watch.save_history_blob(contents=f"Widget\n$149.99\n{i + 3} people are viewing this\n",
                                timestamp=ts, snapshot_id=f'id{i}')
    watch.save_last_fetched_html(timestamp=ts, contents=PAGE)
    return datastore, uuid, watch


def _post(client, uuid, kind, data=None):
    res = client.post(url_for('ui.ui_ai_assist.watch_ai_assist', uuid=uuid, kind=kind), data=data or {})
    return res.status_code, json.loads(res.data.decode('utf-8'))


def test_ai_assist_filters_and_noise(client, live_server, measure_memory_usage, datastore_path):
    datastore, uuid, watch = _setup(client)
    before = dict(watch)

    reply = json.dumps({'include_filters': ['[itemprop=price]', '#missing'], 'reason': 'itemprop'})
    with patch('changedetectionio.llm.client.completion', return_value=(reply, 10, 8, 2)):
        code, data = _post(client, uuid, 'filters', {'goal': 'the price'})
    assert code == 200, data
    assert [s['value'] for s in data['include_filters']] == ['[itemprop=price]']

    reply = json.dumps({'ignore_text': [{'pattern': 'people are viewing', 'why': 'counter'}]})
    with patch('changedetectionio.llm.client.completion', return_value=(reply, 10, 8, 2)):
        code, data = _post(client, uuid, 'noise')
    assert code == 200, data
    assert [a['value'] for a in data['ignore_text']] == ['people are viewing']
    assert data['alerts_avoided'] == data['diffs_checked'] == 4

    # Suggestions only - the watch is untouched.
    assert watch.get('include_filters') == before.get('include_filters')
    assert watch.get('ignore_text') == before.get('ignore_text')
    delete_all_watches(client)


def test_ai_assist_diagnose_uses_saved_error_html(client, live_server, measure_memory_usage, datastore_path):
    datastore, uuid, watch = _setup(client)
    watch['last_error'] = 'Warning, no filters were found, no change detection ran'
    watch['include_filters'] = ['.old-price']
    watch.save_error_html(PAGE)
    assert 'itemprop="price"' in watch.get_error_html()

    reply = json.dumps({'cause': 'layout_changed', 'diagnosis': 'Renamed.', 'fixes': ['x'],
                        'include_filters': ['[itemprop=price]']})
    with patch('changedetectionio.llm.client.completion', return_value=(reply, 10, 8, 2)):
        code, data = _post(client, uuid, 'diagnose')
    assert code == 200, data
    assert data['html_note'] == 'HTML of the failing fetch'
    assert [s['value'] for s in data['include_filters']] == ['[itemprop=price]']
    delete_all_watches(client)


def test_ai_assist_errors(client, live_server, measure_memory_usage, datastore_path):
    datastore, uuid, watch = _setup(client, llm=False)
    datastore.data['settings']['application']['llm'] = {}

    code, data = _post(client, uuid, 'tags')
    assert code == 400 and 'not configured' in data['error']

    code, data = _post(client, uuid, 'bogus')
    assert code == 404

    code, data = _post(client, 'a0b1c2d3-0000-0000-0000-000000000000', 'tags')
    assert code == 404

    datastore.data['settings']['application']['llm'] = {'model': 'gpt-4o-mini'}
    code, data = _post(client, uuid, 'diagnose')
    assert code == 400 and 'no current error' in data['error']
    delete_all_watches(client)


def test_edit_page_shows_the_assistant_only_when_configured(client, live_server, measure_memory_usage, datastore_path):
    datastore, uuid, watch = _setup(client)
    res = client.get(url_for('ui.ui_edit.edit_page', uuid=uuid))
    assert b'id="ai-assist"' in res.data
    assert b'ai-assist.js' in res.data

    datastore.data['settings']['application']['llm'] = {}
    res = client.get(url_for('ui.ui_edit.edit_page', uuid=uuid))
    assert b'id="ai-assist"' not in res.data
    delete_all_watches(client)
