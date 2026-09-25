#!/usr/bin/env python3

import os
import pytest
from flask import url_for
from .util import set_original_response, set_modified_response, wait_for_all_checks, delete_all_watches

from changedetectionio.content_fetchers.scrapling_http import scrapling_is_available

pytestmark = pytest.mark.skipif(not scrapling_is_available(), reason="Scrapling (and its playwright/patchright imports) not installed")


def _get_watch(client, uuid):
    return client.application.config.get('DATASTORE').data['watching'][uuid]


def test_scrapling_fetcher_is_offered(client, live_server, measure_memory_usage, datastore_path):
    from changedetectionio import content_fetchers
    assert 'html_scrapling' in [name for name, description in content_fetchers.available_fetchers()]


def test_scrapling_basic_change_detection(client, live_server, measure_memory_usage, datastore_path):
    set_original_response(datastore_path=datastore_path)
    test_url = url_for('test_endpoint', _external=True)
    uuid = client.application.config.get('DATASTORE').add_watch(url=test_url, extras={'fetch_backend': 'html_scrapling'})
    client.post(url_for("ui.form_watch_checknow"), follow_redirects=True)
    wait_for_all_checks(client)

    assert not _get_watch(client, uuid).get('last_error')
    res = client.get(url_for("ui.ui_preview.preview_page", uuid=uuid), follow_redirects=True)
    assert b'Which is across multiple lines' in res.data

    set_modified_response(datastore_path=datastore_path)
    client.post(url_for("ui.form_watch_checknow"), follow_redirects=True)
    wait_for_all_checks(client)

    res = client.get(url_for("watchlist.index"))
    assert b'has-unread-changes' in res.data
    res = client.get(url_for("ui.ui_edit.watch_get_latest_html", uuid=uuid))
    assert b'which has this one new line' in res.data

    delete_all_watches(client)


def test_scrapling_headers_and_impersonation(client, live_server, measure_memory_usage, datastore_path):
    test_url = url_for('test_headers', _external=True)
    uuid = client.application.config.get('DATASTORE').add_watch(url=test_url, extras={
        'fetch_backend': 'html_scrapling',
        'headers': {'xxx': 'ooo', 'cool': 'yeah'},
    })
    client.post(url_for("ui.form_watch_checknow"), follow_redirects=True)
    wait_for_all_checks(client)

    assert not _get_watch(client, uuid).get('last_error')
    res = client.get(url_for("ui.ui_preview.preview_page", uuid=uuid), follow_redirects=True)
    # Custom headers are sent
    assert b"Xxx:ooo" in res.data
    assert b"Cool:yeah" in res.data
    # And the User-Agent comes from the impersonated browser, not a python HTTP library
    assert b"User-Agent:Mozilla/5.0" in res.data
    assert b"python-requests" not in res.data
    # Response headers are recorded (case-insensitive lookup)
    assert 'custom' in _get_watch(client, uuid).get('remote_server_reply')

    delete_all_watches(client)


@pytest.mark.parametrize("method", ["POST", "PATCH"])
def test_scrapling_request_method_and_body(client, live_server, measure_memory_usage, datastore_path, method):
    # PATCH has no public helper in Scrapling so it goes via the generic request path
    test_url = url_for('test_method' if method == 'PATCH' else 'test_body', _external=True)
    uuid = client.application.config.get('DATASTORE').add_watch(url=test_url, extras={
        'fetch_backend': 'html_scrapling',
        'method': method,
        'body': 'Test Body Value {{ 1+1 }}',
    })
    client.post(url_for("ui.form_watch_checknow"), follow_redirects=True)
    wait_for_all_checks(client)

    assert not _get_watch(client, uuid).get('last_error')
    res = client.get(url_for("ui.ui_preview.preview_page", uuid=uuid), follow_redirects=True)
    if method == 'PATCH':
        assert b'PATCH' in res.data
    else:
        assert b'Test Body Value 2' in res.data

    delete_all_watches(client)


def test_scrapling_http_error(client, live_server, measure_memory_usage, datastore_path):
    with open(os.path.join(datastore_path, "endpoint-content.txt"), "w") as f:
        f.write("Now you going to get a 404 error code\n")

    test_url = url_for('test_endpoint', status_code=404, _external=True)
    client.application.config.get('DATASTORE').add_watch(url=test_url, extras={'fetch_backend': 'html_scrapling'})
    client.post(url_for("ui.form_watch_checknow"), follow_redirects=True)
    wait_for_all_checks(client)

    res = client.get(url_for("watchlist.index"))
    assert b'has-unread-changes' not in res.data
    assert b'Page not found' in res.data

    delete_all_watches(client)


def test_scrapling_encoding_sniffing(client, live_server, measure_memory_usage, datastore_path):
    # No charset in the Content-Type header, the <meta charset> must be used to decode
    content = '<html><head><meta charset="iso-8859-1"></head><body>Caf\xe9 cr\xe8me</body></html>'.encode('iso-8859-1')
    with open(os.path.join(datastore_path, "endpoint-content.txt"), "wb") as f:
        f.write(content)

    test_url = url_for('test_endpoint', content_type='text/html', _external=True)
    uuid = client.application.config.get('DATASTORE').add_watch(url=test_url, extras={'fetch_backend': 'html_scrapling'})
    client.post(url_for("ui.form_watch_checknow"), follow_redirects=True)
    wait_for_all_checks(client)

    assert not _get_watch(client, uuid).get('last_error')
    res = client.get(url_for("ui.ui_preview.preview_page", uuid=uuid), follow_redirects=True)
    assert 'Café crème'.encode('utf-8') in res.data

    delete_all_watches(client)
