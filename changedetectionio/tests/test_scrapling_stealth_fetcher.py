#!/usr/bin/env python3

import os
import pytest
from flask import url_for
from .util import set_original_response, set_modified_response, wait_for_all_checks, delete_all_watches

from changedetectionio.content_fetchers.scrapling_http import scrapling_stealth_is_available


def _stealth_browser_available():
    if os.getenv('SCRAPLING_CDP_URL'):
        return True
    if not scrapling_stealth_is_available():
        return False
    try:
        from patchright.sync_api import sync_playwright
        with sync_playwright() as p:
            return os.path.exists(p.chromium.executable_path)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _stealth_browser_available(),
                                reason="Scrapling stealth browser not installed (patchright install chromium) and no SCRAPLING_CDP_URL")

# Keep the tests quick, the default is a 5 second settle delay
os.environ['WEBDRIVER_DELAY_BEFORE_CONTENT_READY'] = '0'


def _get_watch(client, uuid):
    return client.application.config.get('DATASTORE').data['watching'][uuid]


def test_scrapling_stealth_change_detection(client, live_server, measure_memory_usage, datastore_path):
    set_original_response(datastore_path=datastore_path)
    test_url = url_for('test_endpoint', _external=True)
    uuid = client.application.config.get('DATASTORE').add_watch(url=test_url, extras={'fetch_backend': 'html_scrapling_stealth'})
    client.post(url_for("ui.form_watch_checknow"), follow_redirects=True)
    wait_for_all_checks(client)

    watch = _get_watch(client, uuid)
    assert not watch.get('last_error')
    res = client.get(url_for("ui.ui_preview.preview_page", uuid=uuid), follow_redirects=True)
    assert b'Which is across multiple lines' in res.data

    # Browser fetcher extras - screenshot and visual-selector element data
    assert watch.get_screenshot()
    assert os.path.isfile(os.path.join(watch.data_dir, 'elements.deflate'))

    set_modified_response(datastore_path=datastore_path)
    client.post(url_for("ui.form_watch_checknow"), follow_redirects=True)
    wait_for_all_checks(client)

    res = client.get(url_for("watchlist.index"))
    assert b'has-unread-changes' in res.data
    res = client.get(url_for("ui.ui_edit.watch_get_latest_html", uuid=uuid))
    assert b'which has this one new line' in res.data

    delete_all_watches(client)


def test_scrapling_stealth_headers_and_js(client, live_server, measure_memory_usage, datastore_path):
    test_url = url_for('test_headers', _external=True)
    uuid = client.application.config.get('DATASTORE').add_watch(url=test_url, extras={
        'fetch_backend': 'html_scrapling_stealth',
        'headers': {'xxx': 'ooo'},
        'webdriver_js_execute_code': 'document.body.innerHTML += "<p>js-was-executed</p>"',
    })
    client.post(url_for("ui.form_watch_checknow"), follow_redirects=True)
    wait_for_all_checks(client)

    assert not _get_watch(client, uuid).get('last_error')
    res = client.get(url_for("ui.ui_preview.preview_page", uuid=uuid), follow_redirects=True)
    assert b"Xxx:ooo" in res.data
    assert b"js-was-executed" in res.data
    assert b"HeadlessChrome" not in res.data

    delete_all_watches(client)


def test_scrapling_stealth_http_error(client, live_server, measure_memory_usage, datastore_path):
    with open(os.path.join(datastore_path, "endpoint-content.txt"), "w") as f:
        f.write("Now you going to get a 404 error code\n")

    test_url = url_for('test_endpoint', status_code=404, _external=True)
    uuid = client.application.config.get('DATASTORE').add_watch(url=test_url, extras={'fetch_backend': 'html_scrapling_stealth'})
    client.post(url_for("ui.form_watch_checknow"), follow_redirects=True)
    wait_for_all_checks(client)

    res = client.get(url_for("watchlist.index"))
    assert b'has-unread-changes' not in res.data
    assert b'Page not found' in res.data
    # The error screenshot is kept from the browser
    assert _get_watch(client, uuid).get_error_snapshot()

    delete_all_watches(client)
