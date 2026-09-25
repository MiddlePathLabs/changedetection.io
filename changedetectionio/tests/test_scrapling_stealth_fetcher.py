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


def test_scrapling_stealth_navigation_guard(client, live_server, measure_memory_usage, datastore_path, monkeypatch):
    """Navigations to private/reserved addresses are blocked, but only the main page doing it is an error.
    iframes (ads etc) that a DNS ad-blocker resolves to 0.0.0.0 must not fail the whole check."""
    import asyncio
    from changedetectionio.content_fetchers import scrapling_stealth

    # The test server is on localhost which is always "private", so pretend only the marked URLs are
    monkeypatch.setenv('ALLOW_IANA_RESTRICTED_ADDRESSES', 'false')
    monkeypatch.setattr(scrapling_stealth, 'is_fetch_url_allowed', lambda u: (True, ''))
    from urllib.parse import urlparse
    monkeypatch.setattr(scrapling_stealth, 'is_url_private_or_parser_confused', lambda u: urlparse(u).path == '/test-endpoint2')

    private_url = url_for('test_endpoint2', _external=True)

    def run(html):
        f = scrapling_stealth.fetcher()
        asyncio.run(f.run(url=url_for('test_endpoint', content=html, _external=True), request_headers={}, timeout=30))
        return f

    # iframe to a "private" address - dropped, page still fetched fine
    f = run(f'<html><body><p>main-page-content</p><iframe src="{private_url}"></iframe></body></html>')
    assert f.status_code == 200
    assert 'main-page-content' in f.content

    # The main page navigating to a "private" address is still refused
    with pytest.raises(Exception, match="Redirect blocked"):
        run(f'<html><body>redirecting<script>location.href="{private_url}"</script></body></html>')


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
