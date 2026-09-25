from flask_babel import lazy_gettext as _l
from loguru import logger
import asyncio
import hashlib
import json
import os

from changedetectionio import strtobool
from changedetectionio.content_fetchers import SCREENSHOT_MAX_HEIGHT_DEFAULT, visualselector_xpath_selectors, \
    XPATH_ELEMENT_JS, INSTOCK_DATA_JS, FAVICON_FETCHER_JS
from changedetectionio.content_fetchers.base import Fetcher
from changedetectionio.content_fetchers.exceptions import BrowserStepsInUnsupportedFetcher, EmptyReply, Non200ErrorCodeReceived, PageUnloadable
from changedetectionio.content_fetchers.scrapling_http import decode_body
from changedetectionio.validate_url import is_fetch_url_allowed, is_url_private_or_parser_confused


# Uses Scrapling's StealthyFetcher (https://github.com/D4Vinci/Scrapling) - a patched Chromium (patchright) with
# fingerprint hardening that can also get through Cloudflare's Turnstile/interstitial challenges.
#
# By default a local Chromium is launched per check (install it with `patchright install --with-deps chromium`),
# or set SCRAPLING_CDP_URL to drive an already running Chrome over CDP (some stealth is lost that way, since the
# browser binary itself is then not the patched one).
class fetcher(Fetcher):
    fetcher_description = _l("Scrapling Stealth Chrome - anti-bot/Cloudflare bypass")

    supports_browser_steps = False
    supports_screenshots = True
    supports_xpath_element_data = True

    proxy = None

    @classmethod
    def get_status_icon_data(cls):
        return {
            'filename': 'google-chrome-icon.png',
            'alt': 'Using a Scrapling stealth Chrome browser',
            'title': 'Using a Scrapling stealth Chrome browser'
        }

    def __init__(self, proxy_override=None, custom_browser_connection_url=None, **kwargs):
        super().__init__(**kwargs)
        self.proxy = proxy_override
        self.browser_connection_url = (custom_browser_connection_url or os.getenv('SCRAPLING_CDP_URL', '')).strip('"') or None
        self.browser_connection_is_custom = bool(custom_browser_connection_url)

    async def run(self,
                  fetch_favicon=True,
                  current_include_filters=None,
                  empty_pages_are_a_change=False,
                  ignore_status_codes=False,
                  is_binary=False,
                  request_body=None,
                  request_headers=None,
                  request_method=None,
                  screenshot_format=None,
                  timeout=None,
                  url=None,
                  watch_uuid=None,
                  ):

        if self.browser_steps:
            raise BrowserStepsInUnsupportedFetcher(url=url)

        from requests.structures import CaseInsensitiveDict
        from scrapling.fetchers import StealthyFetcher
        from changedetectionio.content_fetchers.playwright import capture_full_page_async

        ok, reason = is_fetch_url_allowed(url)
        if not ok:
            raise Exception(reason)

        allow_iana_restricted = strtobool(os.getenv('ALLOW_IANA_RESTRICTED_ADDRESSES', 'false'))
        extra_wait = int(os.getenv("WEBDRIVER_DELAY_BEFORE_CONTENT_READY", 5)) + self.render_extract_delay

        # A custom User-Agent must go via `useragent`, otherwise Scrapling generates a real one matching the browser
        extra_headers = dict(request_headers or {})
        useragent = None
        for k in list(extra_headers.keys()):
            if k.lower() == 'user-agent':
                useragent = extra_headers.pop(k)

        # Results gathered from inside the live page, Scrapling swallows exceptions from page_action so keep them here
        captured = {}

        async def page_setup(page):
            page.on("console", lambda msg: logger.debug(f"Scrapling console: Watch URL: {url} {msg.type}: {msg.text}"))

            # Re-validate every navigation (redirects, meta-refresh, JS location changes) so an open redirect
            # on a public host can't be used to reach private/reserved addresses (SSRF)
            if not allow_iana_restricted:
                async def navigation_guard(route):
                    request = route.request
                    if request.is_navigation_request() and \
                            await asyncio.to_thread(is_url_private_or_parser_confused, request.url):
                        await route.abort('blockedbyclient')
                        # Only the page itself going somewhere private is an error. iframes (ads, trackers) are just
                        # dropped, DNS ad-blockers (Pi-hole etc) resolve those to 0.0.0.0 which counts as reserved.
                        if request.frame == page.main_frame:
                            captured['blocked_url'] = request.url
                        else:
                            logger.debug(f"Blocked iframe navigation to private/reserved address '{request.url}' on {url}")
                        return
                    await route.fallback()

                await page.route("**/*", navigation_guard)

        async def page_action(page):
            try:
                if self.webdriver_js_execute_code is not None and len(self.webdriver_js_execute_code.strip()):
                    await page.evaluate(self.webdriver_js_execute_code)

                await page.wait_for_timeout(extra_wait * 1000)

                # Don't extract while a navigation is mid-flight
                try:
                    await page.wait_for_load_state('load', timeout=extra_wait * 1000)
                except Exception as e:
                    logger.debug(f"Page did not reach a settled load state, continuing anyway: {e}")

                if fetch_favicon:
                    try:
                        captured['favicon_blob'] = await page.evaluate(FAVICON_FETCHER_JS)
                    except Exception as e:
                        logger.error(f"Error fetching FavIcon info {str(e)}, continuing.")

                await page.evaluate("var include_filters={}".format(json.dumps(current_include_filters) if current_include_filters is not None else "''"))
                captured['xpath_data'] = await page.evaluate(XPATH_ELEMENT_JS, {
                    "visualselector_xpath_selectors": visualselector_xpath_selectors,
                    "max_height": int(os.getenv("SCREENSHOT_MAX_HEIGHT", SCREENSHOT_MAX_HEIGHT_DEFAULT))
                })
                captured['instock_data'] = await page.evaluate(INSTOCK_DATA_JS)
                captured['screenshot'] = await capture_full_page_async(page=page, screenshot_format=self.screenshot_format,
                                                                       watch_uuid=watch_uuid, lock_viewport_elements=self.lock_viewport_elements)
            except Exception as e:
                captured['error'] = e

        fetch_kwargs = {
            'headless': True,
            'solve_cloudflare': strtobool(os.getenv('SCRAPLING_SOLVE_CLOUDFLARE', 'true')),
            'network_idle': strtobool(os.getenv('SCRAPLING_NETWORK_IDLE', 'false')),
            'block_webrtc': True,
            'google_search': strtobool(os.getenv('SCRAPLING_STEALTHY_HEADERS', 'true')),
            # Browser retries are expensive, the watch will be re-checked anyway
            'retries': int(os.getenv('SCRAPLING_BROWSER_RETRY_MAX_COUNT', '1')),
            'extra_headers': extra_headers,
            'page_setup': page_setup,
            'page_action': page_action,
            # Our own delay happens inside page_action (before the screenshot etc)
            'wait': 0,
            'additional_args': {'ignore_https_errors': True},
        }
        if timeout:
            fetch_kwargs['timeout'] = int(timeout) * 1000
        if useragent:
            fetch_kwargs['useragent'] = useragent
        if self.proxy:
            fetch_kwargs['proxy'] = self.proxy
        if self.browser_connection_url:
            fetch_kwargs['cdp_url'] = self.browser_connection_url

        try:
            r = await StealthyFetcher.async_fetch(url, **fetch_kwargs)
        except Exception as e:
            if captured.get('blocked_url'):
                raise Exception(f"Redirect blocked: '{captured['blocked_url']}' resolves to a private/reserved IP address "
                                f"or contains a parser-differential payload.") from e
            msg = str(e)
            if "Executable doesn't exist" in msg:
                msg = "Scrapling stealth browser is not installed, run 'patchright install --with-deps chromium' or set SCRAPLING_CDP_URL"
            raise PageUnloadable(url=url, status_code=None, message=msg) from e

        if captured.get('blocked_url'):
            raise Exception(f"Redirect blocked: '{captured['blocked_url']}' resolves to a private/reserved IP address "
                            f"or contains a parser-differential payload.")

        if captured.get('error'):
            logger.error(f"Scrapling stealth - error while processing page {url}: {captured['error']}")
            raise PageUnloadable(url=url, status_code=r.status, message=str(captured['error']))

        self.headers = CaseInsensitiveDict(r.headers)
        self.status_code = r.status
        self.favicon_blob = captured.get('favicon_blob')
        self.xpath_data = captured.get('xpath_data')
        self.instock_data = captured.get('instock_data')
        self.screenshot = captured.get('screenshot')

        raw_content = r.body
        if self.status_code != 200 and not ignore_status_codes:
            raise Non200ErrorCodeReceived(url=url, status_code=self.status_code, screenshot=self.screenshot)

        if not empty_pages_are_a_change and not raw_content.strip():
            logger.debug("Content Fetcher > Content was empty, empty_pages_are_a_change = False")
            raise EmptyReply(url=url, status_code=self.status_code)

        if is_binary:
            self.content = hashlib.md5(raw_content).hexdigest()
        else:
            # HTML comes back from the rendered DOM already as utf-8, anything else is the raw response body
            content_type = self.headers.get('content-type', '')
            self.content = raw_content.decode('utf-8', errors='replace') if 'html' in content_type \
                else decode_body(raw_content, content_type=content_type, url=url)
        self.raw_content = raw_content

    async def quit(self, watch=None):
        return


# Plugin registration for built-in fetcher
class ScraplingStealthFetcherPlugin:
    """Plugin class that registers the Scrapling stealth browser fetcher as a built-in plugin."""

    def register_content_fetcher(self):
        """Register the Scrapling stealth browser fetcher"""
        return ('html_scrapling_stealth', fetcher)


# Create module-level instance for plugin registration
scrapling_stealth_plugin = ScraplingStealthFetcherPlugin()
