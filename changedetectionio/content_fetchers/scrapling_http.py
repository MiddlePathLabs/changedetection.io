from flask_babel import lazy_gettext as _l
from loguru import logger
from urllib.parse import urljoin
import codecs
import hashlib
import os
import re

from changedetectionio import strtobool
from changedetectionio.content_fetchers.exceptions import BrowserStepsInUnsupportedFetcher, EmptyReply, Non200ErrorCodeReceived
from changedetectionio.content_fetchers.requests import fetcher as requests_fetcher, sniff_encoding
from changedetectionio.validate_url import is_fetch_url_allowed, is_url_private_or_parser_confused

REDIRECT_STATUS_CODES = (301, 302, 303, 307, 308)
CHARSET_RE = re.compile(r'charset=["\']?([\w.:-]+)', re.IGNORECASE)


def scrapling_is_available():
    """Scrapling is optional, only offer this fetcher when it (and everything its HTTP side imports) is installed.

    Even the HTTP-only code path imports playwright and patchright at module level.
    """
    from importlib.util import find_spec
    return all(find_spec(m) for m in ('scrapling', 'curl_cffi', 'browserforge', 'playwright', 'patchright'))


def scrapling_stealth_is_available():
    """The stealth browser side of Scrapling needs a few more of its [fetchers] extras"""
    from importlib.util import find_spec
    return scrapling_is_available() and all(find_spec(m) for m in ('msgspec', 'anyio', 'protego'))


def decode_body(raw_content, content_type, url=None):
    """Bytes -> str using the Content-Type charset, or sniffing when there is none"""
    charset_match = CHARSET_RE.search(content_type or '')
    encoding = charset_match.group(1) if charset_match else sniff_encoding(content=raw_content, content_type=content_type, url=url)
    try:
        codecs.lookup(encoding or 'utf-8')
    except LookupError:
        logger.warning(f"URL: {url} Unknown encoding '{encoding}', falling back to utf-8")
        encoding = 'utf-8'
    return raw_content.decode(encoding or 'utf-8', errors='replace')


# Uses https://github.com/D4Vinci/Scrapling (curl_cffi underneath) - the TLS/HTTP2 fingerprint and headers
# look like a real browser, which gets past a lot of the basic anti-bot checks that block `requests`,
# without the cost of running a browser. Subclasses the requests fetcher for __init__/quit() behaviour.
class fetcher(requests_fetcher):
    fetcher_description = _l("Scrapling - Fast HTTP Client with browser TLS fingerprint")

    # Which browser TLS fingerprint to present, see curl_cffi's BrowserType for the list ('chrome', 'firefox', 'safari' etc)
    impersonate = os.getenv('SCRAPLING_IMPERSONATE', 'chrome')

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

        # curl has no business with local files, the requests fetcher already handles ALLOW_FILE_URI
        if url.startswith('file://'):
            return await super().run(url=url, timeout=timeout, request_headers=request_headers, request_body=request_body,
                                     request_method=request_method, ignore_status_codes=ignore_status_codes,
                                     current_include_filters=current_include_filters, is_binary=is_binary,
                                     empty_pages_are_a_change=empty_pages_are_a_change, watch_uuid=watch_uuid)

        from curl_cffi.curl import CurlError
        from requests.structures import CaseInsensitiveDict
        from scrapling.fetchers import FetcherSession

        proxy = None
        proxies = {}
        if self.proxy_override:
            proxy = self.proxy_override
        else:
            if self.system_http_proxy:
                proxies['http'] = self.system_http_proxy
            if self.system_https_proxy:
                proxies['https'] = self.system_https_proxy

        request_method = (request_method or 'GET').upper()
        allow_iana_restricted = strtobool(os.getenv('ALLOW_IANA_RESTRICTED_ADDRESSES', 'false'))

        session_kwargs = {
            'impersonate': self.impersonate,
            # Adds a real-browser set of headers (and a google.com referer), user supplied headers always win
            'stealthy_headers': strtobool(os.getenv('SCRAPLING_STEALTHY_HEADERS', 'true')),
            'retries': int(os.getenv('SCRAPLING_RETRY_MAX_COUNT', '3')),
            # Redirects are followed manually below so each hop can be validated (SSRF)
            'follow_redirects': False,
            'verify': False,
            'proxy': proxy,
            'proxies': proxies,
        }
        if timeout:
            session_kwargs['timeout'] = timeout

        async def _request(session, method, target_url, body=None):
            kwargs = {'headers': dict(request_headers or {})}
            if body:
                kwargs['data'] = body.encode('utf-8') if type(body) is str else body
            method_fn = getattr(session, method.lower(), None)
            if method_fn:
                return await method_fn(target_url, **kwargs)
            # PATCH/OPTIONS etc have no public helper in Scrapling
            return await session._make_request(method, url=target_url, **kwargs)

        try:
            # Fresh DNS check at fetch time — catches DNS rebinding regardless of add-time cache.
            ok, reason = is_fetch_url_allowed(url)
            if not ok:
                raise Exception(reason)

            async with FetcherSession(**session_kwargs) as session:
                r = await _request(session, request_method, url, body=request_body)

                # Manually follow redirects so each hop's resolved IP can be validated,
                # preventing SSRF via an open redirect on a public host.
                current_url = url
                for _ in range(10):
                    location = r.headers.get('location') or r.headers.get('Location')
                    if r.status not in REDIRECT_STATUS_CODES or not location:
                        break
                    redirect_url = urljoin(current_url, location)
                    if not allow_iana_restricted:
                        if is_url_private_or_parser_confused(redirect_url):
                            raise Exception(f"Redirect blocked: '{redirect_url}' resolves to a private/reserved IP address "
                                            f"or contains a parser-differential payload.")
                    current_url = redirect_url
                    r = await _request(session, 'GET', redirect_url)
                else:
                    raise Exception("Too many redirects")

        except CurlError as e:
            msg = str(e)
            if proxy or proxies:
                msg = f"Proxy connection failed? {msg}"
            raise Exception(msg) from e

        self.headers = CaseInsensitiveDict(r.headers)
        raw_content = r.body

        text = ''
        if not is_binary and raw_content:
            text = decode_body(raw_content, content_type=self.headers.get('content-type', ''), url=url)

        if not raw_content:
            logger.debug(f"Scrapling returned empty content for '{url}'")
            if not empty_pages_are_a_change:
                raise EmptyReply(url=url, status_code=r.status)
            else:
                logger.debug(f"URL {url} gave zero byte content reply with Status Code {r.status}, but empty_pages_are_a_change = True")

        if r.status != 200 and not ignore_status_codes:
            raise Non200ErrorCodeReceived(url=url, status_code=r.status, page_html=text)

        self.status_code = r.status
        if is_binary:
            # Binary files just return their checksum until we add something smarter
            self.content = hashlib.md5(raw_content).hexdigest()
        else:
            self.content = text

        self.raw_content = raw_content

        # If the content is an image, set it as screenshot for SSIM/visual comparison
        content_type = self.headers.get('content-type', '').lower()
        if 'image/' in content_type:
            self.screenshot = raw_content
            logger.debug(f"Image content detected ({content_type}), set as screenshot for comparison")


# Plugin registration for built-in fetcher
class ScraplingHttpFetcherPlugin:
    """Plugin class that registers the Scrapling HTTP fetcher as a built-in plugin."""

    def register_content_fetcher(self):
        """Register the Scrapling HTTP fetcher"""
        return ('html_scrapling', fetcher)


# Create module-level instance for plugin registration
scrapling_http_plugin = ScraplingHttpFetcherPlugin()
