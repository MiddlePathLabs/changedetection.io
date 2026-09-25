"""
Unit tests for llm/assist.py - the AI setup assistant on the watch edit page.

The LLM call is mocked; what is under test is that every suggestion is checked against the
real page / history before it reaches the user.
"""
import json
from unittest.mock import MagicMock, patch

import pytest

from changedetectionio.llm import assist

PRODUCT_HTML = """
<html><head><title>Widget 3000</title><script>var x = 1;</script><style>.a{}</style></head>
<body>
  <nav class="site-nav"><a href="/">Home</a></nav>
  <main id="product">
    <h1 itemprop="name">Widget 3000</h1>
    <div class="price-box"><span itemprop="price" data-testid="price">$149.99</span></div>
    <div class="css-1x2y3z">3 people are viewing this</div>
    <section class="related-items"><div>Other widget $99</div></section>
  </main>
  <footer>Copyright 2026</footer>
</body></html>
"""


class FakeWatch(dict):
    def __init__(self, snapshots=None, html=None, error_html=None, error_text=None, **fields):
        defaults = {'url': 'https://shop.example/widget', 'page_title': 'Widget 3000', 'tags': [],
                        'include_filters': [], 'subtractive_selectors': [], 'ignore_text': [],
                        'llm_intent': '', 'last_error': False, 'consecutive_filter_failures': 0,
                        'browser_steps_last_error_step': None}
        super().__init__({**defaults, **fields})
        self._snapshots = snapshots or {}
        self._html = html or {}
        self._error_html = error_html
        self._error_text = error_text

    @property
    def history(self):
        return dict.fromkeys(self._snapshots)

    def get_history_snapshot(self, timestamp=None):
        return self._snapshots.get(timestamp, '')

    def get_fetched_html(self, timestamp):
        return self._html.get(timestamp, False)

    def get_error_html(self):
        return self._error_html or False

    def get_error_text(self):
        return self._error_text or False

    def get_last_fetched_text_before_filters(self):
        return ''


def _datastore(tags=None, llm=None):
    ds = MagicMock()
    ds.data = {'settings': {'application': {
        'llm': llm if llm is not None else {'model': 'gpt-4o-mini'},
        'tags': tags or {},
    }}}
    return ds


def _reply(obj):
    return patch('changedetectionio.llm.client.completion', return_value=(json.dumps(obj), 100, 80, 20))


# ---------------------------------------------------------------------------
# Selector / rule validation
# ---------------------------------------------------------------------------

class TestCheckSelector:
    def test_css_match_and_sample(self):
        count, sample = assist.check_selector('[itemprop=price]', PRODUCT_HTML)
        assert count == 1 and sample == '$149.99'

    def test_xpath(self):
        count, sample = assist.check_selector('xpath://h1', PRODUCT_HTML)
        assert count == 1 and sample == 'Widget 3000'

    def test_no_match(self):
        assert assist.check_selector('#does-not-exist', PRODUCT_HTML) == (0, '')

    def test_invalid_selector_is_not_an_exception(self):
        assert assist.check_selector('div[[[', PRODUCT_HTML) == (0, '')


class TestLineRules:
    def test_plain_text_is_case_insensitive_contains(self):
        m = assist.compile_line_rule('People are viewing')
        assert m('3 people are viewing this')
        assert not m('Price $10')

    def test_regex(self):
        m = assist.compile_line_rule(r'/\d+ people/')
        assert m('12 people are viewing')

    def test_invalid_and_match_everything_rejected(self):
        assert assist.compile_line_rule('/[unclosed/') is None
        assert assist.compile_line_rule('/.*/') is None
        assert assist.compile_line_rule('   ') is None


class TestHtmlOutline:
    def test_drops_scripts_and_keeps_hooks(self):
        out = assist.html_outline(PRODUCT_HTML)
        assert 'var x' not in out
        assert 'itemprop="price"' in out and '$149.99' in out
        assert 'id="product"' in out


# ---------------------------------------------------------------------------
# 1. Filters
# ---------------------------------------------------------------------------

class TestSuggestFilters:
    def test_only_matching_stable_selectors_survive(self):
        watch = FakeWatch(snapshots={'100': 'x'}, html={'100': PRODUCT_HTML})
        reply = {
            'include_filters': ['[itemprop=price]', '#nope', 'div:nth-child(2)', '.css-1x2y3z'],
            'subtractive_selectors': ['.related-items', '.missing'],
            'ignore_text': ['people are viewing', '/[bad/'],
            'trigger_text': [],
            'reason': 'Price element has itemprop.',
        }
        with _reply(reply):
            r = assist.suggest_filters(watch, _datastore(), goal='the price')
        assert [s['value'] for s in r['include_filters']] == ['[itemprop=price]']
        assert r['include_filters'][0]['sample'] == '$149.99'
        assert [s['value'] for s in r['subtractive_selectors']] == ['.related-items']
        assert [s['value'] for s in r['ignore_text']] == ['people are viewing']
        assert r['based_on'] == '100'

    def test_falls_back_to_intent_for_goal(self):
        watch = FakeWatch(snapshots={'100': 'x'}, html={'100': PRODUCT_HTML}, llm_intent='price drops')
        with _reply({'include_filters': []}) as m:
            assist.suggest_filters(watch, _datastore(), goal='')
        assert 'price drops' in m.call_args.kwargs['messages'][1]['content']

    def test_needs_a_goal(self):
        watch = FakeWatch(snapshots={'100': 'x'}, html={'100': PRODUCT_HTML})
        with pytest.raises(assist.AssistError):
            assist.suggest_filters(watch, _datastore(), goal='')

    def test_needs_saved_html(self):
        with pytest.raises(assist.AssistError):
            assist.suggest_filters(FakeWatch(snapshots={'100': 'x'}), _datastore(), goal='price')

    def test_not_configured(self):
        watch = FakeWatch(snapshots={'100': 'x'}, html={'100': PRODUCT_HTML})
        with pytest.raises(assist.AssistError) as e:
            assist.suggest_filters(watch, _datastore(llm={}), goal='price')
        assert 'not configured' in e.value.message

    def test_unreadable_reply_is_a_502(self):
        watch = FakeWatch(snapshots={'100': 'x'}, html={'100': PRODUCT_HTML})
        with patch('changedetectionio.llm.client.completion', return_value=('sorry, no', 10, 5, 5)):
            with pytest.raises(assist.AssistError) as e:
                assist.suggest_filters(watch, _datastore(), goal='price')
        assert e.value.http_status == 502


# ---------------------------------------------------------------------------
# 2. Noise
# ---------------------------------------------------------------------------

def _noisy_history(n=6):
    snaps = {}
    for i in range(n):
        price = '$149.99' if i < 4 else '$129.99'
        snaps[str(100 + i)] = (
            f"Widget 3000\nPrice {price}\n{i + 3} people are viewing this\n"
            f"Updated {i + 1} minutes ago\nFooter text\n"
        )
    return snaps


class TestNoiseCandidates:
    def test_digits_are_folded_and_counted_per_diff(self):
        snaps = _noisy_history()
        cands, per_diff = assist.noise_candidates(list(snaps.values()))
        shapes = {c['shape']: c['changes'] for c in cands}
        assert shapes['# people are viewing this'] == 5
        assert shapes['Updated # minutes ago'] == 5
        assert len(per_diff) == 5


class TestSuggestNoise:
    def test_valid_rules_kept_and_alerts_avoided_counted(self):
        watch = FakeWatch(snapshots=_noisy_history())
        reply = {'ignore_text': [
            {'pattern': 'people are viewing', 'why': 'live counter'},
            {'pattern': r'/Updated \d+ minutes ago/', 'why': 'relative time'},
            {'pattern': 'something not on the page', 'why': 'hallucinated'},
            {'pattern': 'e', 'why': 'far too broad'},
        ]}
        with _reply(reply):
            r = assist.suggest_noise(watch, _datastore())
        assert [a['value'] for a in r['ignore_text']] == ['people are viewing', r'/Updated \d+ minutes ago/']
        # 5 diffs; the one where the price changed is still reported.
        assert r['diffs_checked'] == 5
        assert r['alerts_avoided'] == 4

    def test_rule_that_would_also_hide_the_price_is_rejected(self):
        """'/\\d+/' hits the noisy counter lines, but also the price line - which changed only
        once and is exactly what the user is watching."""
        watch = FakeWatch(snapshots=_noisy_history())
        with _reply({'ignore_text': [{'pattern': r'/\d+/'}, {'pattern': 'people are viewing'}]}):
            r = assist.suggest_noise(watch, _datastore())
        assert [a['value'] for a in r['ignore_text']] == ['people are viewing']

    def test_short_pages_are_fine(self):
        """A 3-line page where one line is noise: that line is 33% of the page, still valid."""
        snaps = {str(i): f"Widget\n$149.99\n{i + 3} people are viewing this\n" for i in range(5)}
        with _reply({'ignore_text': [{'pattern': 'people are viewing'}]}):
            r = assist.suggest_noise(FakeWatch(snapshots=snaps), _datastore())
        assert [a['value'] for a in r['ignore_text']] == ['people are viewing']
        assert r['alerts_avoided'] == 4

    def test_already_ignored_rules_not_suggested_again(self):
        watch = FakeWatch(snapshots=_noisy_history(), ignore_text=['People are viewing'])
        with _reply({'ignore_text': [{'pattern': 'people are viewing'}]}):
            r = assist.suggest_noise(watch, _datastore())
        assert r['ignore_text'] == []

    def test_needs_three_snapshots(self):
        watch = FakeWatch(snapshots={'1': 'a', '2': 'b'})
        with pytest.raises(assist.AssistError):
            assist.suggest_noise(watch, _datastore())

    def test_no_repeat_changes_skips_the_llm(self):
        # Every check adds a different line; nothing changes twice.
        watch = FakeWatch(snapshots={'1': 'a\n', '2': 'a\nb\n', '3': 'a\nb\nc\n'})
        with patch('changedetectionio.llm.client.completion') as m:
            r = assist.suggest_noise(watch, _datastore())
        m.assert_not_called()
        assert r['ignore_text'] == []


# ---------------------------------------------------------------------------
# 3. Diagnose
# ---------------------------------------------------------------------------

class TestDiagnose:
    def test_no_error_is_refused(self):
        with pytest.raises(assist.AssistError):
            assist.diagnose_error(FakeWatch(), _datastore())

    def test_lost_filter_gets_validated_replacement_from_error_html(self):
        watch = FakeWatch(
            snapshots={'100': 'x'},
            error_html=PRODUCT_HTML,
            last_error='Warning, no filters were found, no change detection ran',
            consecutive_filter_failures=2,
            include_filters=['.old-price'],
        )
        reply = {'cause': 'layout_changed', 'diagnosis': 'The price element was renamed.',
                 'fixes': ['Replace the filter'], 'include_filters': ['[data-testid=price]', '.old-price']}
        with _reply(reply) as m:
            r = assist.diagnose_error(watch, _datastore())
        assert r['cause'] == 'layout_changed'
        assert [s['value'] for s in r['include_filters']] == ['[data-testid=price]']
        assert r['html_note'] == 'HTML of the failing fetch'
        assert '.old-price' in m.call_args.kwargs['messages'][1]['content']

    def test_blocked_page_uses_error_text_and_offers_no_selector(self):
        watch = FakeWatch(last_error='Error - 403 (Access denied) received',
                          error_text='Access Denied. Please verify you are human.')
        reply = {'cause': 'blocked', 'diagnosis': 'Bot protection.', 'fixes': ['Use a browser fetcher'],
                 'include_filters': ['#x']}
        with _reply(reply) as m:
            r = assist.diagnose_error(watch, _datastore())
        assert r['cause'] == 'blocked'
        assert r['include_filters'] == []
        assert 'verify you are human' in m.call_args.kwargs['messages'][1]['content']


# ---------------------------------------------------------------------------
# 4. Tags
# ---------------------------------------------------------------------------

class TestSuggestTags:
    def test_existing_names_canonicalised_and_new_names_checked(self):
        tags = {'u1': {'title': 'Electronics'}, 'u2': {'title': 'Job Boards'}, 'u3': {'title': 'Deals'}}
        watch = FakeWatch(snapshots={'100': 'Widget 3000 $149.99'}, tags=['u3'])
        reply = {'existing': ['electronics', 'Made Up', 'Deals'], 'new': ['Gadgets, and more', 'Second']}
        with _reply(reply):
            r = assist.suggest_tags(watch, _datastore(tags=tags))
        # 'Deals' is already on the watch, 'Made Up' does not exist.
        assert r['existing'] == ['Electronics']
        assert r['new'] == ['Gadgets and more']

    def test_new_name_that_already_exists_is_dropped(self):
        tags = {'u1': {'title': 'Electronics'}}
        watch = FakeWatch(snapshots={'100': 'x'})
        with _reply({'existing': [], 'new': ['ELECTRONICS']}):
            r = assist.suggest_tags(watch, _datastore(tags=tags))
        assert r['new'] == []
