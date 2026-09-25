"""
AI setup assistant for a watch - on-demand suggestions the user reviews and applies.

Four helpers, each triggered by an explicit click on the watch edit page (never by the
worker, so they cost tokens only when someone asks):

  suggest_filters(watch, datastore, goal)  CSS/XPath include filters, elements to remove,
                                           ignore/trigger text for what the user wants to watch
  suggest_noise(watch, datastore)          ignore_text rules for lines that keep changing
                                           without meaning (timestamps, counters, tokens)
  diagnose_error(watch, datastore)         plain-English cause of the watch's current error,
                                           with fixes and (for a lost filter) a replacement
  suggest_tags(watch, datastore)           groups this watch belongs in

Nothing here writes to the watch. Every suggestion is validated against the real page or
history before it is returned (a selector must match, a regex must compile and must not
swallow the page), because an unchecked suggestion that looks plausible is worse than none.
"""

import re
from collections import Counter

from loguru import logger

from . import client as llm_client
from .bm25_trim import trim_to_relevant
from .diff_text import build_llm_diff
from .response_parser import parse_json_object

# Page outline sent to the model. Big enough for a real product/listing page after
# scripts, styles and attribute noise are stripped; trimmed by relevance past that.
OUTLINE_MAX_CHARS = 30_000
MAX_SUGGESTIONS_PER_LIST = 3
NOISE_MAX_SNAPSHOTS = 12
NOISE_MAX_CANDIDATES = 40
ASSIST_MAX_TOKENS = 800

_POSITIONAL_SELECTOR_RE = re.compile(r'nth-child|nth-of-type|:eq\(|\[\d+\]', re.IGNORECASE)
# Build-tool generated class names (css-1x2y3z, sc-AxjAm, jsx-123456, _3xY9z) change on redeploy.
_HASHED_CLASS_RE = re.compile(r'\.(?:css|sc|jsx|emotion|styled)-[\w-]*\d|\._[a-zA-Z0-9]{5,}\b')

_DROP_TAGS = ('script', 'style', 'noscript', 'svg', 'template', 'iframe', 'canvas', 'link', 'meta', 'path')
_KEEP_ATTRS = ('id', 'class', 'itemprop', 'itemtype', 'data-testid', 'data-test', 'data-qa', 'role',
               'aria-label', 'name', 'property')


class AssistError(Exception):
    """A handled failure with the HTTP status the route should answer with."""

    def __init__(self, message, http_status=400):
        super().__init__(message)
        self.message = message
        self.http_status = http_status


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_json(datastore, system_prompt: str, user_prompt: str) -> dict:
    from .evaluator import (
        _cached_system,
        _runtime_llm_config,
        _thinking_extra_body,
        accumulate_global_tokens,
        apply_local_token_multiplier,
        get_llm_settings,
        is_global_token_budget_exceeded,
        resolve_llm_timeout,
    )

    cfg = _runtime_llm_config(datastore)
    if not cfg:
        raise AssistError('AI / LLM is not configured, or is switched off in settings.')
    if is_global_token_budget_exceeded(datastore):
        raise AssistError('Monthly AI token budget reached. Resets next month.', http_status=429)

    settings = get_llm_settings(datastore)
    try:
        raw, tokens, input_tokens, output_tokens = (tuple(llm_client.completion(
            model=cfg['model'],
            messages=[
                _cached_system(system_prompt, model=cfg['model']),
                {'role': 'user', 'content': user_prompt},
            ],
            api_key=cfg.get('api_key'),
            api_base=cfg.get('api_base'),
            timeout=resolve_llm_timeout(cfg),
            max_tokens=apply_local_token_multiplier(ASSIST_MAX_TOKENS, cfg),
            extra_body=_thinking_extra_body(cfg['model'], settings.thinking_budget),
            debug=settings.debug,
        )) + (0, 0))[:4]
    except Exception as e:
        from changedetectionio.blueprint.ui.diff import _clean_litellm_error
        logger.warning(f"AI assist call failed: {e}")
        raise AssistError(f'AI request failed: {_clean_litellm_error(e)}', http_status=502) from e

    accumulate_global_tokens(datastore, tokens, input_tokens=input_tokens,
                             output_tokens=output_tokens, model=cfg['model'])
    try:
        return parse_json_object(raw)
    except ValueError:
        logger.warning(f"AI assist: unusable reply {(raw or '')[:300]!r}")
        raise AssistError('The AI reply could not be read. Try again, or try a different model.',
                          http_status=502) from None


def _str_list(value, limit=MAX_SUGGESTIONS_PER_LIST) -> list[str]:
    """Coerce a model-supplied list into unique, non-empty strings."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = []
    for v in value:
        if isinstance(v, dict):
            v = v.get('pattern') or v.get('selector') or v.get('value') or ''
        v = str(v).strip() if v is not None else ''
        if v and v not in out:
            out.append(v)
    return out[:limit]


# ---------------------------------------------------------------------------
# Page access and outline
# ---------------------------------------------------------------------------

def latest_fetched_html(watch) -> tuple[str, str]:
    """(html, timestamp) of the newest saved raw HTML, or ('', '') when none is kept."""
    for ts in reversed(list(watch.history.keys())):
        try:
            html = watch.get_fetched_html(ts)
        except Exception:
            html = False
        if html:
            return html, ts
    return '', ''


def _parse_html(html: str):
    import lxml.html
    try:
        return lxml.html.fromstring(html)
    except Exception as e:
        raise AssistError(f'Could not parse the saved page HTML: {e}') from e


def html_outline(html: str, focus: str = '', max_chars: int = OUTLINE_MAX_CHARS) -> str:
    """A compact, one-element-per-line view of the page for choosing selectors.

    Keeps only the attributes a stable selector is built from, and a short slice of each
    element's own text, so the model sees structure and content side by side without the
    megabytes of markup. Trimmed by relevance to `focus` when still too long.
    """
    root = _parse_html(html)
    for bad in root.xpath('//' + ' | //'.join(_DROP_TAGS)):
        bad.drop_tree()
    for comment in root.xpath('//comment()'):
        comment.drop_tree()

    lines = []

    def walk(el, depth):
        if not isinstance(el.tag, str):
            return
        attrs = []
        for a in _KEEP_ATTRS:
            v = el.get(a)
            if v:
                if a == 'class':
                    v = ' '.join(v.split()[:4])
                attrs.append(f'{a}="{v[:60]}"')
        text = ' '.join((el.text or '').split())[:90]
        children = [c for c in el if isinstance(c.tag, str)]
        # Skip attribute-less wrappers with no own text: they add depth, not information.
        if attrs or text or not children:
            lines.append(f"{'  ' * min(depth, 12)}<{el.tag}{(' ' + ' '.join(attrs)) if attrs else ''}>{text}")
            depth += 1
        for c in children:
            walk(c, depth)

    body = root.find('body')
    walk(body if body is not None else root, 0)
    outline = '\n'.join(lines)
    if len(outline) > max_chars:
        outline = trim_to_relevant(outline, focus or 'price title main content', max_chars=max_chars)
    return outline


def _selector_kind(selector: str) -> str:
    s = selector.strip()
    if s.startswith(('json:', 'jq:', 'jqraw:')):
        return 'json'
    if s.startswith(('xpath:', 'xpath1:', '/', '(')):
        return 'xpath'
    return 'css'


def check_selector(selector: str, html: str, _soup=None, _tree=None) -> tuple[int, str]:
    """(match count, sample text of the first match) for a CSS or XPath selector.

    Uses the same engines as the processor (BeautifulSoup/soupsieve for CSS, lxml for XPath)
    so "it matched here" means it will match when the watch runs.
    """
    kind = _selector_kind(selector)
    if kind == 'json':
        return 0, ''
    try:
        if kind == 'xpath':
            expr = re.sub(r'^xpath1?:', '', selector.strip())
            tree = _tree if _tree is not None else _parse_html(html)
            found = tree.xpath(expr)
            if not isinstance(found, list):
                found = [found]
            texts = [(f.text_content() if hasattr(f, 'text_content') else str(f)) for f in found]
        else:
            from bs4 import BeautifulSoup
            soup = _soup if _soup is not None else BeautifulSoup(html, 'html.parser')
            found = soup.select(selector)
            texts = [f.get_text(' ', strip=True) for f in found]
    except Exception:
        return 0, ''
    sample = next((' '.join(t.split()) for t in texts if t and t.strip()), '')
    return len(found), sample[:200]


def _validate_selectors(selectors, html, require_text=True) -> list[dict]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'html.parser')
    tree = _parse_html(html)
    out = []
    for sel in selectors:
        if _POSITIONAL_SELECTOR_RE.search(sel) or _HASHED_CLASS_RE.search(sel):
            logger.debug(f"AI assist: rejected fragile selector {sel!r}")
            continue
        count, sample = check_selector(sel, html, _soup=soup, _tree=tree)
        if count and (sample or not require_text):
            out.append({'value': sel, 'matches': count, 'sample': sample})
        else:
            logger.debug(f"AI assist: rejected selector with no usable match {sel!r}")
    return out


# ---------------------------------------------------------------------------
# ignore_text / trigger_text rules
# ---------------------------------------------------------------------------

def compile_line_rule(rule: str):
    """A predicate with the processor's ignore_text semantics, or None if the rule is invalid.

    /regex/flags is a regex search; anything else is a case-insensitive substring.
    """
    from changedetectionio.html_tools import (
        PERL_STYLE_REGEX,
        perl_style_slash_enclosed_regex_to_options,
    )
    rule = (rule or '').strip()
    if not rule:
        return None
    if re.search(PERL_STYLE_REGEX, rule, re.IGNORECASE):
        try:
            rx = re.compile(perl_style_slash_enclosed_regex_to_options(rule))
        except re.error:
            return None
        if rx.search(''):  # matches everything
            return None
        return rx.search
    needle = rule.lower()
    return lambda line: needle in line.lower()


def _validate_rules(rules) -> list[dict]:
    return [{'value': r} for r in rules if compile_line_rule(r)]


# ---------------------------------------------------------------------------
# 1. Filter suggestions
# ---------------------------------------------------------------------------

FILTERS_SYSTEM_PROMPT = (
    "You configure a website change monitor. Given what the user wants to watch and an outline "
    "of the page (one element per line: tag, identifying attributes, then the element's own text), "
    "propose settings that make the monitor see only what matters.\n\n"
    "Respond with ONLY a JSON object:\n"
    '{"include_filters": ["css selector"], "subtractive_selectors": ["css selector"], '
    '"ignore_text": ["text or /regex/"], "trigger_text": ["text or /regex/"], "reason": "one or two sentences"}\n\n'
    "Rules:\n"
    "- include_filters: the smallest element(s) containing what the user wants. Prefer stable hooks: "
    "id, itemprop, data-testid/data-test, semantic tags (main, article), meaningful class names. "
    "NEVER positional selectors (nth-child, nth-of-type, [2]) and NEVER auto-generated class names "
    "(css-1x2y3z, sc-AbCdE, jsx-123). Return an empty list if the whole page is what matters.\n"
    "- subtractive_selectors: noisy parts INSIDE the included area to remove (related items, ads, "
    "reviews, recommendation carousels, cookie banners). Often empty.\n"
    "- ignore_text: lines that change without meaning (relative times, view counters, 'X people "
    "viewing'). Plain text is a case-insensitive 'line contains' match; /regex/ for patterns. Often empty.\n"
    "- trigger_text: only when the user names specific words or phrases to wait for. Usually empty.\n"
    f"- At most {MAX_SUGGESTIONS_PER_LIST} entries per list. Selectors must exist in the outline."
)


def suggest_filters(watch, datastore, goal: str = '') -> dict:
    from .evaluator import resolve_intent

    goal = (goal or '').strip() or resolve_intent(watch, datastore)[0]
    if not goal:
        raise AssistError('Describe what you want to watch on this page first.')

    html, ts = latest_fetched_html(watch)
    if not html:
        raise AssistError('No saved page HTML yet - run a check first (text-only and JSON watches are not supported).')

    user_prompt = (
        f"URL: {watch.get('url', '')}\n"
        f"Page title: {watch.get('page_title') or ''}\n"
        f"What the user wants to watch: {goal}\n"
        f"Current include filters: {watch.get('include_filters') or []}\n\n"
        f"Page outline:\n{html_outline(html, focus=goal)}"
    )
    data = _call_json(datastore, FILTERS_SYSTEM_PROMPT, user_prompt)

    include = _validate_selectors(_str_list(data.get('include_filters')), html)
    subtract = _validate_selectors(_str_list(data.get('subtractive_selectors')), html, require_text=False)
    return {
        'include_filters': include,
        'subtractive_selectors': subtract,
        'ignore_text': _validate_rules(_str_list(data.get('ignore_text'))),
        'trigger_text': _validate_rules(_str_list(data.get('trigger_text'))),
        'reason': str(data.get('reason') or '').strip(),
        'based_on': ts,
    }


# ---------------------------------------------------------------------------
# 2. Noise detection
# ---------------------------------------------------------------------------

_DIGITS_RE = re.compile(r'\d+')


def _shape(line: str) -> str:
    """Line with every run of digits folded, so '3 hours ago' and '5 hours ago' are one shape."""
    return _DIGITS_RE.sub('#', ' '.join(line.split()))


def noise_candidates(texts: list[str]) -> tuple[list[dict], list[set]]:
    """Changed-line shapes across consecutive snapshots, most frequently changing first.

    Returns (candidates, per_diff_changed_lines). A candidate is
    {'shape', 'changes': how many diffs it appeared in, 'examples': [raw lines]}.
    """
    counts = Counter()
    examples = {}
    per_diff = []
    for a, b in zip(texts, texts[1:], strict=False):
        changed = set()
        for ln in build_llm_diff(a, b, context=0).splitlines():
            if ln.startswith(('+', '-')) and not ln.startswith('@@'):
                body = ln[1:].strip()
                if body:
                    changed.add(body)
        per_diff.append(changed)
        for shape in {_shape(x) for x in changed}:
            counts[shape] += 1
        for x in changed:
            ex = examples.setdefault(_shape(x), [])
            if x not in ex and len(ex) < 3:
                ex.append(x)
    candidates = [{'shape': s, 'changes': n, 'examples': examples.get(s, [])}
                  for s, n in counts.most_common() if n >= 2]
    return candidates[:NOISE_MAX_CANDIDATES], per_diff


NOISE_SYSTEM_PROMPT = (
    "You tune a website change monitor that alerts whenever page text changes. You get lines that "
    "changed repeatedly across recent checks (digits folded to #, with real examples and how many "
    "checks each changed in), plus what the user cares about.\n\n"
    "Pick the lines that are NOISE - they change without meaningful news: clocks, relative times "
    "('3 hours ago'), visitor/view counters, 'N people are looking', session or cache-buster tokens, "
    "rotating ads or promos, copyright years, build hashes. Anything the user's intent could care "
    "about is NOT noise (e.g. prices when they watch prices, stock levels, headlines on a news page).\n\n"
    "Respond with ONLY a JSON object:\n"
    '{"ignore_text": [{"pattern": "text or /regex/", "why": "short reason"}], "reason": "one sentence"}\n\n'
    "Patterns: plain text is a case-insensitive 'line contains' match - prefer it when a fixed "
    "phrase identifies the line. Use /regex/ (e.g. /\\d+ people viewing/) when the line has no "
    "fixed part. Keep patterns specific enough not to hide unrelated lines. "
    "Return an empty list when nothing is clearly noise."
)


def suggest_noise(watch, datastore) -> dict:
    from .evaluator import resolve_intent

    keys = list(watch.history.keys())[-NOISE_MAX_SNAPSHOTS:]
    if len(keys) < 3:
        raise AssistError('Needs at least 3 saved snapshots to spot what keeps changing.')
    texts = [watch.get_history_snapshot(timestamp=k) or '' for k in keys]

    candidates, per_diff = noise_candidates(texts)
    if not candidates:
        return {'ignore_text': [], 'reason': 'No line changed more than once across the recent snapshots.',
                'diffs_checked': len(per_diff), 'alerts_avoided': 0}

    existing = [r for r in (watch.get('ignore_text') or []) if r]
    intent = resolve_intent(watch, datastore)[0]
    listing = '\n'.join(
        f"- [{c['changes']}/{len(per_diff)} checks] {c['shape'][:200]}  e.g. " +
        ' | '.join(e[:120] for e in c['examples'][:2])
        for c in candidates
    )
    user_prompt = (
        f"URL: {watch.get('url', '')}\n"
        f"What the user cares about: {intent or '(not stated - assume the main content of the page)'}\n"
        f"Already ignored: {existing or 'nothing'}\n\n"
        f"Lines that changed repeatedly:\n{listing}"
    )
    data = _call_json(datastore, NOISE_SYSTEM_PROMPT, user_prompt)

    raw = data.get('ignore_text')
    items = raw if isinstance(raw, list) else []
    candidate_lines = [e for c in candidates for e in c['examples']]
    candidate_shapes = {c['shape'] for c in candidates}
    # Lines on the current page that are NOT repeat-changers: the price that changed once,
    # headings, the product name. A rule that hides any of them could hide the next real
    # change, so it is rejected however few lines the page has.
    other_lines = [ln.strip() for ln in texts[-1].splitlines()
                   if ln.strip() and _shape(ln.strip()) not in candidate_shapes]
    existing_lower = {e.lower() for e in existing}

    accepted = []
    for item in items[:8]:
        pattern = (item.get('pattern') if isinstance(item, dict) else item) or ''
        pattern = str(pattern).strip()
        why = str(item.get('why') or '').strip() if isinstance(item, dict) else ''
        match = compile_line_rule(pattern)
        if not match or pattern.lower() in existing_lower:
            continue
        hits = [ln for ln in candidate_lines if match(ln)]
        if not hits:
            continue  # doesn't touch any line we showed it - hallucinated
        collateral = [ln for ln in other_lines if match(ln)]
        if collateral:
            logger.debug(f"AI assist: rejected over-broad ignore rule {pattern!r} - would also hide {collateral[:3]}")
            continue
        accepted.append({'value': pattern, 'why': why, 'examples': hits[:3]})

    # How many recent changes would not have been a change at all with these rules.
    matchers = [compile_line_rule(r) for r in existing + [a['value'] for a in accepted]]
    matchers = [m for m in matchers if m]
    avoided = sum(
        1 for changed in per_diff
        if changed and all(any(m(ln) for m in matchers) for ln in changed)
    ) if accepted else 0

    return {
        'ignore_text': accepted,
        'reason': str(data.get('reason') or '').strip(),
        'diffs_checked': len(per_diff),
        'alerts_avoided': avoided,
    }


# ---------------------------------------------------------------------------
# 3. Error diagnosis
# ---------------------------------------------------------------------------

DIAGNOSE_SYSTEM_PROMPT = (
    "You troubleshoot a website change monitor. A watch is failing. You get the error, the watch's "
    "settings, and what the fetched page contained. Work out the most likely cause and how to fix it "
    "in the monitor's own settings.\n\n"
    "Typical causes: bot protection / captcha / 'access denied' page (fix: use a real browser fetcher, "
    "a proxy, or slow the check interval); page moved or removed (404); login required; the CSS/XPath "
    "filter no longer matches because the site changed its layout (fix: a new selector); content "
    "loaded by JavaScript (fix: a browser-based fetcher, a wait); a browser step failing on a "
    "changed button.\n\n"
    "Respond with ONLY a JSON object:\n"
    '{"cause": "blocked|not_found|login_required|layout_changed|javascript_content|browser_steps|network|other", '
    '"diagnosis": "2-3 plain sentences on what is going on", '
    '"fixes": ["concrete step in the monitor settings"], '
    '"include_filters": ["replacement css selector, only when the filter no longer matches"]}\n'
    "Base the diagnosis on the evidence given; say so when it is inconclusive."
)

_FILTER_LOST_RE = re.compile(r'no filters were found|filter', re.IGNORECASE)


def diagnose_error(watch, datastore) -> dict:
    last_error = watch.get('last_error')
    failures = int(watch.get('consecutive_filter_failures') or 0)
    step_error = watch.get('browser_steps_last_error_step')
    if not last_error and not failures and not step_error:
        raise AssistError('This watch has no current error to diagnose.')

    try:
        error_text = watch.get_error_text() or ''
    except Exception:
        error_text = ''
    try:
        page_text = watch.get_last_fetched_text_before_filters() or ''
    except Exception:
        page_text = ''

    error_html, html_note = '', ''
    try:
        error_html = watch.get_error_html() or ''
    except Exception:
        error_html = ''
    if error_html:
        html_note = 'HTML of the failing fetch'
    else:
        error_html, ts = latest_fetched_html(watch)
        if error_html:
            html_note = f'HTML from the last successful check ({ts}); the page may have changed since'

    parts = [
        f"URL: {watch.get('url', '')}",
        f"Error: {last_error or '(none)'}",
        f"Consecutive filter failures: {failures}",
        f"Fetch method: {watch.get('fetch_backend') or 'system default'}",
        f"Include filters: {watch.get('include_filters') or []}",
        f"Remove elements: {watch.get('subtractive_selectors') or []}",
    ]
    if step_error:
        parts.append(f"Browser step that failed: #{step_error} of {len(watch.get('browser_steps') or [])}")
    if error_text:
        parts.append(f"\nText of the error page:\n{error_text[:3000]}")
    if page_text:
        parts.append(f"\nText of the last fetched page (before filters):\n{page_text[:4000]}")
    if error_html:
        parts.append(f"\nPage outline ({html_note}):\n{html_outline(error_html, max_chars=15_000)}")

    data = _call_json(datastore, DIAGNOSE_SYSTEM_PROMPT, '\n'.join(parts))

    include = []
    if error_html and (failures or _FILTER_LOST_RE.search(str(last_error or ''))):
        include = _validate_selectors(_str_list(data.get('include_filters')), error_html)

    return {
        'cause': str(data.get('cause') or 'other').strip(),
        'diagnosis': str(data.get('diagnosis') or '').strip(),
        'fixes': _str_list(data.get('fixes'), limit=5),
        'include_filters': include,
        'html_note': html_note,
    }


# ---------------------------------------------------------------------------
# 4. Tag suggestions
# ---------------------------------------------------------------------------

TAGS_SYSTEM_PROMPT = (
    "You organise watches in a website change monitor into groups (tags). Given a page and the "
    "groups that already exist, pick the groups this page belongs in.\n\n"
    "Respond with ONLY a JSON object:\n"
    '{"existing": ["exact name of an existing group"], "new": ["short new group name"], "reason": "one sentence"}\n\n'
    "Rules:\n"
    "- Strongly prefer existing groups; use their names exactly.\n"
    "- Suggest a new group only when nothing existing fits: 1-3 words, Title Case, a category "
    "(e.g. 'Electronics', 'Job Boards', 'Security Advisories'), never the site's own name.\n"
    "- At most 3 existing and 1 new. Empty lists are fine."
)


def suggest_tags(watch, datastore) -> dict:
    tags = datastore.data['settings']['application'].get('tags', {}) or {}
    titles = {}
    for t in tags.values():
        title = (t.get('title') or '').strip()
        if title:
            titles.setdefault(title.lower(), title)
    current = {(tags.get(u) or {}).get('title', '').lower() for u in watch.get('tags', [])}

    keys = list(watch.history.keys())
    excerpt = (watch.get_history_snapshot(timestamp=keys[-1]) or '')[:3000] if keys else ''
    user_prompt = (
        f"URL: {watch.get('url', '')}\n"
        f"Page title: {watch.get('page_title') or watch.get('title') or ''}\n"
        f"Existing groups: {sorted(titles.values())[:200] or 'none yet'}\n\n"
        f"Page text excerpt:\n{excerpt or '(no snapshot yet)'}"
    )
    data = _call_json(datastore, TAGS_SYSTEM_PROMPT, user_prompt)

    existing = []
    for name in _str_list(data.get('existing')):
        canonical = titles.get(name.lower())
        if canonical and name.lower() not in current and canonical not in existing:
            existing.append(canonical)
    new = []
    for name in _str_list(data.get('new'), limit=1):
        name = re.sub(r'[,\s]+', ' ', name).strip()[:40]
        if name and name.lower() not in titles and name.lower() not in current:
            new.append(name)
    return {'existing': existing, 'new': new, 'reason': str(data.get('reason') or '').strip()}
