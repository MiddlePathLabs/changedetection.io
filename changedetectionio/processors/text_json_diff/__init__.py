from loguru import logger
import re

# Processor capabilities
supports_visual_selector = True
supports_browser_steps = True
supports_text_filters_and_triggers = True
supports_text_filters_and_triggers_elements = True
supports_request_type = True



def _task(watch, update_handler):
    from changedetectionio import html_tools
    from changedetectionio.content_fetchers.exceptions import ReplyWithContentButNoText
    from changedetectionio.processors.text_json_diff.processor import FilterNotFoundInResponse

    text_after_filter = ''
    watch['_preview_mode'] = True
    watch['_preview_regex_timeout'] = html_tools.BOUNDED_REGEX_TIMEOUT_SECONDS

    try:
        # The slow process (we run 2 of these in parallel)
        # Always force reprocess for preview - we want to show the filtered content regardless of checksums
        changed_detected, update_obj, text_after_filter = update_handler.run_changedetection(watch=watch, force_reprocess=True)
    except FilterNotFoundInResponse as e:
        text_after_filter = f"Filter not found in HTML: {str(e)}"
    except ReplyWithContentButNoText as e:
        text_after_filter = "Filter found but no text (empty result)"
    except Exception as e:
        text_after_filter = f"Error: {str(e)}"

    if not text_after_filter.strip():
        text_after_filter = 'Empty content'

    # because run_changedetection always returns bytes due to saving the snapshots etc
    text_after_filter = text_after_filter.decode('utf-8') if isinstance(text_after_filter, bytes) else text_after_filter

    return text_after_filter


def _compute_ignore_line_numbers_for_preview(text_pre_extract, ignore_patterns, extract_patterns):
    """1-indexed output line numbers in the post-extract display that correspond
    to input lines matching ignore_text patterns.

    Needed because extract_text (#4138) transforms line content — e.g. "0.54.10"
    becomes ".54.10" — so a substring match for "0.54.10" against the post-extract
    text fails and the preview UI can no longer mark the line as ignored. We find
    the ignored line numbers in the pre-extract text and replay extract_by_regex
    line-by-line to map them forward.
    """
    from changedetectionio import html_tools
    from changedetectionio.processors.text_json_diff.processor import ContentTransformer

    if not text_pre_extract or not ignore_patterns:
        return []

    ignored_input_lines = set(
        html_tools.strip_ignore_text(
            content=text_pre_extract,
            wordlist=ignore_patterns,
            mode='line numbers',
            regex_timeout=html_tools.BOUNDED_REGEX_TIMEOUT_SECONDS,
        )
    )
    if not ignored_input_lines:
        return []

    if not extract_patterns:
        return sorted(ignored_input_lines)

    # Replay extract_by_regex per-line. Each emitted match ends with exactly one
    # '\n', so counting newlines tells us how many output lines this input produced.
    output_line_counter = 0
    result = []
    for input_idx, line in enumerate(text_pre_extract.splitlines()):
        is_ignored = (input_idx + 1) in ignored_input_lines
        matches_in_line = ContentTransformer.extract_by_regex(
            line, extract_patterns, regex_timeout=html_tools.BOUNDED_REGEX_TIMEOUT_SECONDS
        ).count('\n')
        for _ in range(matches_in_line):
            output_line_counter += 1
            if is_ignored:
                result.append(output_line_counter)

    return result


FILTER_PREVIEW_MAX_SNAPSHOTS = 5
FILTER_PREVIEW_MAX_RAW_CHARS = 300_000
FILTER_PREVIEW_OUTPUT_CHARS = 12_000


def _preview_list(value):
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    if isinstance(value, (list, tuple)):
        return [str(line).strip() for line in value if str(line).strip()]
    return []


def _clip_preview_text(value):
    value = str(value or '')
    if len(value) <= FILTER_PREVIEW_OUTPUT_CHARS:
        return value, False
    return value[:FILTER_PREVIEW_OUTPUT_CHARS] + '\n… [preview truncated]', True


def _preview_goal_terms(goal):
    stop_words = {
        'about', 'after', 'alert', 'alerts', 'and', 'any', 'for', 'from', 'only',
        'that', 'the', 'this', 'when', 'with', 'would', 'you',
    }
    return sorted({term for term in re.findall(r"[a-z0-9][a-z0-9$%._-]{2,}", (goal or '').lower())
                   if term not in stop_words})[:30]


def build_combined_filter_preview(datastore, watch, proposed_config=None, goal=''):
    """Evaluate one draft filter configuration against bounded saved HTML history."""
    from copy import deepcopy
    from changedetectionio import html_tools
    from changedetectionio.processors.text_json_diff.processor import (
        render_filter_preview_snapshot,
    )

    draft = deepcopy(watch)
    proposed_config = proposed_config or {}
    for field in (
            'include_filters', 'subtractive_selectors', 'ignore_text', 'trigger_text',
            'extract_lines_containing', 'extract_text', 'text_should_not_be_present'):
        if field in proposed_config:
            draft[field] = _preview_list(proposed_config.get(field))
    for field in ('strip_ignored_lines', 'remove_duplicate_lines', 'sort_text_alphabetically', 'trim_text_whitespace'):
        if field in proposed_config:
            draft[field] = proposed_config.get(field)

    history_keys = list(getattr(watch, 'history', {}).keys())[-FILTER_PREVIEW_MAX_SNAPSHOTS:]
    snapshots = []
    evaluation_records = []
    warnings = []
    unavailable_versions = []
    errors = []
    for version in history_keys:
        version_text = str(version)
        try:
            raw_html = watch.get_fetched_html(timestamp=version)
        except Exception:
            raw_html = False
        if not raw_html:
            unavailable_versions.append(version_text)
            snapshots.append({
                'version': version_text,
                'available': False,
                'reason': 'Saved raw HTML is unavailable for this version.',
            })
            continue
        if len(raw_html) > FILTER_PREVIEW_MAX_RAW_CHARS:
            reason = f'Saved raw HTML exceeds the {FILTER_PREVIEW_MAX_RAW_CHARS:,}-character preview limit.'
            errors.append({'version': version_text, 'reason': reason})
            snapshots.append({'version': version_text, 'available': False, 'reason': reason})
            continue
        try:
            result = render_filter_preview_snapshot(
                draft,
                datastore,
                raw_html,
                regex_timeout=html_tools.BOUNDED_REGEX_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            reason = 'Regex evaluation exceeded the bounded preview time limit.'
            errors.append({'version': version_text, 'reason': reason})
            snapshots.append({'version': version_text, 'available': False, 'reason': reason})
            continue
        except Exception as exc:
            reason = f'Preview could not process this saved version: {exc}'
            errors.append({'version': version_text, 'reason': reason})
            snapshots.append({'version': version_text, 'available': False, 'reason': reason})
            continue

        before = result['before_filter']
        after = result['after_filter']
        goal_terms_removed = [term for term in _preview_goal_terms(goal)
                              if term in before.lower() and term not in after.lower()]
        if not after.strip():
            warnings.append({
                'kind': 'empty_output',
                'version': version_text,
                'message': 'The combined configuration leaves no content to monitor.',
            })
        if before.strip() and len(after) / max(len(before), 1) < 0.1:
            warnings.append({
                'kind': 'excessive_removal',
                'version': version_text,
                'message': 'The combined configuration removes more than 90% of the available text.',
            })
        if goal_terms_removed:
            warnings.append({
                'kind': 'goal_loss',
                'version': version_text,
                'terms': goal_terms_removed,
                'message': 'Goal-related terms present before filtering disappear from the retained output.',
            })

        clipped = {}
        clipped_truncated_any = False
        for field in ('before_filter', 'after_filter', 'retained_content', 'removed_content'):
            clipped[field], clipped_truncated = _clip_preview_text(result[field])
            if clipped_truncated:
                clipped_truncated_any = True
        if clipped_truncated_any:
            clipped['truncated'] = True
        snapshots.append({
            'version': version_text,
            'available': True,
            **clipped,
            'raw_chars': result['raw_chars'],
            'retained_chars': result['retained_chars'],
            'removed_chars': max(result['raw_chars'] - result['retained_chars'], 0),
            'goal_terms_removed': goal_terms_removed,
            'blocked': result['blocked'],
            'trigger_found': result['trigger_found'],
        })
        evaluation_records.append({
            'version': version_text,
            'after_filter': after,
            'blocked': result['blocked'],
        })

        # A selector can match by itself and still be destroyed by the combined
        # subtractive/include configuration. This is the failure mode individual
        # selector validation cannot see.
        if not after.strip() and raw_html:
            try:
                from changedetectionio.llm.assist import check_selector
                individually_valid = []
                for selector in draft.get('include_filters', []):
                    if check_selector(selector, raw_html)[0]:
                        individually_valid.append(selector)
                for selector in draft.get('subtractive_selectors', []):
                    if check_selector(selector, raw_html)[0]:
                        individually_valid.append(selector)
                if individually_valid:
                    warnings.append({
                        'kind': 'destructive_combination',
                        'version': version_text,
                        'selectors': individually_valid[:8],
                        'message': 'Selectors match this page individually, but their combined result is empty.',
                    })
            except Exception as exc:
                logger.debug(f'Could not inspect combined selector interaction: {exc}')

    evaluated = [snapshot for snapshot in snapshots if snapshot.get('available')]
    transitions = []
    previous = None
    for snapshot in evaluation_records:
        if previous is not None:
            changed = snapshot['after_filter'] != previous['after_filter']
            would_alert = changed and not snapshot['blocked']
            transitions.append({
                'from_version': previous['version'],
                'to_version': snapshot['version'],
                'changed': changed,
                'blocked': bool(snapshot['blocked']),
                'would_alert': would_alert,
            })
        previous = snapshot

    if unavailable_versions:
        warnings.append({
            'kind': 'coverage',
            'message': 'Some selected history versions have no saved raw HTML and were not evaluated.',
            'versions': unavailable_versions,
        })
    for error in errors:
        warnings.append({'kind': 'coverage', **error})
    if not history_keys:
        warnings.append({
            'kind': 'coverage',
            'message': 'No saved history versions are available for filter preview.',
        })

    status = 'ok'
    if errors or unavailable_versions or not history_keys:
        status = 'partial'
    if not history_keys or (history_keys and not evaluated):
        status = 'fail_open'

    return {
        'status': status,
        'coverage': {
            'requested_snapshots': len(history_keys),
            'evaluated_snapshots': len(evaluated),
            'unavailable_versions': unavailable_versions,
            'errors': errors,
        },
        'alert_impact': {
            'transitions': transitions,
            'would_alert': sum(1 for item in transitions if item['would_alert']),
            'blocked': sum(1 for item in transitions if item['blocked']),
            'unchanged': sum(1 for item in transitions if not item['changed']),
        },
        'warnings': warnings,
        'snapshots': snapshots,
    }


def prepare_filter_prevew(datastore, watch_uuid, form_data):
    '''Used by @app.route("/edit/<uuid_str:uuid>/preview-rendered", methods=['POST'])'''
    from changedetectionio import forms, html_tools
    from changedetectionio.model.Watch import model as watch_model
    from concurrent.futures import ThreadPoolExecutor
    from copy import deepcopy
    from flask import request
    import brotli
    import importlib
    import os
    import time
    now = time.time()

    text_after_filter = ''
    text_before_filter = ''
    text_pre_extract = ''
    trigger_line_numbers = []
    ignore_line_numbers = []
    blocked_line_numbers = []

    tmp_watch = deepcopy(datastore.data['watching'].get(watch_uuid))

    if tmp_watch and tmp_watch.history and os.path.isdir(tmp_watch.data_dir):
        # Splice in the temporary stuff from the form
        form = forms.processor_text_json_diff_form(formdata=form_data if request.method == 'POST' else None,
                                                   data=form_data
                                                   )

        # Only update vars that came in via the AJAX post
        p = {k: v for k, v in form.data.items() if k in form_data.keys()}
        tmp_watch.update(p)

        # Apply llm_intent from form directly — it's not part of processor_text_json_diff_form
        # but the AJAX sends all visible inputs, so it arrives in form_data
        if hasattr(form_data, 'get') and 'llm_intent' in form_data:
            tmp_watch['llm_intent'] = (form_data.get('llm_intent') or '').strip()

        blank_watch_no_filters = watch_model(datastore_path=datastore.datastore_path, __datastore=datastore.data)
        blank_watch_no_filters['url'] = tmp_watch.get('url')

        latest_filename = next(reversed(tmp_watch.history))
        html_fname = os.path.join(tmp_watch.data_dir, f"{latest_filename}.html.br")
        with open(html_fname, 'rb') as f:
            decompressed_data = brotli.decompress(f.read()).decode('utf-8') if html_fname.endswith('.br') else f.read().decode('utf-8')

            # Just like a normal change detection except provide a fake "watch" object and dont call .call_browser()
            processor_module = importlib.import_module("changedetectionio.processors.text_json_diff.processor")
            update_handler = processor_module.perform_site_check(datastore=datastore,
                                                                 watch_uuid=tmp_watch.get('uuid')  # probably not needed anymore anyway?
                                                                 )
            # Use the last loaded HTML as the input
            update_handler.datastore = datastore
            update_handler.fetcher.content = str(decompressed_data) # str() because playwright/puppeteer/requests return string
            update_handler.fetcher.headers['content-type'] = tmp_watch.get('content-type')

            # Process our watch with filters and the HTML from disk, and also a blank watch with no filters but also with the same HTML from disk.
            # The third task runs with extract_text cleared so we can compute ignore_line_numbers
            # against the pre-extract text (extract_text transforms lines so post-extract substring
            # matching for ignore patterns would otherwise fail — see #4138 follow-up).
            # Do this as parallel threads (not processes) to avoid pickle issues with Lock objects
            tmp_watch_no_extract = deepcopy(tmp_watch)
            tmp_watch_no_extract['extract_text'] = []
            try:
                with ThreadPoolExecutor(max_workers=3) as executor:
                    future1 = executor.submit(_task, tmp_watch, update_handler)
                    future2 = executor.submit(_task, blank_watch_no_filters, update_handler)
                    future3 = executor.submit(_task, tmp_watch_no_extract, update_handler)

                    text_after_filter = future1.result()
                    text_before_filter = future2.result()
                    text_pre_extract = future3.result()
            except Exception as e:
                x=1

    try:
        trigger_line_numbers = html_tools.strip_ignore_text(content=text_after_filter,
                                                            wordlist=tmp_watch['trigger_text'],
                                                            mode='line numbers',
                                                            regex_timeout=html_tools.BOUNDED_REGEX_TIMEOUT_SECONDS,
                                                            )
    except Exception as e:
        text_before_filter = f"Error: {str(e)}"

    try:
        text_to_ignore = tmp_watch.get('ignore_text', []) + datastore.data['settings']['application'].get('global_ignore_text', [])
        ignore_line_numbers = _compute_ignore_line_numbers_for_preview(
            text_pre_extract=text_pre_extract,
            ignore_patterns=text_to_ignore,
            extract_patterns=tmp_watch.get('extract_text', [])
        )
    except Exception as e:
        text_before_filter = f"Error: {str(e)}"

    try:
        blocked_line_numbers = html_tools.strip_ignore_text(content=text_after_filter,
                                                           wordlist=tmp_watch.get('text_should_not_be_present', []) + datastore.data['settings']['application'].get('text_should_not_be_present', []),
                                                           mode='line numbers',
                                                           regex_timeout=html_tools.BOUNDED_REGEX_TIMEOUT_SECONDS,
                                                           )
    except Exception as e:
        text_before_filter = f"Error: {str(e)}"

    # LLM preview extraction — asks the LLM to directly answer the intent
    # against the current filtered content (no diff comparison).
    # e.g. intent "how many articles?" → answer "30 articles listed"
    # Results are NOT cached back to the real watch.
    llm_evaluation = None
    try:
        from changedetectionio.llm.evaluator import preview_extract
        if text_after_filter and text_after_filter.strip() not in ('', 'Empty content'):
            llm_evaluation = preview_extract(tmp_watch, datastore, content=text_after_filter)
    except Exception as e:
        logger.warning(f"LLM preview evaluation failed for {watch_uuid}: {e}")

    combined_filter_preview = build_combined_filter_preview(
        datastore,
        tmp_watch,
        goal=tmp_watch.get('llm_intent', '') if tmp_watch else '',
    ) if tmp_watch else {
        'status': 'fail_open',
        'coverage': {'requested_snapshots': 0, 'evaluated_snapshots': 0,
                     'unavailable_versions': [], 'errors': []},
        'alert_impact': {'transitions': [], 'would_alert': 0, 'blocked': 0, 'unchanged': 0},
        'warnings': [{'kind': 'coverage', 'message': 'Watch was not found.'}],
        'snapshots': [],
    }

    logger.trace(f"Parsed in {time.time() - now:.3f}s")

    return ({
        'after_filter': text_after_filter,
        'before_filter': text_before_filter.decode('utf-8') if isinstance(text_before_filter, bytes) else text_before_filter,
        'blocked_line_numbers': blocked_line_numbers,
        'duration': time.time() - now,
        'ignore_line_numbers': ignore_line_numbers,
        'llm_evaluation': llm_evaluation,
        'combined_filter_preview': combined_filter_preview,
        'trigger_line_numbers': trigger_line_numbers,
        })


