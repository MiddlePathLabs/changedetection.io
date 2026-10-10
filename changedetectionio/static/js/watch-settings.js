
function render_filter_preview_analysis(preview) {
    const $panel = $('#filter-preview-analysis');
    if (!$panel.length || !preview) return;
    $panel.empty().show();

    const coverage = preview.coverage || {};
    const impact = preview.alert_impact || {};
    const status = String(preview.status || 'unknown');
    $panel.append($('<h4>').text('Combined filter analysis'));
    $panel.append($('<p>').text(
        'Status: ' + status + '. Evaluated ' + (coverage.evaluated_snapshots || 0) +
        ' of ' + (coverage.requested_snapshots || 0) + ' selected history versions; ' +
        (impact.would_alert || 0) + ' transition(s) would alert and ' +
        (impact.blocked || 0) + ' would be blocked.'
    ));

    const warnings = preview.warnings || [];
    if (warnings.length) {
        const $warningList = $('<ul>');
        warnings.forEach(function (warning) {
            const details = warning.terms ? ' (' + warning.terms.join(', ') + ')' : '';
            $warningList.append($('<li>').text((warning.message || 'Warning') + details));
        });
        $panel.append($('<strong>').text('Review before saving'), $warningList);
    }

    (preview.snapshots || []).forEach(function (snapshot) {
        const title = snapshot.available
            ? snapshot.version + ' — retained ' + (snapshot.retained_chars || 0) +
              ', removed ' + (snapshot.removed_chars || 0) + ' characters'
            : snapshot.version + ' — ' + (snapshot.reason || 'not evaluated');
        const $details = $('<details>').append($('<summary>').text(title));
        if (snapshot.available) {
            $details.append($('<p>').text(snapshot.blocked ? 'Historical impact: blocked' : 'Historical impact: eligible for alert'));
            $details.append($('<strong>').text('Retained content'));
            $details.append($('<pre>').text(snapshot.retained_content || snapshot.after_filter || ''));
            $details.append($('<strong>').text('Removed content'));
            $details.append($('<pre>').text(snapshot.removed_content || '(none)'));
        } else {
            $details.append($('<p>').text(snapshot.reason || 'Saved raw HTML unavailable.'));
        }
        $panel.append($details);
    });
}

window.render_filter_preview_analysis = render_filter_preview_analysis;

function request_textpreview_update() {
    if (!$('body').hasClass('preview-text-enabled')) {
        console.error("Preview text was requested but body tag was not setup")
        return
    }

    const data = {};
    $('textarea:visible, input:visible, #filters-and-triggers textarea, #filters-and-triggers input').each(function () {
        const $element = $(this); // Cache the jQuery object for the current element
        const name = $element.attr('name'); // Get the name attribute of the element
        if (!name) return;
        // Radios share a name across multiple inputs; .val() returns the value
        // attribute regardless of checked state, so iterating would let the last
        // unchecked radio overwrite the user's actual selection. Skip unchecked.
        if ($element.is(':radio') && !$element.is(':checked')) return;
        data[name] = $element.is(':checkbox') ? ($element.is(':checked') ? $element.val() : false) : $element.val();
    });

    // llm_intent lives in a separate (potentially hidden) tab — include it explicitly
    const $llmIntent = $('textarea[name="llm_intent"]');
    if ($llmIntent.length) {
        data['llm_intent'] = $llmIntent.val();
    }

    $('body').toggleClass('spinner-active', 1);
    $('#llm-preview-result').hide();

    $.abortiveSingularAjax({
        type: "POST",
        url: preview_text_edit_filters_url,
        data: data,
        namespace: 'watchEdit'
    }).done(function (data) {
        console.debug(data['duration'])
        $('#filters-and-triggers #text-preview-before-inner').text(data['before_filter']);
        $('#filters-and-triggers #text-preview-inner')
            .text(data['after_filter'])
            .highlightLines([
                {
                    'color': 'var(--highlight-trigger-text-bg-color)',
                    'lines': data['trigger_line_numbers'],
                    'title': i18nT('highlightTrigger', "Triggers a change if this text appears, AND something changed in the document.")
                },
                {
                    'color': 'var(--highlight-ignored-text-bg-color)',
                    'lines': data['ignore_line_numbers'],
                    'title': i18nT('highlightIgnored', "Ignored for calculating changes, but still shown.")
                },
                {
                    'color': 'var(--highlight-blocked-text-bg-color)',
                    'lines': data['blocked_line_numbers'],
                    'title': i18nT('highlightBlocked', "No change-detection will occur because this text exists.")
                }
            ])

        // LLM preview extraction result
        const $llmResult = $('#llm-preview-result');
        if ($llmResult.length && data['llm_evaluation']) {
            const ev = data['llm_evaluation'];
            const found = ev['found'];
            $llmResult.attr('data-found', found ? '1' : '0');
            $llmResult.find('.llm-preview-verdict').text(
                found ? '✓ Would trigger a change' : '✗ Would not trigger a change'
            );
            $llmResult.find('.llm-preview-answer').text(ev['answer'] || '');
            $llmResult.show();
        } else if ($llmResult.length) {
            $llmResult.hide();
        }
        render_filter_preview_analysis(data['combined_filter_preview']);
    }).fail(function (error) {
        if (error.statusText === 'abort') {
            console.log('Request was aborted due to a new request being fired.');
        } else {
            $('#filters-and-triggers #text-preview-inner').text('There was an error communicating with the server.');
        }
    })
}

window.preview_filter_draft = function () {
    if (!$('#activate-text-preview').length) return;
    if (!$('body').hasClass('preview-text-enabled')) {
        $('#activate-text-preview').trigger('click');
    } else {
        request_textpreview_update();
    }
};


$(document).ready(function () {

    $('#notification-setting-reset-to-default').click(function (e) {
        $('#notification_title').val('');
        $('#notification_body').val('');
        $('#notification_format').val('System default');
        $('#notification_urls').val('');
        $('#notification_muted_none').prop('checked', true); // in the case of a ternary field
        e.preventDefault();
    });
    $("#notification-token-toggle").click(function (e) {
        e.preventDefault();
        $('#notification-tokens-info').toggle();
    });

    toggleOpacity('#time_between_check_use_default', '#time-check-widget-wrapper, #time-between-check-schedule', false);


    const vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0);
    $("#text-preview-inner").css('max-height', (vh - 300) + "px");
    $("#text-preview-before-inner").css('max-height', (vh - 300) + "px");

    $("#activate-text-preview").click(function (e) {
        $('body').toggleClass('preview-text-enabled')
        request_textpreview_update();
        const method = $('body').hasClass('preview-text-enabled') ? 'on' : 'off';
        $('#filters-and-triggers textarea')[method]('blur', request_textpreview_update.throttle(1000));
        $('#filters-and-triggers input')[method]('change', request_textpreview_update.throttle(1000));
        $("#filters-and-triggers-tab")[method]('click', request_textpreview_update.throttle(1000));
    });

    $('#filter-repair-restore').click(function () {
        const previous = window.filter_repair_previous || {};
        Object.keys(previous).forEach(function (field) {
            const $field = $('#' + field);
            if (!$field.length) return;
            const value = previous[field];
            if (Array.isArray(value)) {
                $field.val(value.join('\n'));
            } else if ($field.is(':checkbox')) {
                $field.prop('checked', Boolean(value));
            } else {
                $field.val(value == null ? '' : value);
            }
        });
        $('#filter-repair-restore-status').text('Previous configuration restored to the draft. Preview it, then press Save to keep it.');
        window.preview_filter_draft();
    });
    var _mt = {};
    _mt[i18nT('tabContentAfter', 'Content after filters')] = "#text-preview-inner";
    _mt[i18nT('tabContentBefore', 'Content raw/before filters')] = "#text-preview-before-inner";
    $('.minitabs-wrapper').miniTabs(_mt);
});

