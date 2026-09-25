// AI setup assistant on the watch edit page (templates/edit/include_ai_assist.html).
// Each button POSTs once to /edit/<uuid>/ai-assist/<kind>; the reply is a list of already
// validated suggestions. Chosen ones are appended to the matching form field on its tab -
// nothing is saved until the user presses Save.
$(function () {
    const $box = $('#ai-assist');
    if (!$box.length) return;
    const t = window.ai_assist_i18n || {};
    const $result = $('#ai-assist-result');

    function esc(s) {
        return $('<div>').text(s == null ? '' : String(s)).html();
    }

    // Append lines to a textarea list field (one entry per line), skipping ones already there.
    function appendLines(fieldId, values) {
        const $f = $('#' + fieldId);
        if (!$f.length) return false;
        const current = ($f.val() || '').split('\n').map(s => s.trim()).filter(Boolean);
        values.forEach(v => { if (!current.includes(v)) current.push(v); });
        $f.val(current.join('\n')).trigger('change');
        return true;
    }

    // Tags are a single comma-separated input of group titles.
    function appendTags(values) {
        const $f = $('#tags');
        if (!$f.length) return false;
        const current = ($f.val() || '').split(',').map(s => s.trim()).filter(Boolean);
        const lower = current.map(s => s.toLowerCase());
        values.forEach(v => { if (!lower.includes(v.toLowerCase())) current.push(v); });
        $f.val(current.join(', ')).trigger('change');
        return true;
    }

    // One checkbox list per target field, with an "Add selected" button.
    function renderGroup(field, items, label) {
        if (!items || !items.length) return null;
        const $g = $('<div class="ai-assist-group">').append($('<strong>').text(label || t[field] || field));
        const $ul = $('<ul style="list-style:none;margin:0.3em 0;padding:0">');
        items.forEach(function (it) {
            const bits = [];
            if (it.matches) bits.push(it.matches + ' ' + (it.matches === 1 ? (t.match || 'match') : (t.matches || 'matches')));
            if (it.sample) bits.push('"' + it.sample + '"');
            if (it.why) bits.push(it.why);
            if (it.examples && it.examples.length) bits.push('e.g. ' + it.examples.slice(0, 2).join(' | '));
            if (it.is_new) bits.push(t.new_group || 'new group');
            const $li = $('<li>').append(
                $('<label>').append(
                    $('<input type="checkbox" checked>').attr('data-value', it.value),
                    ' <code>' + esc(it.value) + '</code>',
                    bits.length ? ' <small style="opacity:0.75">' + esc(bits.join(' — ')) + '</small>' : ''
                )
            );
            $ul.append($li);
        });
        const $btn = $('<button type="button" class="pure-button button-xsmall">').text(t.add || 'Add selected');
        const $done = $('<span class="pure-form-message-inline">');
        $btn.on('click', function () {
            const values = $ul.find('input:checked').map(function () { return $(this).attr('data-value'); }).get();
            if (!values.length) return;
            const ok = field === 'tags' ? appendTags(values) : appendLines(field, values);
            $done.text(ok ? (t.added || 'Added') : field + '?');
        });
        return $g.append($ul, $btn, ' ', $done);
    }

    function note(text) {
        return $('<p>').append($('<small>').text(text));
    }

    function render(kind, data) {
        $result.empty();
        const groups = [];
        const notes = [];
        if (kind === 'diagnose') {
            if (data.diagnosis) $result.append($('<p>').text(data.diagnosis));
            if (data.fixes && data.fixes.length) {
                const $ul = $('<ul>');
                data.fixes.forEach(f => $ul.append($('<li>').text(f)));
                $result.append($('<strong>').text(t.fixes || 'Suggested fixes'), $ul);
            }
            groups.push(renderGroup('include_filters', data.include_filters));
            if (data.html_note) notes.push(note((t.based_on || 'Based on') + ': ' + data.html_note));
        } else if (kind === 'filters') {
            ['include_filters', 'subtractive_selectors', 'ignore_text', 'trigger_text'].forEach(f => groups.push(renderGroup(f, data[f])));
        } else if (kind === 'noise') {
            groups.push(renderGroup('ignore_text', data.ignore_text));
            if (data.ignore_text && data.ignore_text.length && data.diffs_checked) {
                notes.push($('<p>').text((t.avoided || '').replace('%(n)s', data.alerts_avoided).replace('%(m)s', data.diffs_checked)));
            }
        } else if (kind === 'tags') {
            const items = (data.existing || []).map(v => ({value: v}))
                .concat((data.new || []).map(v => ({value: v, is_new: true})));
            groups.push(renderGroup('tags', items));
        }
        const shown = groups.filter(Boolean);
        if (!shown.length && kind !== 'diagnose') $result.append($('<p>').text(t.nothing || 'No suggestions'));
        shown.forEach(g => $result.append(g));
        notes.forEach(n => $result.append(n));
        if (data.reason) $result.append(note(data.reason));
    }

    $box.on('click', '[data-ai-assist]', function () {
        const kind = $(this).attr('data-ai-assist');
        const $buttons = $box.find('[data-ai-assist]');
        const payload = {};
        if (kind === 'filters') {
            payload.goal = $('#ai-assist-goal').val() || $('#llm_intent').val() || '';
        }
        $buttons.prop('disabled', true);
        $result.empty().append($('<p>').text(t.working || '…'));
        $.ajax({
            url: $box.data('url').replace('__KIND__', kind),
            type: 'POST',
            data: payload,
            dataType: 'json'
        }).done(function (data) {
            render(kind, data);
        }).fail(function (xhr) {
            const msg = (xhr.responseJSON && xhr.responseJSON.error) || (t.failed || 'Request failed') + ' (' + xhr.status + ')';
            $result.empty().append($('<p class="error">').text(msg));
        }).always(function () {
            $buttons.prop('disabled', false);
        });
    });
});
