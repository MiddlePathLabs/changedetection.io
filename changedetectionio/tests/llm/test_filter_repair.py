from changedetectionio.processors.text_json_diff import build_combined_filter_preview


class FakeDatastore:
    def __init__(self):
        self.data = {'settings': {'application': {}}}

    def get_tag_overrides_for_watch(self, uuid, attr):
        return []


class FakeWatch(dict):
    is_source_type_url = False
    is_pdf = False

    def __init__(self, snapshots):
        super().__init__(uuid='f1-test', url='https://example.test/item', tags=[], **snapshots['config'])
        self._history = dict.fromkeys(snapshots['html'], True)
        self._html = snapshots['html']

    @property
    def history(self):
        return self._history

    def get_fetched_html(self, timestamp):
        return self._html.get(timestamp, False)


def test_combined_preview_reports_retained_removed_and_alert_impact():
    watch = FakeWatch({
        'config': {'content-type': 'text/html'},
        'html': {
            1: '<html><body><main id="main"><p>Price $20</p><div class="ad">Ad</div></main></body></html>',
            2: '<html><body><main id="main"><p>Price $15</p><div class="ad">Ad</div></main></body></html>',
        },
    })

    preview = build_combined_filter_preview(
        FakeDatastore(),
        watch,
        proposed_config={'include_filters': ['#main'], 'subtractive_selectors': ['.ad']},
        goal='price',
    )

    assert preview['status'] == 'ok'
    assert preview['coverage']['evaluated_snapshots'] == 2
    assert 'Price $20' in preview['snapshots'][0]['retained_content']
    assert 'Ad' in preview['snapshots'][0]['removed_content']
    assert preview['alert_impact']['would_alert'] == 1
    assert not preview['warnings']


def test_combined_preview_detects_destructive_combination_and_goal_loss():
    watch = FakeWatch({
        'config': {'content-type': 'text/html'},
        'html': {
            1: '<html><body><main id="main"><p class="price">Price $20</p></main></body></html>',
        },
    })

    preview = build_combined_filter_preview(
        FakeDatastore(),
        watch,
        proposed_config={'include_filters': ['#main'], 'subtractive_selectors': ['.price']},
        goal='price',
    )

    kinds = {warning['kind'] for warning in preview['warnings']}
    assert 'empty_output' in kinds
    assert 'destructive_combination' in kinds
    assert 'goal_loss' in kinds
