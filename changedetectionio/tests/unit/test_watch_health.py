from queue import Queue

from changedetectionio.tests.unit.test_notification_group_configs import FakeDatastore, _watch, _two_notification_groups
from changedetectionio.watch_health import record_check_outcome


def setup_health(**watch_values):
    watch = _watch(url='https://example.com/product', check_failure_notification_send=True,
                   check_recovery_notification_send=True, last_error='HTTP/2 error', **watch_values)
    datastore = FakeDatastore(watch, _two_notification_groups(),
                              {'check_failure_notification_threshold_attempts': 2})
    return datastore, watch, Queue()


def test_threshold_one_alert_per_destination_and_recovery_without_history():
    datastore, watch, queue = setup_health()
    watch.history = {}
    record_check_outcome(datastore, queue, watch['uuid'], False)
    assert queue.empty()
    record_check_outcome(datastore, queue, watch['uuid'], False)
    assert watch['check_failure_alerted']
    assert queue.qsize() == 2
    first = queue.get()
    second = queue.get()
    assert first['notification_urls'] != second['notification_urls']
    assert first['check_error'] == 'HTTP/2 error'
    assert first['failed_check_count'] == 2
    record_check_outcome(datastore, queue, watch['uuid'], False)
    assert queue.empty()
    record_check_outcome(datastore, queue, watch['uuid'], True)
    assert queue.qsize() == 2
    assert 'recovered' in queue.get()['notification_title']
    assert watch['consecutive_check_failures'] == 0
    assert not watch['check_failure_alerted']
    queue.get()
    record_check_outcome(datastore, queue, watch['uuid'], True)
    assert queue.empty()
    for _ in range(2):
        record_check_outcome(datastore, queue, watch['uuid'], False)
    assert queue.qsize() == 2


def test_success_breaks_the_failure_streak():
    datastore, watch, queue = setup_health()
    for outcome in (False, True, False):
        record_check_outcome(datastore, queue, watch['uuid'], outcome)
    assert queue.empty()
    assert watch['consecutive_check_failures'] == 1


def test_muted_watch_retries_alert_after_unmute():
    datastore, watch, queue = setup_health(notification_muted=True)
    for _ in range(3):
        record_check_outcome(datastore, queue, watch['uuid'], False)
    assert queue.empty()
    assert not watch['check_failure_alerted']
    watch['notification_muted'] = False
    record_check_outcome(datastore, queue, watch['uuid'], False)
    assert queue.qsize() == 2


def test_disabled_threshold_and_opt_out_do_not_alert():
    for threshold, enabled in ((0, True), (2, False)):
        datastore, watch, queue = setup_health()
        watch['check_failure_notification_send'] = enabled
        datastore.data['settings']['application']['check_failure_notification_threshold_attempts'] = threshold
        for _ in range(3):
            record_check_outcome(datastore, queue, watch['uuid'], False)
        record_check_outcome(datastore, queue, watch['uuid'], True)
        assert queue.empty()


def test_recovery_can_be_disabled():
    datastore, watch, queue = setup_health()
    watch['check_recovery_notification_send'] = False
    for _ in range(2):
        record_check_outcome(datastore, queue, watch['uuid'], False)
    queue.get()
    queue.get()
    record_check_outcome(datastore, queue, watch['uuid'], True)
    assert queue.empty()
    assert not watch['check_failure_alerted']
