"""Persistent, thresholded watch failure and recovery notifications."""

from changedetectionio.notification_service import create_notification_service


def record_check_outcome(datastore, notification_q, watch_uuid, succeeded):
    watch = datastore.data['watching'].get(watch_uuid)
    if not watch:
        return
    failures = watch.get('consecutive_check_failures', 0)
    alerted = bool(watch.get('check_failure_alerted'))
    threshold = datastore.data['settings']['application'].get('check_failure_notification_threshold_attempts', 6)
    enabled = watch.get('check_failure_notification_send', False) and threshold > 0
    service = create_notification_service(datastore, notification_q)
    if succeeded:
        if enabled and alerted and watch.get('check_recovery_notification_send', True):
            service.send_check_health_notification(watch_uuid, recovered=True)
        updates = {'consecutive_check_failures': 0, 'check_failure_alerted': False}
    else:
        failures += 1
        datastore.update_watch(uuid=watch_uuid, update_obj={'consecutive_check_failures': failures})
        if enabled and not alerted:
            if failures >= threshold:
                alerted = service.send_check_health_notification(watch_uuid)
        updates = {'check_failure_alerted': alerted if enabled else False}
    if any(watch.get(key) != value for key, value in updates.items()):
        datastore.update_watch(uuid=watch_uuid, update_obj=updates)
