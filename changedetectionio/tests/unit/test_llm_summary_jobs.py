#!/usr/bin/env python3

"""The AI-summary job registry must never pay for the same summary twice.

The /llm-summary route is now start-then-wait: a POST kicks off generation on this pool, and the
browser finds out it finished either from the llm_summary_ready Socket.IO event or by polling the
route. That turns duplicate-suppression into a correctness requirement rather than a nicety - a
caller that decides "nothing is running" starts a *second* LLM call and bills the operator twice
for one summary. Two tabs on the same watch hit the same path.

What this asserts, in the order it matters:
  - concurrent submits for one key produce exactly one run
  - the pending flag is visible from the moment submit() returns (no gap for a poll to fall into)
  - the flag clears only after the job function has returned, i.e. after it has written its result
  - failures are held for exactly one reader instead of dying with the thread
  - a different key is never suppressed
  - the "job settled" announcement fires after the result is readable, and cannot break the job
  - a running job reports when it started, so the routes can hand the browser a real deadline

run from dir above changedetectionio/ dir
python3 -m unittest changedetectionio.tests.unit.test_llm_summary_jobs
"""

import threading
import unittest

from changedetectionio.llm.summary_jobs import (
    SummaryJobFailed,
    SummaryJobRegistry,
    SummaryQueueFullError,
)


class TestSummaryJobRegistry(unittest.TestCase):

    def setUp(self):
        self.reg = SummaryJobRegistry(max_workers=4)

    def _submit(self, reg, key, fn, on_settled=None):
        done = threading.Event()

        def _settled(settled_key):
            try:
                if on_settled:
                    on_settled(settled_key)
            finally:
                done.set()

        accepted = reg.submit(key, fn, on_settled=_settled)
        return accepted, done

    def _assert_settled(self, done):
        self.assertTrue(done.wait(5), "job did not settle")

    def test_duplicate_submit_runs_once(self):
        """The whole point: N callers, one LLM call."""
        runs = []
        release = threading.Event()
        entered = threading.Event()

        def job():
            runs.append(1)
            entered.set()
            release.wait(5)

        accepted, done = self._submit(self.reg, 'k', job)
        duplicate_results = [self.reg.submit('k', job) for _ in range(4)]

        self.assertTrue(entered.wait(5), "job never started")
        self.assertEqual([accepted, *duplicate_results], [True, False, False, False, False],
                         "only the first submit should start work")
        release.set()
        self._assert_settled(done)
        self.assertEqual(len(runs), 1, f"job ran {len(runs)} times, expected exactly 1")

    def test_pending_is_visible_immediately_after_submit(self):
        """A poll arriving between submit() and the worker starting must still see 'pending'.

        If it saw 'idle' it would start a duplicate generation.
        """
        release = threading.Event()
        _, done = self._submit(self.reg, 'k', lambda: release.wait(5))
        self.assertTrue(self.reg.is_pending('k'))
        release.set()
        self._assert_settled(done)
        self.assertFalse(self.reg.is_pending('k'))

    def test_pending_clears_only_after_job_body_completes(self):
        """Ordering guard: the result is written inside the job, so the flag must outlive it."""
        wrote = []
        gate = threading.Event()

        def job():
            gate.wait(5)
            wrote.append('saved')      # stands in for save_llm_diff_summary()

        _, done = self._submit(self.reg, 'k', job)
        self.assertTrue(self.reg.is_pending('k'))
        self.assertEqual(wrote, [], "job body should not have finished yet")
        gate.set()
        self._assert_settled(done)
        self.assertEqual(wrote, ['saved'],
                         "pending cleared before the job wrote its result - a poll could now "
                         "see neither a cached summary nor a running job and re-bill the LLM")

    def test_run_started_at_is_stamped_only_while_the_job_actually_runs(self):
        """The deadline the browser counts down to is anchored to this.

        A job queued behind another generation has not started burning its LLM timeout yet, so it
        must report None (the route then measures from now and the deadline keeps sliding) rather
        than a start time that would expire while it is still waiting for a worker.
        """
        reg = SummaryJobRegistry(max_workers=1)
        release = threading.Event()
        running = threading.Event()

        def first():
            running.set()
            release.wait(5)

        _, busy_done = self._submit(reg, 'busy', first)
        _, queued_done = self._submit(reg, 'queued', lambda: None)
        self.assertTrue(running.wait(5), "first job never started")

        self.assertIsNotNone(reg.run_started_at('busy'), "a running job must report its start")
        self.assertIsNone(reg.run_started_at('queued'),
                          "a job still waiting for a worker has not started its timeout")

        release.set()
        self._assert_settled(busy_done)
        self._assert_settled(queued_done)
        self.assertIsNone(reg.run_started_at('busy'),
                          "a finished job must not keep reporting a start time")

    def test_handled_failure_is_delivered_once_with_its_status(self):
        def job():
            raise SummaryJobFailed('Input too large', http_status=400)

        _, done = self._submit(self.reg, 'k', job)
        self._assert_settled(done)

        self.assertEqual(self.reg.take_error('k'), (400, 'Input too large'))
        self.assertIsNone(self.reg.take_error('k'), "an error must not be delivered twice")

    def test_unexpected_exception_becomes_a_500(self):
        def job():
            raise RuntimeError('provider exploded')

        _, done = self._submit(self.reg, 'k', job)
        self._assert_settled(done)

        status, message = self.reg.take_error('k')
        self.assertEqual(status, 500)
        self.assertIn('provider exploded', message)

    def test_failure_does_not_leave_the_key_pending(self):
        """Otherwise the watch is permanently un-summarisable until restart."""
        _, done = self._submit(self.reg, 'k', lambda: (_ for _ in ()).throw(RuntimeError('boom')))
        self._assert_settled(done)
        self.assertFalse(self.reg.is_pending('k'))

    def test_a_new_submit_clears_a_stale_error(self):
        _, failed_done = self._submit(self.reg, 'k', lambda: (_ for _ in ()).throw(RuntimeError('boom')))
        self._assert_settled(failed_done)

        _, retry_done = self._submit(self.reg, 'k', lambda: None)
        self._assert_settled(retry_done)
        self.assertIsNone(self.reg.take_error('k'),
                          "a retry that succeeded must not report the previous failure")

    def test_distinct_keys_are_independent(self):
        runs = []
        release = threading.Event()

        def job(name):
            runs.append(name)
            release.wait(5)

        accepted_a, done_a = self._submit(self.reg, 'watch-a', lambda: job('a'))
        accepted_b, done_b = self._submit(self.reg, 'watch-b', lambda: job('b'))
        self.assertTrue(accepted_a)
        self.assertTrue(accepted_b, "a different version pair / prompt must not be suppressed")
        release.set()
        self._assert_settled(done_a)
        self._assert_settled(done_b)
        self.assertEqual(sorted(runs), ['a', 'b'])

    def test_on_settled_fires_after_the_result_is_readable(self):
        """The Socket.IO announcement must not arrive before the result it announces.

        The browser answers this event with a single fetch (it does not poll in realtime mode), so
        if the event were sent while the key still looked pending, that one fetch would come back
        'pending' and the client would wait for an event that had already been sent.
        """
        observed = {}

        def job():
            pass

        def on_settled(key):
            observed['pending_at_announce'] = self.reg.is_pending(key)
            observed['key'] = key

        _, done = self._submit(self.reg, 'k', job, on_settled=on_settled)
        self._assert_settled(done)

        self.assertEqual(observed.get('key'), 'k')
        self.assertFalse(observed.get('pending_at_announce'),
                         "announced while the job still looked pending")

    def test_on_settled_fires_on_failure_too(self):
        """Otherwise a failed generation would only surface via the slow watchdog re-check."""
        seen = []

        def job():
            raise SummaryJobFailed('nope', http_status=400)

        _, done = self._submit(self.reg, 'k', job, on_settled=lambda key: seen.append(key))
        self._assert_settled(done)

        self.assertEqual(seen, ['k'])
        self.assertEqual(self.reg.take_error('k'), (400, 'nope'),
                         "the error must be readable by the time the client is told to look")

    def test_a_broken_on_settled_does_not_break_the_job(self):
        _, done = self._submit(self.reg, 'k', lambda: None, on_settled=lambda key: 1 / 0)
        self._assert_settled(done)
        self.assertFalse(self.reg.is_pending('k'))
        self.assertIsNone(self.reg.take_error('k'),
                          "a failed announcement must not be reported as a generation failure")

    def test_no_threads_are_started_at_construction(self):
        """Pool is lazy - import/construct must not spawn threads that outlive gc.freeze()."""
        reg = SummaryJobRegistry()
        self.assertIsNone(reg._pool)
        _, done = self._submit(reg, 'k', lambda: None)
        self._assert_settled(done)
        self.assertIsNotNone(reg._pool)

    def test_distinct_job_is_rejected_at_capacity(self):
        reg = SummaryJobRegistry(max_workers=1, max_pending=2)
        release = threading.Event()
        _, first_done = self._submit(reg, 'first', lambda: release.wait(5))
        _, second_done = self._submit(reg, 'second', lambda: release.wait(5))

        with self.assertRaises(SummaryQueueFullError):
            reg.submit('third', lambda: None)

        release.set()
        self._assert_settled(first_done)
        self._assert_settled(second_done)

    def test_duplicate_is_still_deduplicated_at_capacity(self):
        reg = SummaryJobRegistry(max_workers=1, max_pending=1)
        release = threading.Event()
        accepted, done = self._submit(reg, 'same', lambda: release.wait(5))

        self.assertTrue(accepted)
        self.assertFalse(reg.submit('same', lambda: self.fail("duplicate job ran")))

        release.set()
        self._assert_settled(done)

    def test_concurrent_distinct_admission_never_exceeds_capacity(self):
        max_pending = 3
        contender_count = 8
        reg = SummaryJobRegistry(max_workers=1, max_pending=max_pending)
        release = threading.Event()
        start = threading.Barrier(contender_count)
        results = []
        accepted_keys = []
        results_lock = threading.Lock()
        settled = {f'key-{index}': threading.Event() for index in range(contender_count)}

        def contend(index):
            key = f'key-{index}'
            start.wait()
            try:
                accepted = reg.submit(
                    key,
                    lambda: release.wait(5),
                    on_settled=lambda settled_key: settled[settled_key].set(),
                )
            except SummaryQueueFullError:
                result = 'full'
            else:
                result = 'accepted' if accepted else 'duplicate'
            with results_lock:
                results.append(result)
                if result == 'accepted':
                    accepted_keys.append(key)

        threads = [threading.Thread(target=contend, args=(index,)) for index in range(contender_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(5)
            self.assertFalse(thread.is_alive(), "concurrent submitter did not finish")

        self.assertEqual(results.count('accepted'), max_pending)
        self.assertEqual(results.count('full'), contender_count - max_pending)
        self.assertNotIn('duplicate', results)

        release.set()
        for key in accepted_keys:
            self.assertTrue(settled[key].wait(5), f"accepted job {key} did not settle")

    def test_completion_and_failure_release_capacity(self):
        reg = SummaryJobRegistry(max_workers=1, max_pending=1)
        release = threading.Event()
        _, completed = self._submit(reg, 'completed', lambda: release.wait(5))
        with self.assertRaises(SummaryQueueFullError):
            reg.submit('blocked', lambda: None)
        release.set()
        self._assert_settled(completed)

        _, recovered = self._submit(reg, 'after-completion', lambda: None)
        self._assert_settled(recovered)

        def fail():
            raise RuntimeError('expected failure')

        _, failed = self._submit(reg, 'failed', fail)
        self._assert_settled(failed)
        self.assertEqual(reg.take_error('failed'), (500, 'expected failure'))

        _, recovered_after_failure = self._submit(reg, 'after-failure', lambda: None)
        self._assert_settled(recovered_after_failure)


if __name__ == '__main__':
    unittest.main()
