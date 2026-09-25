"""
Retry behaviour of llm.client.completion():
  - transient provider errors (429 / 5xx / overloaded / connection) are retried with backoff
  - timeouts are NOT retried (a retry just multiplied the wait)
  - an empty reply cut off by max_tokens is retried once with more headroom
  - context-window errors are not "fixed" by stripping temperature
"""
import unittest
from unittest.mock import MagicMock, patch

import litellm

from changedetectionio.llm import client as m


def _resp(text='ok', finish='stop', total=10, prompt=6, completion=4):
    r = MagicMock()
    r.choices = [MagicMock(message=MagicMock(content=text, parts=None), finish_reason=finish)]
    r.usage = MagicMock(total_tokens=total, prompt_tokens=prompt, completion_tokens=completion)
    return r


def _http_response(status, headers=None):
    return MagicMock(status_code=status, headers=headers or {})


MSGS = [{'role': 'user', 'content': 'hi'}]


@patch('changedetectionio.llm.client.time.sleep')
class TestTransientRetries(unittest.TestCase):
    def test_rate_limit_is_retried(self, sleep):
        exc = litellm.RateLimitError(message='429', llm_provider='openai', model='gpt-4o-mini',
                                     response=_http_response(429))
        with patch('litellm.completion', side_effect=[exc, _resp('done')]) as call:
            text, *_ = m.completion(model='gpt-4o-mini', messages=MSGS)
        self.assertEqual(text, 'done')
        self.assertEqual(call.call_count, 2)
        sleep.assert_called_once()

    def test_retry_after_header_is_honoured(self, sleep):
        exc = litellm.RateLimitError(message='429', llm_provider='openai', model='gpt-4o-mini',
                                     response=_http_response(429, {'retry-after': '7'}))
        with patch('litellm.completion', side_effect=[exc, _resp()]):
            m.completion(model='gpt-4o-mini', messages=MSGS)
        sleep.assert_called_once_with(7.0)

    def test_overloaded_is_retried(self, sleep):
        exc = litellm.ServiceUnavailableError(message='overloaded', llm_provider='anthropic',
                                              model='claude-sonnet-4-5', response=_http_response(503))
        with patch('litellm.completion', side_effect=[exc, exc, _resp('third time')]) as call:
            text, *_ = m.completion(model='claude-sonnet-4-5', messages=MSGS)
        self.assertEqual(text, 'third time')
        self.assertEqual(call.call_count, 3)

    def test_gives_up_after_default_retries(self, sleep):
        exc = litellm.InternalServerError(message='500', llm_provider='openai', model='gpt-4o-mini')
        with patch('litellm.completion', side_effect=exc) as call:
            with self.assertRaises(litellm.InternalServerError):
                m.completion(model='gpt-4o-mini', messages=MSGS)
        self.assertEqual(call.call_count, m.DEFAULT_RETRIES)

    def test_timeout_is_not_retried(self, sleep):
        exc = litellm.Timeout(message='Request timed out after None seconds',
                              model='gpt-4o-mini', llm_provider='openai')
        with patch('litellm.completion', side_effect=exc) as call:
            with self.assertRaises(litellm.Timeout) as ctx:
                m.completion(model='gpt-4o-mini', messages=MSGS, timeout=42)
        self.assertEqual(call.call_count, 1)
        self.assertIn('after 42 seconds', str(ctx.exception))

    def test_context_window_error_is_not_retried(self, sleep):
        exc = litellm.ContextWindowExceededError(message='too long', model='gpt-4o-mini',
                                                 llm_provider='openai')
        with patch('litellm.completion', side_effect=exc) as call:
            with self.assertRaises(litellm.ContextWindowExceededError):
                m.completion(model='gpt-4o-mini', messages=MSGS)
        self.assertEqual(call.call_count, 1)


class TestEmptyLengthRetry(unittest.TestCase):
    def test_empty_reply_cut_by_max_tokens_is_retried_with_more_room(self):
        with patch('litellm.completion', side_effect=[
            _resp('', finish='length', total=500, prompt=100, completion=400),
            _resp('{"important": true}', total=900, prompt=100, completion=800),
        ]) as call:
            text, total, tin, tout = m.completion(model='deepseek/deepseek-reasoner', messages=MSGS,
                                                  max_tokens=400)
        self.assertEqual(text, '{"important": true}')
        self.assertEqual(call.call_count, 2)
        self.assertGreater(call.call_args_list[1].kwargs['max_tokens'], 400)
        # Both calls were billed.
        self.assertEqual((total, tin, tout), (1400, 200, 1200))

    def test_only_retried_once(self):
        with patch('litellm.completion', return_value=_resp('', finish='length')) as call:
            text, *_ = m.completion(model='x', messages=MSGS, max_tokens=400)
        self.assertEqual(text, '')
        self.assertEqual(call.call_count, 2)

    def test_non_empty_truncated_reply_is_returned_as_is(self):
        with patch('litellm.completion', return_value=_resp('partial', finish='length')) as call:
            text, *_ = m.completion(model='x', messages=MSGS)
        self.assertEqual(text, 'partial')
        self.assertEqual(call.call_count, 1)


class TestExtraBodyNotMutated(unittest.TestCase):
    def test_callers_extra_body_survives_a_400_strip(self):
        extra = {'generationConfig': {'thinkingConfig': {'thinkingBudget': 0}}}
        bad = litellm.BadRequestError(message='invalid', model='gemini/x', llm_provider='gemini',
                                      response=MagicMock(status_code=400))
        with patch('litellm.completion', side_effect=[bad, _resp()]):
            m.completion(model='gemini/gemini-2.5-flash', messages=MSGS, extra_body=extra)
        self.assertEqual(extra, {'generationConfig': {'thinkingConfig': {'thinkingBudget': 0}}})


class TestMaxCallDuration(unittest.TestCase):
    def test_covers_more_than_one_timeout(self):
        self.assertGreater(m.max_call_duration(300), 300)
