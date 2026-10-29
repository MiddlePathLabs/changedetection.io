"""
Keyless local (RFC1918 / loopback) LLM endpoints.

LAN self-hosted servers (llama.cpp, vLLM, LM Studio, ...) commonly run without an
API key, but litellm's OpenAI path always attaches an Authorization header — and
token-validating servers reject the placeholder sent for a missing key with
"Invalid token payload". The client therefore swaps a keyless local call to a
tokenless "Bearer", and the settings model listing (model_list.list_models) sends
no auth header at all.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _fake_litellm_response(text='ready'):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, parts=None),
                                 finish_reason='stop')],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
    )


class TestIsLocalLLMEndpoint:
    @pytest.mark.parametrize('api_base', [
        'http://192.168.40.56:8888/v1',
        'http://10.1.2.3:8000',
        'http://172.16.0.9/v1',
        'http://127.0.0.1:11434',
        'http://localhost:8888/v1',
    ])
    def test_private_and_loopback_hosts_are_local(self, api_base):
        from changedetectionio.llm.client import is_local_llm_endpoint
        assert is_local_llm_endpoint(api_base)

    @pytest.mark.parametrize('api_base', [
        '',                      # cloud providers configured without an api_base
        None,
        'https://api.openai.com/v1',
    ])
    def test_cloud_and_empty_are_not_local(self, api_base):
        from changedetectionio.llm.client import is_local_llm_endpoint
        assert not is_local_llm_endpoint(api_base)


class TestKeylessCompletionHeader:
    def test_keyless_local_gets_tokenless_bearer(self):
        from changedetectionio.llm.client import completion
        with patch('litellm.completion', return_value=_fake_litellm_response()) as mock:
            completion(model='openai/probe-model', api_base='http://192.168.40.56:8888/v1',
                       api_key=None, messages=[{'role': 'user', 'content': 'hi'}])
        kwargs = mock.call_args.kwargs
        # Placeholder satisfies the OpenAI SDK's credential check; the header override
        # replaces the token on the wire with a tokenless "Bearer".
        assert kwargs['api_key'] == 'local-endpoint-no-key'
        assert kwargs['extra_headers'] == {'Authorization': 'Bearer'}

    def test_keyed_local_endpoint_is_untouched(self):
        from changedetectionio.llm.client import completion
        with patch('litellm.completion', return_value=_fake_litellm_response()) as mock:
            completion(model='openai/probe-model', api_base='http://192.168.40.56:8888/v1',
                       api_key='sk-real', messages=[{'role': 'user', 'content': 'hi'}])
        kwargs = mock.call_args.kwargs
        assert kwargs['api_key'] == 'sk-real'
        assert 'Authorization' not in (kwargs.get('extra_headers') or {})

    def test_keyless_cloud_endpoint_gets_no_injection(self):
        # Cloud providers always need a real key; a blank one must fail at the
        # provider rather than silently sending a tokenless Bearer upstream.
        from changedetectionio.llm.client import completion
        with patch('litellm.completion', return_value=_fake_litellm_response()) as mock:
            completion(model='openai/gpt-4o-mini', api_base='https://api.openai.com/v1',
                       api_key=None, messages=[{'role': 'user', 'content': 'hi'}])
        kwargs = mock.call_args.kwargs
        assert 'api_key' not in kwargs
        assert 'Authorization' not in (kwargs.get('extra_headers') or {})


class TestKeylessModelsFetch:
    # Model listing goes through model_list.list_models(), which only sends an
    # Authorization header when a key was given.
    def _call(self, api_base='http://192.168.40.56:8888/v1', status=200, payload=None):
        from changedetectionio.llm.model_list import list_models
        with patch('requests.get') as mock_get:
            mock_get.return_value = SimpleNamespace(
                status_code=status,
                json=lambda: payload if payload is not None else
                    {'data': [{'id': 'my-model'}, {'id': 'vendor/other-model'}]},
                text='server said no',
            )
            return list_models('openai_compatible', api_key=None, api_base=api_base), mock_get

    def test_fetches_v1_models_without_auth(self):
        models, mock_get = self._call()
        args, kwargs = mock_get.call_args
        assert args[0] == 'http://192.168.40.56:8888/v1/models'
        assert not kwargs.get('headers'), "no Authorization header may be sent"
        assert models == ['openai/my-model', 'openai/vendor/other-model']

    def test_error_payload_is_surfaced(self):
        from changedetectionio.llm.model_list import ModelListError
        with pytest.raises(ModelListError, match='Invalid token payload'):
            self._call(status=401, payload={'error': {'message': 'Invalid token payload'}})
