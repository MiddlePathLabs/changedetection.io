"""
Thin wrapper around litellm.completion.
Keeps litellm import isolated so the rest of the codebase doesn't depend on it directly,
and makes the call easy to mock in tests.
"""

import copy
import logging
import os
import time
from dataclasses import dataclass

from loguru import logger

# Default output token cap for JSON-returning calls (intent eval, preview, setup).
# These return small JSON objects — 400 is enough for a verbose explanation while
# still preventing runaway cost. Change summaries pass their own max_tokens via
# _summary_max_tokens() and are NOT subject to this cap.
_MAX_COMPLETION_TOKENS = 400

# Default request timeout (seconds). Raised from 60 to 300 because even cloud
# reasoning models can be slow on the first hit (issue #4225). Overridable via
# LLM_TIMEOUT.
DEFAULT_TIMEOUT = int(os.getenv('LLM_TIMEOUT', 300))
# Relaxed timeout for local / self-hosted endpoints (Ollama, vLLM, LM Studio,
# llama.cpp on localhost or a LAN address). These run on modest hardware and can
# spend many minutes on prompt prefill before the first token, so they get a much
# longer deadline (Hermes-style, 30 min). Overridable via LLM_LOCAL_TIMEOUT; see
# evaluator.resolve_llm_timeout() for how the endpoint is classified.
DEFAULT_LOCAL_TIMEOUT = int(os.getenv('LLM_LOCAL_TIMEOUT', 1800))
# Models and reasoning architectures that reject explicit sampling parameters (temperature/top_p).
# Substring match against the lowercased model name, so 'gpt-5' also covers gpt-5-mini, gpt-5.1 etc.
#
# This is only an optimisation - the BadRequestError handler below strips and retries for
# anything not listed. Being listed just avoids paying a rejected round trip on every call.
#
# NOT all GPT models: gpt-4o, gpt-4-turbo and gpt-3.5-turbo take temperature normally. It is
# reasoning that forbids it. gpt-5 reports "only temperature=1 is supported unless
# reasoning_effort resolves to 'none'" - we never set reasoning_effort=none, so it is always
# in that state for us. Claude 5.x (haiku/sonnet/opus/fable) refuses sampling params itself, not
# because of thinking: Opus 5/5.5, Sonnet 5 and Fable reject any temperature, Haiku 5.5 and
# Sonnet 5.5 any non-default one (which our 0 is).
_NO_TEMPERATURE_MODEL_KEYWORDS = ('flash-lite', 'thinking-exp', 'o1', 'o3', 'o4', 'gpt-5',
                                  'haiku-5', 'sonnet-5', 'opus-5', 'fable-5')

DEFAULT_RETRIES = 3

# Backoff between retries of transient failures (connection reset, 429, 5xx/overloaded).
# Exponential from _RETRY_BACKOFF_BASE, capped; a provider's Retry-After wins when present.
_RETRY_BACKOFF_BASE = 2.0
_RETRY_BACKOFF_MAX = 30.0

# When a reply comes back empty with finish_reason='length' the model spent its whole output
# budget reasoning and never reached the answer. Retry once with this much more headroom
# (capped) instead of handing the caller an empty string.
_EMPTY_LENGTH_RETRY_MULTIPLIER = 4
_EMPTY_LENGTH_RETRY_MAX_TOKENS = 32_000


@dataclass
class CompletionExecution:
    attempts: int = 0


@dataclass(frozen=True)
class CompletionResult:
    """The provider response and the usage facts the provider actually returned."""

    text: str
    finish_reason: str | None
    total_tokens: int
    input_tokens: int
    output_tokens: int
    usage_reported: bool

    @property
    def usage(self) -> dict:
        """A serialisable usage view for callers that need one aggregate object."""
        return {
            'total_tokens': self.total_tokens,
            'input_tokens': self.input_tokens,
            'output_tokens': self.output_tokens,
            'reported': self.usage_reported,
        }


def max_call_duration(timeout: int) -> int:
    """Worst-case wall time of one completion() call, for UI deadlines.

    Timeouts are not retried, but a request can still time out after transient-error retries
    and after the one empty-reply retry, so allow for the slowest realistic path: two full
    timeouts plus every backoff sleep.
    """
    backoff = sum(min(_RETRY_BACKOFF_BASE ** n, _RETRY_BACKOFF_MAX) for n in range(1, DEFAULT_RETRIES))
    return int(timeout * 2 + backoff)


def is_local_llm_endpoint(api_base: str) -> bool:
    """
    True when `api_base`'s host is IANA-restricted (private / loopback / link-local /
    reserved) — i.e. a local or LAN self-hosted LLM (Ollama, vLLM, LM Studio,
    llama.cpp, ...).

    Reuses the same is_private_hostname check the SSRF guard uses, so DNS names
    that resolve into the LAN (docker service names, host.docker.internal) count
    as local too. Empty/cloud api_base → False.
    """
    api_base = (api_base or '').strip()
    if not api_base:
        return False
    try:
        from urllib.parse import urlparse
        from changedetectionio.validate_url import is_private_hostname
        host = urlparse(api_base).hostname
        return bool(host) and is_private_hostname(host)
    except Exception:
        # Never let endpoint classification break an LLM call — fall back to "not local".
        return False


def _retry_after_seconds(exc) -> float | None:
    """Retry-After (seconds) from a provider error's response headers, if it sent one."""
    try:
        headers = getattr(getattr(exc, 'response', None), 'headers', None) or {}
        value = headers.get('retry-after') or headers.get('Retry-After')
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _backoff(attempt: int, exc=None) -> float:
    after = _retry_after_seconds(exc)
    if after is not None and after >= 0:
        return min(after, _RETRY_BACKOFF_MAX * 2)
    return min(_RETRY_BACKOFF_BASE ** attempt, _RETRY_BACKOFF_MAX)


class _LoguruInterceptHandler(logging.Handler):
    # Routes litellm's stdlib log records through loguru so debug output
    # uses the same format/sink as the rest of the app.
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except (ValueError, AttributeError):
            level = record.levelno
        logger.opt(exception=record.exc_info).log(level, record.getMessage())


_debug_installed = False


def _install_litellm_debug():
    # Attach our loguru intercept and clear any pre-existing handlers so litellm's
    # own stdout StreamHandler (installed by _turn_on_debug / set_verbose) doesn't
    # double-emit. Setting the logger level to DEBUG is enough to make litellm
    # produce debug records — we don't call _turn_on_debug() for that reason.
    global _debug_installed
    if _debug_installed:
        return

    handler = _LoguruInterceptHandler()
    handler.setLevel(logging.DEBUG)
    for _name in ('LiteLLM', 'litellm', 'litellm.utils', 'litellm.router'):
        _lg = logging.getLogger(_name)
        _lg.handlers = []
        _lg.setLevel(logging.DEBUG)
        _lg.addHandler(handler)
        _lg.propagate = False

    _debug_installed = True
    logger.info("LLM client: litellm debug logging routed through loguru")


def completion(  # noqa: C901
    model: str,
    messages: list,
    api_key: str = None,
    api_base: str = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_tokens: int = None,
    extra_body: dict = None,
    debug: bool = False,
    execution: CompletionExecution | None = None,
) -> CompletionResult:
    """
    Call the LLM and return a :class:`CompletionResult`.
    Retries up to DEFAULT_RETRIES times (with backoff) on connection errors, rate limits
    and provider 5xx/overloaded errors. A timeout is NOT retried: a model too slow to answer
    within the deadline is just as slow the second time, and retrying multiplied the wait.
    An empty reply cut off by max_tokens (reasoning models) is retried once with more room.
    Token counts are 0 if the provider doesn't return usage data; ``usage_reported``
    distinguishes that case from a genuine zero-token response.
    Raises on network/auth errors — callers handle gracefully.

    timeout: seconds for the request. Local endpoints get a longer value than cloud —
    see evaluator.resolve_llm_timeout().
    """
    try:
        import litellm
    except ImportError:
        raise RuntimeError("litellm is not installed. Add it to requirements.txt.") from None

    if debug:
        _install_litellm_debug()

    _timeout = timeout if timeout is not None else DEFAULT_TIMEOUT

    kwargs = {
        'model': model,
        'messages': messages,
        'timeout': _timeout,
        'max_tokens': max_tokens if max_tokens is not None else _MAX_COMPLETION_TOKENS,
    }
    _m_lower = (model or '').lower()
    if not any(k in _m_lower for k in _NO_TEMPERATURE_MODEL_KEYWORDS):
        kwargs['temperature'] = 0

    if api_key:
        kwargs['api_key'] = api_key
    if api_base:
        kwargs['api_base'] = api_base
        if not api_key and is_local_llm_endpoint(api_base):
            # Keyless local endpoint (llama.cpp / vLLM / LM Studio on the LAN). Two problems
            # with just omitting the key: the OpenAI client refuses to build without a
            # credential, and a placeholder token ("Bearer anonymous") is rejected by servers
            # that validate whatever token they are handed. A tokenless "Bearer" satisfies
            # both — the SDK sees a key, servers that check tokens see an anonymous request,
            # and servers without auth ignore the header entirely.
            kwargs['api_key'] = 'local-endpoint-no-key'
            kwargs['extra_headers'] = {'Authorization': 'Bearer'}
            logger.debug(f"LLM client: keyless local endpoint {api_base!r} — sending tokenless Bearer")
    if extra_body:
        extra_body = copy.deepcopy(extra_body)
        # Anthropic takes thinking as a top-level field and 400s on extra_body ("Extra inputs are not permitted")
        if 'thinking' in extra_body and (_m_lower.startswith('claude') or _m_lower.startswith('anthropic/')):
            kwargs['thinking'] = extra_body.pop('thinking')
        if extra_body:
            kwargs['extra_body'] = extra_body

    _retryable = tuple(
        cls for cls in (
            getattr(litellm, 'APIConnectionError', None),
            getattr(litellm, 'RateLimitError', None),
            getattr(litellm, 'InternalServerError', None),
            getattr(litellm, 'ServiceUnavailableError', None),
        ) if isinstance(cls, type)
    )
    _timeout_exc = getattr(litellm, 'Timeout', None)
    _context_exc = getattr(litellm, 'ContextWindowExceededError', None)
    _expanded_for_empty = False
    _accum_total = _accum_in = _accum_out = 0
    _usage_reported = False

    # Some models reject sampling params outright: Anthropic Claude Opus 4.7/4.8 and
    # Fable return HTTP 400 for 'temperature', and OpenAI reasoning models (o1/o3/gpt-5)
    # only accept the default. litellm's per-model param metadata lags new releases, so
    # drop_params can't be relied on for freshly released models — instead, if the provider
    # rejects a sampling param, strip them and retry once. Models that accept them are
    # unaffected (they still receive temperature=0).
    _sampling_params = ('temperature', 'top_p', 'top_k')
    _stripped_sampling = False

    logger.debug(
        f"LLM client: calling model={model!r} api_base={api_base!r} "
        f"timeout={_timeout}s max_tokens={kwargs['max_tokens']}"
    )
    logger.trace(messages)

    attempt = 0
    while attempt < DEFAULT_RETRIES:
        attempt += 1
        try:
            if execution is not None:
                execution.attempts += 1
            response = litellm.completion(**kwargs)
            choice = response.choices[0]
            message = choice.message
            finish = getattr(choice, 'finish_reason', None)

            text = message.content or ''

            if not text:
                # Some providers (e.g. Gemini) put text in message.parts instead of .content
                parts = getattr(message, 'parts', None)
                if parts:
                    text = ''.join(getattr(p, 'text', '') or '' for p in parts).strip()
                    logger.debug(
                        f"LLM client: extracted text from message.parts ({len(parts)} parts) model={model!r}"
                    )

            usage = getattr(response, 'usage', None)
            _usage_reported = _usage_reported or usage is not None
            input_tokens = int(getattr(usage, 'prompt_tokens', 0) or 0) if usage else 0
            output_tokens = int(getattr(usage, 'completion_tokens', 0) or 0) if usage else 0
            total_tokens = (
                int(getattr(usage, 'total_tokens', 0) or 0)
                if usage
                else (input_tokens + output_tokens)
            )
            # Include tokens billed by an earlier empty/truncated attempt so budgets stay honest.
            _accum_total += total_tokens
            _accum_in += input_tokens
            _accum_out += output_tokens

            if not text and finish == 'length' and not _expanded_for_empty:
                # The model reasoned until it hit max_tokens and never produced an answer.
                # Handing back '' makes every caller fall through to a default, so give it
                # one more go with real headroom.
                _expanded_for_empty = True
                _old = kwargs['max_tokens']
                kwargs['max_tokens'] = min(
                    max(_old * _EMPTY_LENGTH_RETRY_MULTIPLIER, _old + 2000),
                    _EMPTY_LENGTH_RETRY_MAX_TOKENS,
                )
                attempt -= 1
                logger.warning(
                    f"LLM client: empty reply with finish_reason='length' model={model!r} "
                    f"(likely reasoning used the whole budget) — retrying once with "
                    f"max_tokens {_old} -> {kwargs['max_tokens']}"
                )
                continue

            if finish == 'length':
                logger.warning(
                    f"LLM client: response truncated (finish_reason='length') model={model!r} "
                    f"max_tokens={kwargs['max_tokens']} — got {len(text)} chars"
                )

            if not text:
                logger.warning(
                    f"LLM client: empty content from model={model!r} "
                    f"finish_reason={finish!r} "
                    f"message={message!r}"
                )

            logger.debug(
                f"LLM client: model={model!r} finish={finish!r} "
                f"tokens={_accum_total} (in={_accum_in} out={_accum_out}) "
                f"text_len={len(text)}"
            )
            return CompletionResult(
                text=text,
                finish_reason=finish,
                total_tokens=_accum_total,
                input_tokens=_accum_in,
                output_tokens=_accum_out,
                usage_reported=_usage_reported,
            )

        except Exception as e:
            if _timeout_exc is not None and isinstance(e, _timeout_exc):
                # litellm formats its Timeout message with None when the provider doesn't
                # propagate the timeout value — patch the exception args in-place so every
                # caller that logs str(e) sees the real number.
                _fix = f'after {_timeout} seconds'
                try:
                    e.args = tuple(str(a).replace('after None seconds', _fix) for a in e.args)
                    # litellm's __str__ reads .message, not .args
                    if isinstance(getattr(e, 'message', None), str):
                        e.message = e.message.replace('after None seconds', _fix)
                except Exception:
                    pass
                logger.warning(f"LLM call timed out after {_timeout}s model={model!r} error={e}")
                raise

            if _retryable and isinstance(e, _retryable):
                if attempt < DEFAULT_RETRIES:
                    _sleep = _backoff(attempt, e)
                    logger.warning(
                        f"LLM call transient error {type(e).__name__} (attempt {attempt}/{DEFAULT_RETRIES}), "
                        f"retrying in {_sleep:.0f}s — model={model!r} error={e}"
                    )
                    time.sleep(_sleep)
                    continue
                logger.warning(
                    f"LLM call failed after {DEFAULT_RETRIES} attempts "
                    f"model={model!r} error={e}"
                )
                raise

            if not isinstance(e, litellm.BadRequestError):
                logger.warning(f"LLM call failed: model={model!r} error={e}")
                raise

            if _context_exc is not None and isinstance(e, _context_exc):
                # Stripping temperature can't make an oversized prompt fit.
                logger.warning(f"LLM call failed: prompt exceeds context window model={model!r} error={e}")
                raise

            # If the provider rejected an unsupported sampling param or extra_body
            # (e.g. Gemini INVALID_ARGUMENT on thinkingConfig or temperature), drop
            # them and retry once.
            if not _stripped_sampling:
                dropped = [p for p in _sampling_params if kwargs.pop(p, None) is not None]
                # Only when the error is about it, so an unrelated 400 (prompt too long) isn't retried
                if 'thinking' in kwargs and 'thinking' in str(e).lower():
                    kwargs.pop('thinking')
                    dropped.append('thinking')
                extra_body = kwargs.get('extra_body')
                if isinstance(extra_body, dict):
                    gen_cfg = extra_body.get('generationConfig')
                    if (
                        isinstance(gen_cfg, dict)
                        and gen_cfg.pop('thinkingConfig', None) is not None
                    ):
                        dropped.append('thinkingConfig')
                        if not gen_cfg:
                            extra_body.pop('generationConfig', None)
                        if not extra_body:
                            kwargs.pop('extra_body', None)
                    elif 'thinkingConfig' in extra_body:
                        extra_body.pop('thinkingConfig', None)
                        dropped.append('thinkingConfig')
                        if not extra_body:
                            kwargs.pop('extra_body', None)

                if dropped:
                    _stripped_sampling = True
                    attempt -= 1
                    logger.warning(
                        f"LLM client: model={model!r} rejected request ({e}); "
                        f"stripped {dropped} and retrying once"
                    )
                    continue
            logger.warning(f"LLM call failed: model={model!r} error={e}")
            raise


def _completion_worker(conn, log_level):
    # Runs in a spawned child: litellm is imported here, and all of it (~200MB RSS) goes
    # back to the OS when the process exits.
    import json
    import sys
    from dataclasses import asdict
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    execution = CompletionExecution()
    try:
        kwargs = json.loads(conn.recv_bytes().decode('utf-8'))
        result = {'ok': True, 'result': asdict(completion(execution=execution, **kwargs))}
    except Exception as e:
        result = {'ok': False, 'error': str(e), 'error_type': type(e).__name__}
    result['attempts'] = execution.attempts
    try:
        conn.send_bytes(json.dumps(result).encode('utf-8'))
    finally:
        conn.close()


def completion_isolated(execution: CompletionExecution | None = None, **kwargs) -> CompletionResult:
    """Same as completion(), but runs it in a short-lived subprocess when litellm isn't loaded yet.

    For one-off calls like the settings "Test connection" button, so trying out a provider
    doesn't leave litellm resident for the life of the process. If litellm is already
    imported (AI is configured and in use) that memory is spent anyway, so just call
    completion() directly. Linux only, same as the extruct isolation in restock_diff.

    Errors from the child are re-raised as RuntimeError carrying the original message.
    """
    import json
    import platform
    import sys

    if 'litellm' in sys.modules or platform.system() != 'Linux':
        return completion(execution=execution, **kwargs)

    import multiprocessing
    try:
        log_level = logger._core.min_level
    except AttributeError:
        log_level = 'DEBUG'

    ctx = multiprocessing.get_context('spawn')
    parent_conn, child_conn = ctx.Pipe()
    p = ctx.Process(target=_completion_worker, args=(child_conn, log_level), daemon=True)
    p.start()
    child_conn.close()
    try:
        parent_conn.send_bytes(json.dumps(kwargs).encode('utf-8'))
        # completion() retries timeouts/connection errors, so allow for every attempt
        # plus the child's start-up (importing litellm takes a few seconds)
        _timeout = kwargs.get('timeout') or DEFAULT_TIMEOUT
        deadline = _timeout * (DEFAULT_RETRIES + 1) + 60
        if not parent_conn.poll(deadline):
            raise TimeoutError(f"LLM call did not finish within {deadline}s")
        try:
            result = json.loads(parent_conn.recv_bytes().decode('utf-8'))
        except EOFError:
            raise RuntimeError("LLM subprocess exited without a result, check the logs") from None
    finally:
        parent_conn.close()
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
            p.join()

    if execution is not None:
        execution.attempts += result.get('attempts', 0)
    if not result['ok']:
        raise RuntimeError(result['error'])
    return CompletionResult(**result['result'])
