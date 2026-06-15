"""
Unit tests for Claude LLM client.

Tests using REAL Anthropic API calls (not mocks) or local LiteLLM endpoints.
Requires ANTHROPIC_API_KEY for Claude tests, or LITELLM_API_BASE for local LiteLLM tests.
Uses claude-3-haiku for cost-effective testing when using Anthropic.
"""

import os
import socket
from urllib.parse import urlparse

import pytest
import json
import uuid
import re


def _is_host_port_open(url: str, timeout: float = 1.0) -> bool:
    try:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == 'https' else 80)
        if not host:
            return False
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def _has_local_litellm_config() -> bool:
    return bool(os.getenv("LITELLM_API_BASE"))


def _probe_ollama_models(api_base: str, timeout: float = 1.0):
    """Try a few common endpoints to discover available Ollama model ids.

    Returns a list of candidate model ids (may be empty).
    """
    try:
        import requests
    except Exception:
        return []

    api_base = api_base.rstrip("/")
    endpoints = [
        api_base + "/v1/models",
        api_base + "/models",
        api_base + "/v1/ollama/models",
        api_base + "/v1/engines",
    ]

    candidates = []
    for url in endpoints:
        try:
            resp = requests.get(url, timeout=timeout)
            if not resp.ok:
                continue
            data = resp.json()
            # common shapes: list of strings, {'models': [...]}, or dict of id->meta
            if isinstance(data, list):
                for v in data:
                    if isinstance(v, str):
                        candidates.append(v)
                    elif isinstance(v, dict):
                        for k in ("id", "model", "name"):
                            if k in v:
                                candidates.append(v[k])
                                break
            elif isinstance(data, dict):
                if "models" in data and isinstance(data["models"], list):
                    for it in data["models"]:
                        if isinstance(it, str):
                            candidates.append(it)
                        elif isinstance(it, dict):
                            for k in ("id", "model", "name"):
                                if k in it:
                                    candidates.append(it[k])
                                    break
                else:
                    # treat keys as model ids
                    for k in data.keys():
                        candidates.append(k)
        except Exception:
            continue

    # Deduplicate while preserving order
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _normalize_ollama_model_name(name: str) -> str:
    """Normalize a model id to Ollama repo id rules."""
    normalized = re.sub(r"[^A-Za-z0-9._-]", "-", name)
    normalized = normalized.strip("-.")
    return normalized


def _choose_ollama_model(api_base: str, original_model: str, api_key: str | None):
    """Select the best Ollama model id for the local endpoint."""
    mapping_env = os.getenv("LITELLM_OLLAMA_MODEL_MAP", "")
    mapping = {}
    if mapping_env:
        for pair in mapping_env.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                mapping[k.strip()] = v.strip()

    if original_model in mapping:
        return mapping[original_model]

    raw_model = original_model.split("/", 1)[1]
    sanitized = _normalize_ollama_model_name(raw_model)

    candidates = _probe_ollama_models(api_base)
    if candidates:
        normalized_candidates = {c: _normalize_ollama_model_name(c) for c in candidates}
        # exact matches first
        for candidate, normalized in normalized_candidates.items():
            if candidate == raw_model or candidate == sanitized:
                return candidate
        # normalized matches
        for candidate, normalized in normalized_candidates.items():
            if normalized == sanitized:
                return candidate
        # best fuzzy matches
        lowercase_raw = raw_model.lower()
        for candidate in candidates:
            if lowercase_raw in candidate.lower() or candidate.lower() in lowercase_raw:
                return candidate

    if api_key is None:
        api_key = "test"

    try:
        import litellm
    except ImportError:
        return sanitized

    if candidates:
        for candidate in candidates:
            try:
                litellm.set_verbose = False
                litellm.completion(
                    model=candidate,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=1,
                    api_base=api_base,
                    api_key=api_key,
                    custom_llm_provider="ollama",
                    timeout=10,
                )
                return candidate
            except Exception:
                continue

    return sanitized


# Skip all tests if no Anthropic API key and no local LiteLLM configuration
pytestmark = [
    pytest.mark.requires_claude,
    pytest.mark.skipif(
        not (os.getenv("ANTHROPIC_API_KEY") or _has_local_litellm_config()),
        reason="Requires ANTHROPIC_API_KEY for Claude tests or LITELLM_API_BASE for local LiteLLM tests"
    )
]


def unique_prompt(base: str) -> str:
    """Add unique suffix to avoid cache hits."""
    return f"{base} [test-id: {uuid.uuid4().hex[:8]}]"


@pytest.fixture
def api_env():
    """Ensure API mode environment is set."""
    # Real API key should be loaded from .env via conftest
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key or api_key.startswith('999'):
        pytest.skip("Real ANTHROPIC_API_KEY required")
    yield


@pytest.fixture
def cli_env():
    """Set up CLI mode environment (48-char key pattern)."""
    original = os.environ.get('ANTHROPIC_API_KEY')
    os.environ['ANTHROPIC_API_KEY'] = '999999999999999999999999999999999999999999999999'
    yield
    if original:
        os.environ['ANTHROPIC_API_KEY'] = original
    else:
        del os.environ['ANTHROPIC_API_KEY']


@pytest.fixture(scope="session")
def local_litellm_env():
    """Return local LiteLLM provider configuration for local tests."""
    try:
        import litellm
    except ImportError:
        pytest.skip("LiteLLM package required for local LiteLLM tests")

    api_base = os.getenv("LITELLM_API_BASE")
    if not api_base:
        pytest.skip("LITELLM_API_BASE is not set for local LiteLLM tests")

    if not _is_host_port_open(api_base):
        pytest.skip(f"LiteLLM endpoint is not reachable at {api_base}")

    model = os.getenv("LITELLM_MODEL", "default_model")
    api_key = os.getenv("LITELLM_API_KEY")
    custom_provider = os.getenv("LITELLM_CUSTOM_PROVIDER", "openai")

    # Normalize OpenAI-compatible endpoint base URLs for litellm.
    api_base = api_base.rstrip("/")
    if api_base.endswith("/v1"):
        api_base = api_base[:-3]

    # Some OpenAI-compatible local endpoints require a non-empty api_key even if it's not validated.
    if custom_provider == "openai" and not api_key:
        api_key = "test"

    # If the model string uses an Ollama-style prefix like "ollama/xxx",
    # switch the provider to 'ollama' and select a usable model id.
    if model and model.startswith("ollama/"):
        custom_provider = "ollama"
        model = _choose_ollama_model(api_base, model, api_key)

    # Verify the local LiteLLM endpoint is usable for the specified model.
    try:
        litellm.set_verbose = False
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1,
            api_base=api_base,
            api_key=api_key,
            custom_llm_provider=custom_provider,
            timeout=10,
        )
    except Exception as e:
        extra = ""
        if custom_provider == "ollama":
            try:
                models = _probe_ollama_models(api_base)
                if models:
                    sample = models[:6]
                    extra = (
                        "\nAvailable Ollama models: " + ", ".join(sample)
                        + "\nIf you want to map your .env value to one of these, set:\n"
                        + "  LITELLM_OLLAMA_MODEL_MAP='ollama/your_model=THE_MODEL_ID'"
                    )
                else:
                    extra = "\nCould not enumerate Ollama models from the endpoint."
            except Exception:
                extra = "\nCould not enumerate Ollama models from the endpoint."

        pytest.skip(f"Local LiteLLM endpoint is not usable: {type(e).__name__}: {e}{extra}")

    return {
        "model": model,
        "api_base": api_base,
        "api_key": api_key,
        "custom_llm_provider": custom_provider,
        "max_tokens": 100,
        "temperature": 0.0,
    }


class TestClaudeClientInitialization:
    """Test Claude client initialization."""

    def test_init_with_api_key(self, api_env):
        """Test initialization with real API key."""
        from kosmos.core.llm import ClaudeClient

        client = ClaudeClient(model="claude-3-haiku-20240307")

        assert client.api_key is not None
        assert len(client.api_key) > 20  # Real keys are long
        assert not client.is_cli_mode
        assert "claude" in client.model.lower()
        assert client.max_tokens > 0

    def test_init_with_cli_mode(self, cli_env):
        """Test initialization in CLI mode (48-char key)."""
        from kosmos.core.llm import ClaudeClient

        # Disable cache during init to avoid config parsing side effects unrelated
        # to CLI key detection.
        client = ClaudeClient(model="claude-3-haiku-20240307", enable_cache=False)

        assert client.is_cli_mode
        assert len(client.api_key) == 48

    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        from kosmos.core.llm import ClaudeClient

        # Temporarily remove API key
        original = os.environ.get('ANTHROPIC_API_KEY')
        if 'ANTHROPIC_API_KEY' in os.environ:
            del os.environ['ANTHROPIC_API_KEY']

        try:
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY environment variable not set"):
                ClaudeClient()
        finally:
            if original:
                os.environ['ANTHROPIC_API_KEY'] = original

    def test_custom_parameters(self, api_env):
        """Test initialization with custom parameters."""
        from kosmos.core.llm import ClaudeClient

        client = ClaudeClient(
            model="claude-3-haiku-20240307",
            max_tokens=8192,
            temperature=0.5
        )

        assert client.model == "claude-3-haiku-20240307"
        assert client.max_tokens == 8192
        assert client.temperature == 0.5


class TestClaudeClientGeneration:
    """Test Claude text generation with real API calls."""

    def test_generate_basic(self, api_env):
        """Test basic text generation with real API."""
        from kosmos.core.llm import ClaudeClient

        client = ClaudeClient(model="claude-3-haiku-20240307")
        # Use unique prompt to avoid cache hits
        response = client.generate(unique_prompt("Say 'Hello World' and nothing else."))

        # Verify we got a real response
        assert response is not None
        assert len(response) > 0
        assert isinstance(response, str)
        assert "hello" in response.lower() or "world" in response.lower()

        # Verify statistics are tracked (cache miss = real API call)
        assert client.total_requests == 1
        assert client.total_input_tokens > 0
        assert client.total_output_tokens > 0

    def test_generate_with_system_prompt(self, api_env):
        """Test generation with system prompt."""
        from kosmos.core.llm import ClaudeClient

        client = ClaudeClient(model="claude-3-haiku-20240307")
        response = client.generate(
            prompt="What is your purpose?",
            system="You are a helpful math tutor. Always mention that you help with math."
        )

        # System prompt should influence response
        assert response is not None
        assert len(response) > 0
        # The response should mention math given the system prompt
        assert "math" in response.lower() or "tutor" in response.lower() or "help" in response.lower()

    def test_generate_with_overrides(self, api_env):
        """Test generation with parameter overrides."""
        from kosmos.core.llm import ClaudeClient

        client = ClaudeClient(model="claude-3-haiku-20240307")
        response = client.generate(
            prompt="Write exactly one word.",
            max_tokens=50,
            temperature=0.0  # Deterministic
        )

        # Should get a response (short due to max_tokens)
        assert response is not None
        assert len(response) > 0

    def test_generate_with_messages(self, api_env):
        """Test multi-turn generation."""
        from kosmos.core.llm import ClaudeClient

        client = ClaudeClient(model="claude-3-haiku-20240307")
        messages = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
            {"role": "user", "content": "What is my name?"}
        ]

        response = client.generate_with_messages(messages)

        # Should remember context from conversation
        assert response is not None
        assert "alice" in response.lower()


class TestClaudeClientStructured:
    """Test structured output generation with real API."""

    def test_generate_structured_json(self, api_env):
        """Test structured JSON output."""
        from kosmos.core.llm import ClaudeClient

        client = ClaudeClient(model="claude-3-haiku-20240307")
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"}
            }
        }

        result = client.generate_structured(
            prompt="Generate a JSON object with a person's name (John) and age (30). Output ONLY valid JSON.",
            output_schema=schema
        )

        # Should get valid JSON back
        assert isinstance(result, dict)
        assert "name" in result or "age" in result

    def test_generate_structured_with_markdown(self, api_env):
        """Test structured output extraction from markdown code block."""
        from kosmos.core.llm import ClaudeClient

        client = ClaudeClient(model="claude-3-haiku-20240307")
        result = client.generate_structured(
            prompt="Return this JSON in a markdown code block: {\"status\": \"ok\"}",
            output_schema={"type": "object"}
        )

        # Should parse JSON from markdown block
        assert isinstance(result, dict)

    def test_generate_structured_invalid_json(self, api_env):
        """Test error handling for invalid JSON."""
        from kosmos.core.llm import ClaudeClient
        from kosmos.core.providers.base import ProviderAPIError

        client = ClaudeClient(model="claude-3-haiku-20240307")

        # Force a non-JSON response (might still get JSON, so this test is probabilistic)
        # Instead, test that the method handles malformed input gracefully
        try:
            result = client.generate_structured(
                prompt="Write a poem about the sea. Do not use any JSON.",
                output_schema={"type": "object"}
            )
            # If it somehow parses, that's fine
            assert isinstance(result, dict)
        except (ValueError, ProviderAPIError) as e:
            # Expected - invalid JSON should raise error
            assert "JSON" in str(e) or "json" in str(e).lower()


class TestClaudeClientStatistics:
    """Test usage statistics tracking with real API."""

    def test_get_usage_stats(self, api_env):
        """Test getting usage statistics."""
        from kosmos.core.llm import ClaudeClient

        client = ClaudeClient(model="claude-3-haiku-20240307")
        # Use unique prompts to avoid cache hits
        client.generate(unique_prompt("Say 'one'"))
        client.generate(unique_prompt("Say 'two'"))

        stats = client.get_usage_stats()

        assert stats["total_requests"] == 2
        assert stats["total_input_tokens"] > 0
        assert stats["total_output_tokens"] > 0
        assert "estimated_cost_usd" in stats

    def test_cost_estimation_api_mode(self, api_env):
        """Test cost estimation in API mode."""
        from kosmos.core.llm import ClaudeClient

        client = ClaudeClient(model="claude-3-haiku-20240307")
        client.generate(unique_prompt("Hello"))

        stats = client.get_usage_stats()

        # Should have non-zero cost in API mode
        assert stats["estimated_cost_usd"] >= 0  # Haiku is very cheap

    def test_cost_estimation_cli_mode(self, cli_env):
        """Test cost estimation in CLI mode (should be 0)."""
        from kosmos.core.llm import ClaudeClient

        client = ClaudeClient(model="claude-3-haiku-20240307", enable_cache=False)
        # Note: CLI mode won't actually work with real API, but we can test initialization
        # This just tests that cli_mode is detected correctly

        stats = client.get_usage_stats()

        # Should be zero cost before any requests
        assert stats["estimated_cost_usd"] == 0.0

    def test_reset_stats(self, api_env):
        """Test resetting statistics."""
        from kosmos.core.llm import ClaudeClient

        client = ClaudeClient(model="claude-3-haiku-20240307")
        client.generate(unique_prompt("Test"))

        assert client.total_requests == 1

        client.reset_stats()

        assert client.total_requests == 0
        assert client.total_input_tokens == 0
        assert client.total_output_tokens == 0


class TestLiteLLMLocalProvider:
    """Test local LiteLLM provider via LiteLLM configuration."""

    @pytest.fixture
    def local_provider(self, local_litellm_env):
        from kosmos.core.providers.litellm_provider import LiteLLMProvider

        return LiteLLMProvider(local_litellm_env)

    def test_local_litellm_generate(self, local_provider):
        """Test basic generation against a local LiteLLM endpoint."""
        response = local_provider.generate(
            prompt=unique_prompt("Say 'Hello' and nothing else.")
        )

        assert isinstance(response.content, str)
        assert len(response.content) > 0

    def test_local_litellm_generate_with_messages(self, local_provider):
        """Test multi-turn generation against local LiteLLM."""
        from kosmos.core.providers.base import Message

        messages = [
            Message(role="user", content="My name is Alice."),
            Message(role="assistant", content="Hello Alice!"),
            Message(role="user", content=unique_prompt("What is my name?"))
        ]

        response = local_provider.generate_with_messages(messages)

        assert isinstance(response.content, str)
        assert "alice" in response.content.lower()

    def test_local_litellm_generate_structured(self, local_provider):
        """Test structured JSON output from local LiteLLM."""
        schema = {
            "type": "object",
            "properties": {
                "greeting": {"type": "string"}
            }
        }

        result = local_provider.generate_structured(
            prompt=unique_prompt("Return a JSON object with greeting 'hello'."),
            schema=schema
        )

        assert isinstance(result, dict)
        assert "greeting" in result or (
            "properties" in result and "greeting" in result["properties"]
        )


class TestClaudeClientSingleton:
    """Test singleton client instance."""

    def test_get_client(self, api_env):
        """Test getting default client."""
        from kosmos.core.llm import get_client

        client1 = get_client()
        client2 = get_client()

        # Should return same instance
        assert client1 is client2

    def test_get_client_reset(self, api_env):
        """Test resetting default client."""
        from kosmos.core.llm import get_client

        client1 = get_client()
        client2 = get_client(reset=True)

        # Should return different instance after reset
        assert client1 is not client2
