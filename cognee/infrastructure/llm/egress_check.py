"""Check the URL an LLM or embedding provider will contact against the egress policy.

Called where a client is constructed, so a provider the policy forbids fails there with the URL
named, not at its first request. Without an active policy every call is a no-op.
"""

from cognee_egress import EgressDenied, active_policy, check_url

# Base URL each provider's SDK (litellm) uses when no endpoint is configured.
DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
    "mistral": "https://api.mistral.ai",
    "ollama": "http://localhost:11434",
}

# Providers that run on this machine and contact no remote host.
_LOCAL_PROVIDERS = {"fastembed", "llama_cpp", "mcp-sampling"}


def check_provider_endpoint(provider: str, endpoint: str | None, role: str) -> None:
    """Check the URL a provider will really contact against the egress policy (no-op without one).

    ``endpoint`` is the configured endpoint; when it is empty the provider's default base URL is
    checked. Raises ``EgressDenied`` naming the URL when the policy refuses it.
    """
    if active_policy() is None:
        return
    if provider == "bedrock":
        check_url(_bedrock_url(), role=role)
        return
    if endpoint:
        check_url(endpoint, role=role)
        return
    if provider in _LOCAL_PROVIDERS or provider == "azure":
        # Azure's endpoint is required by its client, which raises when it is missing.
        return
    if provider not in DEFAULT_BASE_URLS:
        raise EgressDenied(
            f"cannot determine the endpoint of provider {provider!r}; set the endpoint explicitly"
        )
    check_url(DEFAULT_BASE_URLS[provider], role=role)


def _bedrock_url() -> str:
    """The Bedrock runtime URL the client uses: its endpoint override, else the regional one."""
    from cognee.infrastructure.files.storage.s3_config import get_s3_config

    s3_config = get_s3_config()
    if s3_config.aws_bedrock_runtime_endpoint:
        return s3_config.aws_bedrock_runtime_endpoint
    if not s3_config.aws_region:
        raise EgressDenied("cannot determine the Bedrock endpoint: the AWS region is unknown")
    return f"https://bedrock-runtime.{s3_config.aws_region}.amazonaws.com"
