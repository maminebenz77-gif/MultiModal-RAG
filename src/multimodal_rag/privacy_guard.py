"""Shared privacy guard: when a profile sets allow_external=False (the
air-gapped server), it must be structurally impossible to construct
anything — a provider, a vector store client — that talks to a non-local
host. Enforced here, not trusted to config alone.

Used by providers/factory.py and stores/factory.py, which are the ONLY
places allowed to construct concrete provider/store classes — that
single rule is what makes this guard actually effective. If code could
reach around a factory and construct a client directly, the guard would
never run.
"""

import ipaddress
import os
import socket
from urllib.parse import urlparse


class ExternalCallBlockedError(RuntimeError):
    """Raised when the active profile forbids external calls but
    constructing a provider/store would require one."""


def is_internal_host(host: str) -> bool:
    """True if `host` is loopback or a private-network address.

    Handles both literal IPs and hostnames (resolved via DNS, which is
    exactly what should succeed for an internal company hostname on the
    server's own network, and fail closed otherwise).
    """
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        pass
    try:
        resolved = socket.gethostbyname(host)
    except socket.gaierror:
        return False
    return ipaddress.ip_address(resolved).is_private


def enforce_privacy_guard(base_url: str | None, allow_external: bool) -> None:
    if allow_external:
        return

    # Belt-and-suspenders: even if a provider correctly checks base_url,
    # also force offline mode so a library (e.g. huggingface_hub) can't
    # silently try to download an uncached model from the internet.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if base_url is None:
        return  # no network endpoint at all -> nothing to check

    host = urlparse(base_url).hostname
    if host is None or not is_internal_host(host):
        raise ExternalCallBlockedError(
            f"Refusing to construct a client pointing at {base_url!r}: "
            f"allow_external=False and {host!r} is not a local/internal host."
        )
