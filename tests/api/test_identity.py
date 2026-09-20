"""api.identity: Principal + get_principal.

Real RSA signing/verification (PyJWT + cryptography) so token validation
itself is genuinely exercised -- but the JWKS *fetch* is stubbed (a fake
_jwk_client returning a known key directly). That's PyJWT's own library
code, already covered by its own test suite; what's under test here is
THIS module's logic: claim mapping, the clearance fallback, and which
failures become which HTTP errors.
"""

from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException

from multimodal_rag.api import identity as identity_module
from multimodal_rag.api.identity import get_principal
from multimodal_rag.identity import Principal

_ISSUER = "https://issuer.example.com"
_AUDIENCE = "test-audience"


@pytest.fixture(scope="module")
def keypair():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


class _FakeSigningKey:
    def __init__(self, key) -> None:
        self.key = key


class _FakeJwkClient:
    """Stands in for PyJWKClient -- returns a known key directly instead
    of fetching a real JWKS document over HTTP."""

    def __init__(self, key) -> None:
        self._key = key

    def get_signing_key_from_jwt(self, token: str) -> _FakeSigningKey:
        return _FakeSigningKey(self._key)


@pytest.fixture(autouse=True)
def _oidc_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        identity_module,
        "get_settings",
        lambda: SimpleNamespace(
            auth_mode="oidc",
            oidc_issuer=_ISSUER,
            oidc_audience=_AUDIENCE,
            oidc_jwks_url="https://issuer.example.com/.well-known/jwks.json",
        ),
    )


def _token(private_key, **claim_overrides) -> str:
    claims = {"sub": "alice@example.com", "iss": _ISSUER, "aud": _AUDIENCE, **claim_overrides}
    return jwt.encode(claims, private_key, algorithm="RS256")


def test_disabled_auth_mode_returns_an_unrestricted_principal_without_a_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        identity_module, "get_settings", lambda: SimpleNamespace(auth_mode="disabled")
    )
    principal = get_principal(authorization=None)
    assert principal == Principal.unrestricted()
    assert principal.is_admin is True


def test_oidc_mode_rejects_a_missing_token(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(HTTPException) as exc_info:
        get_principal(authorization=None)
    assert exc_info.value.status_code == 401


def test_oidc_mode_rejects_a_non_bearer_authorization_header() -> None:
    with pytest.raises(HTTPException) as exc_info:
        get_principal(authorization="Basic dXNlcjpwYXNz")
    assert exc_info.value.status_code == 401


def test_oidc_mode_accepts_a_validly_signed_token(
    keypair, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_key, public_key = keypair
    monkeypatch.setattr(identity_module, "_jwk_client", lambda url: _FakeJwkClient(public_key))

    token = _token(private_key, clearance="c2")
    principal = get_principal(authorization=f"Bearer {token}")

    assert principal == Principal(principal_id="user:alice@example.com", clearance="c2")


def test_oidc_mode_rejects_a_token_signed_by_the_wrong_key(
    keypair, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_key, _ = keypair
    # A DIFFERENT keypair -- the verifier below checks the token's
    # signature against this one, which never signed it.
    other_public_key = rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key()
    monkeypatch.setattr(
        identity_module, "_jwk_client", lambda url: _FakeJwkClient(other_public_key)
    )

    token = _token(private_key)
    with pytest.raises(HTTPException) as exc_info:
        get_principal(authorization=f"Bearer {token}")
    assert exc_info.value.status_code == 401


def test_oidc_mode_rejects_an_expired_token(keypair, monkeypatch: pytest.MonkeyPatch) -> None:
    private_key, public_key = keypair
    monkeypatch.setattr(identity_module, "_jwk_client", lambda url: _FakeJwkClient(public_key))

    token = _token(private_key, exp=1)  # 1970 -- long expired
    with pytest.raises(HTTPException) as exc_info:
        get_principal(authorization=f"Bearer {token}")
    assert exc_info.value.status_code == 401


def test_oidc_mode_rejects_a_token_for_the_wrong_audience(
    keypair, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_key, public_key = keypair
    monkeypatch.setattr(identity_module, "_jwk_client", lambda url: _FakeJwkClient(public_key))

    token = _token(private_key, aud="someone-elses-app")
    with pytest.raises(HTTPException) as exc_info:
        get_principal(authorization=f"Bearer {token}")
    assert exc_info.value.status_code == 401


def test_unrecognized_clearance_claim_degrades_to_public_not_a_rejection(
    keypair, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An identity provider a step ahead of this deployment (a clearance
    level it doesn't know about yet) shouldn't lock every one of its
    users out -- degrade to the safest level instead."""
    private_key, public_key = keypair
    monkeypatch.setattr(identity_module, "_jwk_client", lambda url: _FakeJwkClient(public_key))

    token = _token(private_key, clearance="top-secret")
    principal = get_principal(authorization=f"Bearer {token}")

    assert principal.clearance == "public"


def test_missing_clearance_claim_defaults_to_public(
    keypair, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_key, public_key = keypair
    monkeypatch.setattr(identity_module, "_jwk_client", lambda url: _FakeJwkClient(public_key))

    token = _token(private_key)  # no "clearance" claim at all
    principal = get_principal(authorization=f"Bearer {token}")

    assert principal.clearance == "public"


def test_oidc_mode_with_missing_settings_fails_closed(
    keypair, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_key, _ = keypair
    monkeypatch.setattr(
        identity_module,
        "get_settings",
        lambda: SimpleNamespace(
            auth_mode="oidc", oidc_issuer=None, oidc_audience=None, oidc_jwks_url=None
        ),
    )

    token = _token(private_key)
    with pytest.raises(HTTPException) as exc_info:
        get_principal(authorization=f"Bearer {token}")
    assert exc_info.value.status_code == 500
