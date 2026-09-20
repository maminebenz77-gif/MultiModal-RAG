"""get_principal: the FastAPI dependency that turns a request into a
Principal (identity.py).

auth_mode="disabled" (Settings, the default) skips token validation
entirely and hands every caller Principal.unrestricted() -- not a
special "auth is off" code path bypassing the security machinery, but a
REAL Principal with maximal clearance and an admin bypass, that flows
through the exact same filter-construction code (retrieval/scoped.py)
every other principal does. That matters: it means the scoping code is
exercised on every single test run and every local dev session, not
just in a hypothetical "auth enabled" test suite that might silently
rot.

auth_mode="oidc" requires a valid bearer token, verified against a real
identity provider's public keys (JWKS) -- signature, issuer, audience,
expiry, all checked by PyJWT, not hand-rolled.
"""

from functools import lru_cache

import jwt
from fastapi import Header, HTTPException
from jwt import PyJWKClient

from ..config import Settings, get_settings
from ..identity import Principal

_SIGNING_ALGORITHMS = ["RS256"]
"""What every mainstream OIDC provider (Okta, Entra, Keycloak, Auth0)
signs with by default. Deliberately not "whatever the token's own header
claims" -- accepting an attacker-chosen algorithm (e.g. HS256, verified
against a key an attacker doesn't actually need to know if the verifier
trusts the token to say which algorithm to use) is a well-known JWT
verification bypass; the accepted algorithm list must come from US, not
from the token."""


@lru_cache(maxsize=8)
def _jwk_client(jwks_url: str) -> PyJWKClient:
    # Cached per URL (not per Settings instance) so the underlying
    # PyJWKClient -- which does its OWN key caching internally -- is
    # reused across requests instead of re-fetching the provider's JWKS
    # document on every single call.
    return PyJWKClient(jwks_url)


def _principal_from_claims(claims: dict) -> Principal:
    subject = claims.get("sub")
    if not subject:
        raise HTTPException(status_code=401, detail="Token has no subject (sub) claim")

    clearance = claims.get("clearance", "public")
    if clearance not in ("public", "c1", "c2", "c3"):
        # A token claiming a clearance level this system doesn't
        # recognize is treated as the SAFEST level, not rejected outright
        # -- an identity provider a step ahead of this deployment (a new
        # level it doesn't know about yet) should degrade to "public",
        # not lock every one of its users out entirely.
        clearance = "public"

    return Principal(
        principal_id=f"user:{subject}",
        clearance=clearance,
        is_admin=False,
    )


def get_principal(authorization: str | None = Header(default=None)) -> Principal:
    settings: Settings = get_settings()
    if settings.auth_mode == "disabled":
        return Principal.unrestricted()

    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ")

    if not settings.oidc_issuer or not settings.oidc_audience or not settings.oidc_jwks_url:
        # A misconfigured "oidc" mode (auth_mode flipped on without the
        # three settings it needs) must fail closed -- reject every
        # request -- not silently fall back to unrestricted access.
        raise HTTPException(status_code=500, detail="OIDC is not fully configured")

    try:
        signing_key = _jwk_client(settings.oidc_jwks_url).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=_SIGNING_ALGORITHMS,
            audience=settings.oidc_audience,
            issuer=settings.oidc_issuer,
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail=f"Invalid token: {exc}") from exc

    return _principal_from_claims(claims)
