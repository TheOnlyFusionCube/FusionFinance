# Security policy

## Scope

Security reports are especially important for input validation, path
handling, model/tool boundaries, evidence integrity, secret handling,
dependency loading, and any code that could influence a portfolio decision.
FusionFinance does not authorize real-money brokerage execution.

## Reporting a vulnerability

Do not disclose exploitable details, credentials, private data, or tokens in
a public issue. Use the repository's
[private GitHub security-advisory channel](https://github.com/TheOnlyFusionCube/FusionFinance/security/advisories/new)
when available. If private reporting is unavailable, open a public issue that
only requests a private maintainer contact and omits all vulnerability
details. For non-sensitive hardening suggestions, a normal issue may include
a minimal reproduction and the affected commit.

Include the affected path, impact, reproduction steps, and any safe
mitigation you have tested. Please allow maintainers time to validate and fix
the issue before public disclosure.

## Supported version

The latest commit on `main` is the supported research version. Historical
replays and generated submission artifacts are retained for auditability but
do not receive independent security updates.

## Secrets and external services

The default demo is offline and needs no credentials. Optional data and model
providers must read secrets from environment variables, enforce timeouts,
validate responses, and degrade to a deterministic non-trading fallback.
