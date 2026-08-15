# Security Policy

## Reporting Security Vulnerabilities

The SimpliScribe team takes the security and privacy of healthcare data seriously. If you discover a security vulnerability or potential clinical safety issue, please report it responsibly.

### How to Report

Please **do not** report security vulnerabilities via public GitHub issues. Instead:

1. Send an email to the security team or maintainers describing the vulnerability in detail.
2. Include:
   - Type of vulnerability (e.g., CSRF bypass, session hijacking, SQL injection, PHI leakage).
   - Step-by-step reproduction instructions or proof-of-concept.
   - Potential impact and affected versions.
3. You will receive an acknowledgment within **48 hours**, followed by a timeline for triage and resolution.

---

## Security Architecture & Data Handling

SimpliScribe incorporates several baseline security mechanisms:

* **Session Security**: Signed, HTTP-only session cookies with configurable TTL (`SESSION_MAX_AGE_SECONDS`) and strict CSRF verification on mutating endpoints.
* **Content Security Policy (CSP)**: Restrictive default CSP headers preventing XSS and frame embedding (`X-Frame-Options: DENY`, `no-store` cache controls).
* **Data Retention & Expiry**: Automated purge of historical prescription payloads beyond configured retention limits (`RETENTION_DAYS`, defaults to 30 days).
* **Audit Logging**: Structured, immutable audit event logs for login, analysis creation, and clinical review actions.
* **Fail-Closed Alternatives**: Model and web lookups for drug alternatives remain off by default (`ALTERNATIVES_ENABLED=false`) until explicitly enabled in trusted deployment environments.

For a comprehensive threat analysis, consult [`docs/simpliscribe-threat-model.md`](docs/simpliscribe-threat-model.md).
