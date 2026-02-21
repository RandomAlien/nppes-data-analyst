# Security Policy

## Supported Deployment

This project is intended for public read-only analytics use through Streamlit.

## Reporting a Vulnerability

If you discover a security issue, do not open a public issue with exploit details.
Report privately to the repository owner with:

- summary of the issue
- impact and affected components
- reproduction steps
- suggested mitigation (if available)

## Secret Management

- Store `HF_TOKEN` only in Streamlit secrets.
- Never commit `.env` or `.streamlit/secrets.toml`.
- Rotate tokens immediately if exposed.

## Data and Runtime Hardening

- Keep DuckDB read-only in deployed environments.
- Use `DUCKDB_REMOTE_SHA256` to verify remote DB integrity.
- Limit DB source to trusted storage URLs.
