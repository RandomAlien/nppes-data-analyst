# NPPES Data Analyst (Public Streamlit App)

Production Streamlit app for querying NPPES provider data using DuckDB and Hugging Face Router (Groq-routed model).

## Features

- Natural language to SQL over `US_HCP_PROFILE`
- Strict SQL safety guardrails
- Read-only DuckDB runtime for safe public deployment
- Remote DuckDB loading from external storage
- CSV download of query results

## Architecture

- `streamlit_app.py`: Streamlit user interface
- `core.py`: LLM client, SQL validation, DB loader, and analytics engine
- `requirements.txt`: runtime dependencies

## Runtime Secrets

Set these in Streamlit Community Cloud Secrets:

- `HF_TOKEN` (required)
- `LLM_MODEL` (default: `openai/gpt-oss-20b:groq`)
- `LLM_BASE_URL` (default: `https://router.huggingface.co/v1`)
- `DUCKDB_REMOTE_URL` (recommended in hosted mode)
- `DUCKDB_REMOTE_SHA256` (recommended integrity check)
- `LLM_TIMEOUT_SECONDS` (optional)
- `LLM_MAX_RETRIES` (optional)

Example:

```toml
HF_TOKEN = "hf_xxx"
LLM_MODEL = "openai/gpt-oss-20b:groq"
LLM_BASE_URL = "https://router.huggingface.co/v1"
DUCKDB_REMOTE_URL = "https://huggingface.co/datasets/AJITTIWARI97/NPPES_AGENT/resolve/main/nppes.duckdb"
DUCKDB_REMOTE_SHA256 = "<optional_sha256>"
LLM_TIMEOUT_SECONDS = "60"
LLM_MAX_RETRIES = "3"
```

## Deploy

1. Push only this folder's contents to a dedicated GitHub repo.
2. Create a Streamlit app from that repo.
3. Set entrypoint file to `streamlit_app.py`.
4. Add secrets.
5. Deploy and test with a simple query.

## DuckDB Storage

Recommended free options:

1. Hugging Face Hub dataset file URL (`resolve/main/...duckdb`)
2. Cloudflare R2 public object URL
3. GitHub Release asset URL

## Security

- Never commit `.env`, `.duckdb`, or secrets.
- Keep the app read-only in public deployment.
- Rotate tokens immediately if exposed.

## Limitations

- Initial load may be slower due to remote DB download.
- Hosted filesystem is ephemeral.

## License

MIT (see `LICENSE`).
