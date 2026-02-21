#!/usr/bin/env python3
"""Deploy-only Streamlit app for NPPES Data Analyst."""

from __future__ import annotations

import csv
import io
import os

import streamlit as st

from core import (
    ConfigurationError,
    DEFAULT_HF_GROQ_MODEL,
    HF_ROUTER_BASE_URL,
    HuggingFaceGroqLLMClient,
    NPPESDataAnalystEngine,
    QueryResult,
    load_app_config,
    resolve_db_file,
)


def settings_from_env_and_secrets() -> dict[str, str]:
    values = dict(os.environ)
    for key in (
        "NPPES_DUCKDB_PATH",
        "DUCKDB_REMOTE_URL",
        "DUCKDB_REMOTE_SHA256",
        "LLM_MODEL",
        "LLM_BASE_URL",
        "HF_TOKEN",
        "LLM_TIMEOUT_SECONDS",
        "LLM_MAX_RETRIES",
    ):
        if key in st.secrets:
            values[key] = str(st.secrets[key])
    return values


def rows_to_csv(columns: list[str], rows: list[tuple]) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(columns)
    writer.writerows(rows)
    return output.getvalue()


@st.cache_resource(show_spinner="Initializing engine...")
def build_engine(config_dict: dict[str, str]) -> NPPESDataAnalystEngine:
    config = load_app_config(config_dict)
    db_path = resolve_db_file(config)
    llm = HuggingFaceGroqLLMClient(
        model=config.model,
        hf_token=config.hf_token,
        base_url=config.base_url,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
    )
    return NPPESDataAnalystEngine(db_path=db_path, llm=llm)


st.set_page_config(page_title="NPPES Data Analyst", page_icon="N", layout="wide")
st.title("NPPES Data Analyst")
st.caption("Public Streamlit deployment (HF Router + Groq).")

settings = settings_from_env_and_secrets()
with st.sidebar:
    st.subheader("Runtime Config")
    st.text_input("Model", value=settings.get("LLM_MODEL", DEFAULT_HF_GROQ_MODEL), disabled=True)
    st.text_input("LLM Base URL", value=settings.get("LLM_BASE_URL", HF_ROUTER_BASE_URL), disabled=True)
    st.text_input("Database Path", value=settings.get("NPPES_DUCKDB_PATH", "db/nppes.duckdb"), disabled=True)
    if settings.get("DUCKDB_REMOTE_URL"):
        st.caption("Remote DB download enabled")
    else:
        st.caption("Using local DB path")

try:
    engine = build_engine(settings)
except (ConfigurationError, FileNotFoundError) as exc:
    st.error(f"Configuration error: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Engine initialization failed: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sql" in msg:
            with st.expander("SQL used"):
                st.code(msg["sql"], language="sql")
        if "table" in msg:
            st.dataframe(msg["table"], use_container_width=True)

prompt = st.chat_input("Ask a question about US_HCP_PROFILE...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        result: QueryResult = engine.ask(prompt, st.session_state.history)
        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.history.append({"role": "assistant", "content": result.answer})

        table_preview = [dict(zip(result.columns, row)) for row in result.rows[:100]]
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.answer,
                "sql": result.sql,
                "table": table_preview,
            }
        )

        with st.chat_message("assistant"):
            st.markdown(result.answer)
            with st.expander("SQL used"):
                st.code(result.sql, language="sql")
            st.dataframe(table_preview, use_container_width=True)
            st.download_button(
                label="Download query result CSV",
                data=rows_to_csv(result.columns, result.rows),
                file_name="query_result.csv",
                mime="text/csv",
            )
    except Exception as exc:
        err = f"I hit an error: {exc}"
        st.session_state.messages.append({"role": "assistant", "content": err})
        with st.chat_message("assistant"):
            st.error(err)

