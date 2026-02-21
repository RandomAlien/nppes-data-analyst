#!/usr/bin/env python3
"""Core runtime for deployed Streamlit NPPES analyst app."""

from __future__ import annotations

import hashlib
import json
import os
import re
import textwrap
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
from openai import OpenAI

FORBIDDEN_SQL_KEYWORDS = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "ATTACH",
    "DETACH",
    "COPY",
    "CALL",
    "PRAGMA",
)

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_HF_GROQ_MODEL = "openai/gpt-oss-20b:groq"


class ConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class AppConfig:
    db_path: str
    model: str
    hf_token: str
    base_url: str = HF_ROUTER_BASE_URL
    timeout_seconds: float = 60.0
    max_retries: int = 3
    duckdb_remote_url: str = ""
    duckdb_remote_sha256: str = ""


@dataclass
class QueryResult:
    sql: str
    answer: str
    columns: list[str]
    rows: list[tuple[Any, ...]]


def load_app_config(values: dict[str, str]) -> AppConfig:
    db_path = values.get("NPPES_DUCKDB_PATH", "db/nppes.duckdb").strip()
    model = values.get("LLM_MODEL", DEFAULT_HF_GROQ_MODEL).strip()
    hf_token = values.get("HF_TOKEN", "").strip()
    base_url = values.get("LLM_BASE_URL", HF_ROUTER_BASE_URL).strip() or HF_ROUTER_BASE_URL
    duckdb_remote_url = values.get("DUCKDB_REMOTE_URL", "").strip()
    duckdb_remote_sha256 = values.get("DUCKDB_REMOTE_SHA256", "").strip().lower()

    if not hf_token:
        raise ConfigurationError("HF_TOKEN is required.")
    if ":groq" not in model.lower():
        raise ConfigurationError("LLM_MODEL must include ':groq'.")

    timeout_seconds = float(values.get("LLM_TIMEOUT_SECONDS", "60"))
    max_retries = int(values.get("LLM_MAX_RETRIES", "3"))
    if timeout_seconds <= 0:
        raise ConfigurationError("LLM_TIMEOUT_SECONDS must be > 0.")
    if max_retries < 0:
        raise ConfigurationError("LLM_MAX_RETRIES must be >= 0.")

    return AppConfig(
        db_path=db_path,
        model=model,
        hf_token=hf_token,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        duckdb_remote_url=duckdb_remote_url,
        duckdb_remote_sha256=duckdb_remote_sha256,
    )


def resolve_db_file(config: AppConfig) -> str:
    db_file = Path(config.db_path)
    if db_file.exists():
        return str(db_file)

    if not config.duckdb_remote_url:
        raise FileNotFoundError(
            f"DuckDB not found at '{config.db_path}'. Set DUCKDB_REMOTE_URL for hosted deploy."
        )

    cache_dir = Path("/tmp/nppes_data_analyst")
    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded = cache_dir / "nppes.duckdb"

    if downloaded.exists() and config.duckdb_remote_sha256:
        existing = sha256_file(downloaded)
        if existing == config.duckdb_remote_sha256:
            return str(downloaded)

    urllib.request.urlretrieve(config.duckdb_remote_url, downloaded)  # noqa: S310

    if config.duckdb_remote_sha256:
        actual = sha256_file(downloaded)
        if actual != config.duckdb_remote_sha256:
            downloaded.unlink(missing_ok=True)
            raise ConfigurationError(
                "DUCKDB_REMOTE_SHA256 mismatch after download. Refusing to use database."
            )

    return str(downloaded)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = text.removesuffix("```").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("Model did not return valid JSON.")
        return json.loads(match.group(0))


def normalize_sql(sql: str) -> str:
    sql = sql.strip()
    if sql.startswith("```"):
        sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
        sql = sql.removesuffix("```").strip()
    return sql.rstrip(";").strip()


def validate_sql(sql: str) -> str:
    sql = normalize_sql(sql)
    upper_sql = sql.upper()

    if not sql:
        raise ValueError("SQL cannot be empty.")
    if ";" in sql:
        raise ValueError("Only one SQL statement is allowed.")
    if not re.match(r"^(SELECT|WITH)\b", upper_sql):
        raise ValueError("Only SELECT/WITH queries are allowed.")
    if not re.search(r"\bUS_HCP_PROFILE\b", upper_sql):
        raise ValueError("Query must use US_HCP_PROFILE.")
    for keyword in FORBIDDEN_SQL_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper_sql):
            raise ValueError(f"Forbidden SQL keyword detected: {keyword}")
    return sql


def add_default_limit(sql: str, default_limit: int = 200) -> str:
    if re.search(r"\bLIMIT\s+\d+\b", sql, flags=re.IGNORECASE):
        return sql
    return f"{sql}\nLIMIT {default_limit}"


class HuggingFaceGroqLLMClient:
    def __init__(
        self,
        model: str,
        hf_token: str,
        base_url: str,
        timeout_seconds: float,
        max_retries: int,
    ) -> None:
        self.model = model
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=hf_token,
            base_url=base_url,
            timeout=timeout_seconds,
        )

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.1) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    messages=messages,
                )
                return (response.choices[0].message.content or "").strip()
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"LLM request failed after retries: {last_error}") from last_error


class NPPESDataAnalystEngine:
    def __init__(self, db_path: str, llm: HuggingFaceGroqLLMClient) -> None:
        self.llm = llm
        self.conn = duckdb.connect(database=db_path, read_only=True)

    def schema_hint(self) -> str:
        rows = self.conn.execute("DESCRIBE SELECT * FROM US_HCP_PROFILE").fetchall()
        cols = [row[0] for row in rows]
        return ", ".join(cols)

    def propose_sql(self, user_question: str, history: list[dict[str, str]]) -> str:
        schema = self.schema_hint()
        recent_context = "\n".join(
            f"{item['role']}: {item['content']}" for item in history[-8:]
        )
        system_prompt = textwrap.dedent(
            """
            You are an expert SQL generator for DuckDB.
            Return ONLY JSON in the form: {"sql":"..."}.
            Rules:
            - Output exactly one SQL statement.
            - Query only US_HCP_PROFILE.
            - Use only SELECT/WITH.
            """
        ).strip()
        user_prompt = textwrap.dedent(
            f"""
            Available view: US_HCP_PROFILE
            Columns: {schema}
            Conversation context:
            {recent_context or 'No prior context.'}
            User question:
            {user_question}
            """
        ).strip()

        content = self.llm.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
        payload = extract_json_object(content)
        if "sql" not in payload:
            raise ValueError("Model response missing 'sql' field.")
        return validate_sql(payload["sql"])

    def run_query(self, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
        safe_sql = add_default_limit(sql)
        cursor = self.conn.execute(safe_sql)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        return columns, rows

    def compose_answer(
        self,
        user_question: str,
        sql: str,
        columns: list[str],
        rows: list[tuple[Any, ...]],
    ) -> str:
        preview = [dict(zip(columns, row)) for row in rows[:25]]
        return self.llm.chat(
            [
                {
                    "role": "system",
                    "content": "You are a US healthcare provider data analyst. Answer clearly and include key numbers from the query result.",
                },
                {
                    "role": "user",
                    "content": (
                        f"User question: {user_question}\n"
                        f"SQL used:\n{sql}\n\n"
                        f"Rows returned: {len(rows)}\n"
                        f"Data preview:\n{json.dumps(preview, default=str)}"
                    ),
                },
            ],
            temperature=0.2,
        )

    def ask(self, user_question: str, history: list[dict[str, str]]) -> QueryResult:
        sql = self.propose_sql(user_question, history)
        columns, rows = self.run_query(sql)
        answer = self.compose_answer(user_question, sql, columns, rows)
        return QueryResult(sql=sql, answer=answer, columns=columns, rows=rows)

