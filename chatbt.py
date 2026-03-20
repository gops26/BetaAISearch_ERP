import os
import re
import hashlib
import uuid
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any

import sqlalchemy as sa
from sqlalchemy import inspect, text as sa_text
from sqlalchemy.engine import Engine

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq

from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

# ============================
# LOGGING
# ============================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger("new_rag_assistant")


# ============================
# CONFIG
# ============================

QDRANT_COLLECTION = "RESUME_PARSER"
HF_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
GROQ_MODEL = "llama-3.3-70b-versatile"

RAG_TOP_K = 8
MAX_SQL_LIMIT = 500
MAX_QUESTION_LENGTH = 2000
RAG_MAX_CONTEXT_CHARS = 4000

SQL_CACHE_TTL_SECONDS = 300
SESSION_TTL_MINUTES = 30
MAX_HISTORY_TURNS = 20

# Multi-schema support: comma-separated list of schema names
KNOWN_SCHEMAS: List[str] = [
    s.strip() for s in os.getenv("DB_SCHEMAS", "core,recruit").split(",") if s.strip()
]

CONNECTION_STRING = (
    f"mssql+pyodbc://{os.getenv('DB_USER', 'sa')}:{os.getenv('DB_PASSWORD', 'Strong!1234')}"
    f"@{os.getenv('DB_HOST', '65.38.99.253,1433')}/{os.getenv('DB_NAME', 'Jobpost_DEV')}?"
    f"driver={os.getenv('DB_DRIVER', 'ODBC+Driver+17+for+SQL+Server').replace(' ', '+')}"
)


# ============================
# GLOBALS
# ============================

engine: Engine = sa.create_engine(CONNECTION_STRING, pool_pre_ping=True)
TABLE_SCHEMAS: Dict[str, Dict] = {}   # key: "schema.table"
vector_db: Optional[QdrantVectorStore] = None
retriever: Optional[Any] = None
llm_intent: Optional[ChatGroq] = None   # temperature=0.0  — intent classification
llm_sql: Optional[ChatGroq] = None      # temperature=0.1  — SQL generation
llm_rag: Optional[ChatGroq] = None      # temperature=0.4  — RAG synthesis
_app_start_time: float = time.time()
_sessions: Dict[str, Dict] = {}

_SQL_CACHE: Dict[str, Dict] = {}   # structured path cache
_RAG_CACHE: Dict[str, Dict] = {}   # unstructured path cache


# ============================
# PYDANTIC MODELS
# ============================

class IntentLabel(str, Enum):
    STRUCTURED = "structured"
    UNSTRUCTURED = "unstructured"


class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    limit: int = 100
    offset: int = 0

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How many candidates are in the system?",
                "session_id": "abc-123",
                "limit": 50,
                "offset": 0
            }
        }


class ColumnMeta(BaseModel):
    name: str
    type: str


class SQLResult(BaseModel):
    sql: str
    columns: List[ColumnMeta]
    rows: List[List[Any]]
    total_rows: int
    returned_rows: int
    limit: int
    offset: int


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    answer: str
    path: IntentLabel
    sql_result: Optional[SQLResult] = None
    sources: Optional[List[str]] = None
    timestamp: str
    execution_time_ms: int
    intent_confidence: str


class HealthResponse(BaseModel):
    status: str
    db_connected: bool
    qdrant_connected: bool
    llm_ready: bool
    schema_tables_loaded: int
    schemas_loaded: List[str]
    uptime_seconds: float
    timestamp: str


class SchemaTable(BaseModel):
    schema_name: str
    table: str
    qualified_name: str
    columns: List[str]
    primary_key: Optional[str] = None
    row_count: int


class SchemaResponse(BaseModel):
    tables: List[SchemaTable]
    total_tables: int
    schemas: List[str]
    timestamp: str


class ErrorResponse(BaseModel):
    request_id: str
    error: str
    detail: Optional[str] = None
    timestamp: str


# ============================
# SESSION MANAGEMENT
# ============================

def get_or_create_session(session_id: Optional[str]) -> tuple:
    now = datetime.utcnow()

    if session_id and session_id in _sessions:
        sess = _sessions[session_id]
        if now - sess["last_active"] < timedelta(minutes=SESSION_TTL_MINUTES):
            sess["last_active"] = now
            return session_id, sess["history"]
        else:
            del _sessions[session_id]

    new_id = str(uuid.uuid4())
    _sessions[new_id] = {
        "history": [],
        "last_active": now,
        "created_at": now
    }
    return new_id, _sessions[new_id]["history"]


def append_to_history(session_id: str, role: str, content: str):
    if session_id in _sessions:
        _sessions[session_id]["history"].append({"role": role, "content": content})
        if len(_sessions[session_id]["history"]) > MAX_HISTORY_TURNS * 2:
            _sessions[session_id]["history"] = _sessions[session_id]["history"][-(MAX_HISTORY_TURNS * 2):]


def cleanup_expired_sessions():
    now = datetime.utcnow()
    expired = [
        sid for sid, s in _sessions.items()
        if now - s["last_active"] >= timedelta(minutes=SESSION_TTL_MINUTES)
    ]
    for sid in expired:
        del _sessions[sid]


def format_history(history: list, max_turns: int = 6) -> str:
    if not history:
        return "(No prior conversation)"
    lines = []
    for turn in history[-max_turns:]:
        role = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{role}: {turn['content']}")
    return "\n".join(lines)


def _make_cache_key(parts: list) -> str:
    normalized = "|".join(" ".join(str(p).lower().split()) for p in parts)
    return hashlib.sha256(normalized.encode()).hexdigest()


def _cache_get(cache: dict, key: str):
    entry = cache.get(key)
    if entry:
        if time.time() < entry["expires_at"]:
            return entry["value"]
        del cache[key]
    return None


def _cache_set(cache: dict, key: str, value, ttl: int) -> None:
    cache[key] = {"value": value, "expires_at": time.time() + ttl}


# ============================
# SQL SAFETY VALIDATION
# ============================

_FORBIDDEN_PATTERN = re.compile(
    r'\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|EXEC|EXECUTE|'
    r'GRANT|REVOKE|DENY|MERGE|BULK|OPENROWSET|OPENQUERY|'
    r'xp_cmdshell|sp_executesql|DBCC)\b',
    re.IGNORECASE
    
)

_SELECT_START = re.compile(r'^\s*(WITH\s+\w|\bSELECT\b)', re.IGNORECASE)
_COMMENT_STRIP = re.compile(r'(--[^\n]*|/\*.*?\*/)', re.DOTALL)


def validate_sql(sql: str) -> tuple:
    """Returns (is_safe: bool, reason: str). Always call before executing."""
    clean = _COMMENT_STRIP.sub('', sql).strip()

    if not _SELECT_START.match(clean):
        return False, "Only SELECT queries are permitted"

    match = _FORBIDDEN_PATTERN.search(clean)
    if match:
        return False, f"Forbidden keyword detected: {match.group()}"

    core = clean.rstrip(';')
    if ';' in core:
        return False, "Multiple statements are not allowed"

    return True, "ok"


_SELECT_RE = re.compile(r'(?i)^\s*(SELECT)\s+(?!TOP\s+\d)', re.IGNORECASE)


def _inject_top(sql: str, n: int) -> str:
    """Rewrite SELECT -> SELECT TOP N if TOP is not already present."""
    return _SELECT_RE.sub(f'SELECT TOP {n} ', sql.strip(), count=1)


# ============================
# DB EXECUTION (READ-ONLY)
# ============================

def execute_sql_safely(sql: str, limit: int = 100) -> tuple:
    """
    Execute a SELECT query safely.
    Transaction is ALWAYS rolled back -- no data can ever be modified.
    Returns (rows: List[List], columns: List[ColumnMeta])
    """
    with engine.connect() as conn:
        with conn.begin() as txn:
            try:
                result = conn.execute(sa_text(sql))
                col_names = list(result.keys())
                if result.cursor and result.cursor.description:
                    col_types = [str(result.cursor.description[i][1]) for i in range(len(col_names))]
                else:
                    col_types = ["unknown"] * len(col_names)

                columns = [
                    ColumnMeta(name=col_names[i], type=col_types[i])
                    for i in range(len(col_names))
                ]
                rows = [list(r) for r in result.fetchmany(limit)]
                return rows, columns
            finally:
                txn.rollback()


# ============================
# QDRANT
# ============================

def connect_qdrant() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL", "https://26f16575-3b44-4a29-a6cd-0c30548fdd57.us-west-1-0.aws.cloud.qdrant.io"),
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=60.0
    )


def load_vector_store() -> QdrantVectorStore:
    embeddings = HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)
    client = connect_qdrant()
    return QdrantVectorStore(client=client, collection_name=QDRANT_COLLECTION, embedding=embeddings)


# ============================
# MULTI-SCHEMA CATCHER
# ============================

def powerful_schema_catcher_multi(eng: Engine, schemas: List[str]) -> Dict[str, Dict]:
    """
    Iterate all provided schemas and build a unified TABLE_SCHEMAS dict.
    Key: "schema.table"; each entry includes schema, qualified [schema].[table],
    columns, pk, row count, and sample rows.
    """
    schema_meta: Dict[str, Dict] = {}

    with eng.connect() as conn:
        inspector = inspect(conn)

        for schema_name in schemas:
            try:
                tables = inspector.get_table_names(schema=schema_name)
            except Exception as e:
                logger.warning(f"Could not list tables for schema '{schema_name}': {e}")
                continue

            for table in tables:
                key = f"{schema_name}.{table}"
                qualified = f"[{schema_name}].[{table}]"
                try:
                    columns = [c["name"] for c in inspector.get_columns(table, schema=schema_name)]

                    try:
                        pk_info = inspector.get_pk_constraint(table, schema=schema_name)
                        pk = pk_info.get("constrained_columns", [None])[0]
                    except Exception:
                        pk = None

                    try:
                        count = conn.execute(
                            sa_text(f"SELECT COUNT(*) FROM {qualified}")
                        ).fetchone()[0]
                    except Exception:
                        count = 0

                    # Collect safe (non-binary) columns for sample rows
                    safe_cols = []
                    for col in inspector.get_columns(table, schema=schema_name):
                        dtype = str(col["type"]).lower()
                        if any(x in dtype for x in ["binary", "varbinary", "image", "xml", "geography", "hierarchyid"]):
                            continue
                        safe_cols.append(f"[{col['name']}]")

                    sample_rows = []
                    if safe_cols:
                        try:
                            q = f"SELECT TOP 3 {', '.join(safe_cols)} FROM {qualified}"
                            rows = conn.execute(sa_text(q)).fetchall()
                            sample_rows = [str(dict(r._mapping)) for r in rows]
                        except Exception:
                            pass

                    schema_meta[key] = {
                        "schema": schema_name,
                        "table": table,
                        "qualified": qualified,
                        "columns": columns,
                        "pk": pk,
                        "count": count,
                        "sample_rows": sample_rows
                    }

                except Exception:
                    continue

    return schema_meta


# ============================
# SCHEMA CONTEXT BUILDER
# ============================

def build_schema_context(table_keys: List[str]) -> str:
    """Build a schema context string using qualified [schema].[table] names."""
    if not table_keys:
        table_keys = list(TABLE_SCHEMAS.keys())
    lines = []
    for key in table_keys:
        meta = TABLE_SCHEMAS.get(key, {})
        qualified = meta.get("qualified", f"[{key}]")
        cols = ", ".join(meta.get("columns", []))
        pk = meta.get("pk", "unknown")
        lines.append(f"TABLE {qualified} (PK: {pk})\n  COLUMNS: {cols}")
    return "\n\n".join(lines)


# ============================
# FORMAT RESULT ANSWER
# ============================
def synthesize_sql_answer(question: str, columns: list, rows: list, total_rows: int, limit: int) -> str:
    if not rows:
        return "No records found matching your query."

    col_names = [c.name for c in columns]
    sample_rows = rows[:10]
    rows_str = "\n".join(str(dict(zip(col_names, r))) for r in sample_rows)

    prompt = SQL_ANSWER_PROMPT.format(
        question=question,
        columns=", ".join(col_names),
        rows=rows_str,
        total_rows=total_rows,
        limit=limit
    )

    try:
        return llm_rag.invoke(prompt).content.strip()
    except Exception as e:
        logger.warning(f"SQL answer synthesis failed: {e} — falling back to basic format")
        col = columns[0].name.replace("_", " ").lower() if columns else "result"
        value = rows[0][0] if rows else "N/A"
        return f"The {col} is {value:,}." if isinstance(value, (int, float)) else f"The {col} is {value}."
# ============================
# PROMPT TEMPLATES
# ============================

INTENT_DETECTION_PROMPT = """Classify the user's question as one of two types:
- "structured": needs data from a SQL database (counts, lists, filters, aggregations)
- "unstructured": needs explanations, definitions, summaries, or general knowledge

CONVERSATION HISTORY:
{history}

USER QUESTION:
{question}

Reply with EXACTLY one word: structured OR unstructured"""


SQL_GENERATION_PROMPT = """You are an expert T-SQL engineer. Write ONE valid T-SQL SELECT query.

DATABASE SCHEMA:
{schema_context}

CONVERSATION HISTORY:
{history}

USER QUESTION:
{question}

RULES:
1. SELECT only -- never INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/EXEC
2. Use TOP N instead of LIMIT
3. All identifiers in square brackets: [schema].[table], [column]
4. Single statement -- no semicolons inside query body
5. Return raw SQL only -- no JSON, no markdown
6. If question cannot be answered with the schema: return exactly CANNOT_GENERATE

T-SQL Query:"""


RAG_SYNTHESIS_PROMPT = """You are an enterprise HR assistant. Answer using ONLY the context below.
Do not invent table/column names not in the context.

RETRIEVED CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION:
{question}

Instructions: Be concise. Use bullet points for lists. Cite table names.
If context is insufficient: "I don't have enough context to answer that."

Answer:"""

SQL_ANSWER_PROMPT = """You are a helpful HR assistant. A user asked a question and the database returned results.
Convert the raw data into a clear, natural English sentence or short paragraph.

USER QUESTION:
{question}

SQL COLUMNS:
{columns}

DATA ROWS (up to 10 shown):
{rows}

TOTAL ROWS FOUND: {total_rows}

Rules:
- Write in plain English, not SQL-speak
- Never mention column names verbatim — translate them (e.g. "candidate_count" → "number of candidates")
- If it's a single number, give a direct sentence like "There are 42 candidates in the system."
- If it's a list, summarize cleanly
- If rows are capped, mention "showing top {limit} results"
- Be concise

Answer:"""
# ============================
# INTENT DETECTION
# ============================

_STRUCTURED_SIGNALS = re.compile(
    r'\b(how many|count|total|list all|show all|find all|top \d+|between|where|filter|'
    r'group by|order by|rank|which \w+|how much|percentage|fetch|retrieve)\b',
    re.IGNORECASE
)

_UNSTRUCTURED_SIGNALS = re.compile(
    r'\b(what is|what does|explain|describe|tell me about|summarize|definition of|'
    r'meaning of|who is|overview of|help me understand)\b|how does .* work',
    re.IGNORECASE
)


def _rule_based_intent(question: str) -> tuple:
    """
    Returns (Optional[IntentLabel], confidence_tag: str).
    Returns (None, "rule_uncertain") when signals conflict or are absent.
    """
    has_structured = bool(_STRUCTURED_SIGNALS.search(question))
    has_unstructured = bool(_UNSTRUCTURED_SIGNALS.search(question))

    if has_structured and not has_unstructured:
        return IntentLabel.STRUCTURED, "rule_high_structured"
    if has_unstructured and not has_structured:
        return IntentLabel.UNSTRUCTURED, "rule_high_unstructured"
    return None, "rule_uncertain"


def detect_intent_llm(question: str, history_str: str) -> IntentLabel:
    """Uses llm_intent (temp=0.0) for ambiguous queries. Expects single-word response."""
    prompt = INTENT_DETECTION_PROMPT.format(history=history_str, question=question)
    raw = llm_intent.invoke(prompt).content.strip().lower()
    if "structured" in raw:
        return IntentLabel.STRUCTURED
    return IntentLabel.UNSTRUCTURED


def detect_intent(question: str, history: list) -> tuple:
    """
    Returns (IntentLabel, confidence_tag: str).
    Rule-based first; LLM fallback only when uncertain.
    On LLM failure: defaults to UNSTRUCTURED.
    """
    intent, confidence = _rule_based_intent(question)
    if intent is not None:
        return intent, confidence

    history_str = format_history(history, max_turns=6)
    try:
        intent = detect_intent_llm(question, history_str)
        return intent, "rule_uncertain_llm"
    except Exception as e:
        logger.warning(f"Intent LLM failed ({e}) -- falling back to UNSTRUCTURED")
        return IntentLabel.UNSTRUCTURED, "rule_uncertain_llm_fallback"


# ============================
# STRUCTURED PATH
# ============================

def run_structured_path(question: str, history: list, limit: int, offset: int) -> tuple:
    """
    Returns (answer: str, Optional[SQLResult]).
    Dedicated SQL-generation-only prompt; no routing logic in the prompt.
    """
    BACK_REF = {"it", "that", "those", "them", "this", "same", "previous", "above"}
    needs_long = len(history) > 10 and bool(BACK_REF & set(question.lower().split()))
    history_str = format_history(history, max_turns=10 if needs_long else 6)

    schema_context = build_schema_context(list(TABLE_SCHEMAS.keys()))

    cache_key = _make_cache_key([question, history_str])
    cached = _cache_get(_SQL_CACHE, cache_key)
    if cached is not None:
        raw_sql = cached
        logger.info(f"SQL cache HIT key={cache_key[:8]}")
    else:
        prompt = SQL_GENERATION_PROMPT.format(
            schema_context=schema_context,
            history=history_str,
            question=question
        )
        raw_sql = llm_sql.invoke(prompt).content.strip()
        # Strip accidental markdown fences
        raw_sql = re.sub(r'```(?:sql)?\s*|\s*```', '', raw_sql).strip()
        _cache_set(_SQL_CACHE, cache_key, raw_sql, SQL_CACHE_TTL_SECONDS)

    logger.info(f"Generated SQL: {raw_sql[:200]}")

    # CANNOT_GENERATE sentinel -- graceful 200, no HTTP error
    if raw_sql.strip().upper() == "CANNOT_GENERATE":
        return (
            "I wasn't able to generate a query for that question. "
            "Could you try rephrasing or provide more specific details?",
            None
        )

    is_safe, reason = validate_sql(raw_sql)
    if not is_safe:
        logger.warning(f"SQL failed safety check: {reason} | SQL: {raw_sql[:200]}")
        raise HTTPException(status_code=400, detail=f"Query blocked: {reason}")

    limited_sql = _inject_top(raw_sql, limit)
    try:
        all_rows, columns = execute_sql_safely(limited_sql, limit=limit)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SQL execution error: {e}")
        raise HTTPException(status_code=500, detail=f"SQL execution failed: {str(e)}")

    total_rows = len(all_rows)
    paginated = all_rows[offset:]

    sql_result = SQLResult(
        sql=raw_sql,
        columns=columns,
        rows=paginated,
        total_rows=total_rows,
        returned_rows=len(paginated),
        limit=limit,
        offset=offset
    )

    answer = synthesize_sql_answer(question, columns, paginated, total_rows, limit)
    return answer, sql_result


# ============================
# UNSTRUCTURED (RAG) PATH
# ============================

def _format_retrieved_docs(docs: list, max_chars: int = RAG_MAX_CONTEXT_CHARS) -> tuple:
    """
    Returns (context_str: str, sources: List[str]).
    Truncates combined context to max_chars; extracts source table names from metadata.
    """
    chunks = []
    sources: List[str] = []
    total = 0

    for doc in docs:
        content = doc.page_content or ""
        src = doc.metadata.get("table") or doc.metadata.get("source", "")

        if total + len(content) > max_chars:
            content = content[:max_chars - total]
            chunks.append(content)
            if src and src not in sources:
                sources.append(src)
            break

        chunks.append(content)
        total += len(content)

        if src and src not in sources:
            sources.append(src)

    return "\n\n---\n\n".join(chunks), sources


def run_unstructured_path(question: str, history: list) -> tuple:
    """
    Returns (answer: str, sources: List[str]).
    Retrieves from Qdrant (k=8) then synthesises with llm_rag.
    """
    if retriever is None:
        return "Vector search unavailable. Try a structured query.", []

    history_str = format_history(history, max_turns=6)

    cache_key = _make_cache_key([question, history_str])
    cached = _cache_get(_RAG_CACHE, cache_key)
    if cached is not None:
        logger.info(f"RAG cache HIT key={cache_key[:8]}")
        return cached["answer"], cached["sources"]

    try:
        docs = retriever.invoke(question)
    except Exception as e:
        logger.error(f"Qdrant retrieval error: {e}")
        return "Vector search unavailable. Try a structured query.", []

    context_str, sources = _format_retrieved_docs(docs, max_chars=RAG_MAX_CONTEXT_CHARS)

    if not context_str.strip():
        return "I don't have enough context to answer that.", sources

    prompt = RAG_SYNTHESIS_PROMPT.format(
        context=context_str,
        history=history_str,
        question=question
    )

    try:
        answer = llm_rag.invoke(prompt).content.strip()
    except Exception as e:
        logger.error(f"RAG LLM error: {e}")
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    _cache_set(_RAG_CACHE, cache_key, {"answer": answer, "sources": sources}, SQL_CACHE_TTL_SECONDS)
    return answer, sources


# ============================
# ORCHESTRATOR
# ============================

async def orchestrate(
    question: str,
    history: list,
    limit: int,
    offset: int
) -> tuple:
    """
    Returns (answer, path, confidence, sql_result, sources).
    Routes to structured or unstructured path based on detected intent.
    """
    path, confidence = detect_intent(question, history)
    logger.info(f"Intent: {path} [{confidence}] | question={question[:80]}")

    if path == IntentLabel.STRUCTURED:
        answer, sql_result = run_structured_path(question, history, limit, offset)
        return answer, path, confidence, sql_result, None
    else:
        answer, sources = run_unstructured_path(question, history)
        return answer, path, confidence, None, sources


# ============================
# LIFESPAN
# ============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global TABLE_SCHEMAS, vector_db, retriever, llm_intent, llm_sql, llm_rag, _app_start_time

    logger.info("Starting NEW_RAG Assistant API...")
    logger.info(f"Schemas to load: {KNOWN_SCHEMAS}")

    # 1+2. Build schema map -- FAIL FAST if nothing found
    TABLE_SCHEMAS = powerful_schema_catcher_multi(engine, KNOWN_SCHEMAS)
    if not TABLE_SCHEMAS:
        raise RuntimeError("No tables found in any configured schema. Check DB_SCHEMAS env var.")

    schema_counts: Dict[str, int] = {}
    for meta in TABLE_SCHEMAS.values():
        s = meta["schema"]
        schema_counts[s] = schema_counts.get(s, 0) + 1
    for s, cnt in schema_counts.items():
        logger.info(f"  Schema '{s}': {cnt} tables loaded")

    # 3. Connect Qdrant -- warn & degrade, do NOT abort
    qdrant_available = False
    try:
        client = connect_qdrant()
        client.get_collection(QDRANT_COLLECTION)
        qdrant_available = True
        logger.info(f"Qdrant collection '{QDRANT_COLLECTION}' found")
    except Exception as e:
        logger.warning(
            f"Qdrant collection '{QDRANT_COLLECTION}' not found or unreachable: {e}. "
            "RAG path will be unavailable; SQL path still operational."
        )

    # 4. Load vector store + retriever (skip if Qdrant unavailable)
    if qdrant_available:
        try:
            vector_db = load_vector_store()
            retriever = vector_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": RAG_TOP_K}
            )
            logger.info(f"Retriever ready (k={RAG_TOP_K})")
        except Exception as e:
            logger.warning(f"Failed to load vector store: {e}. RAG path disabled.")

    # 5. Instantiate LLMs
    groq_key = os.getenv("GROQ_API_KEY")
    llm_intent = ChatGroq(api_key=groq_key, model=GROQ_MODEL, temperature=0.0)
    llm_sql    = ChatGroq(api_key=groq_key, model=GROQ_MODEL, temperature=0.1)
    llm_rag    = ChatGroq(api_key=groq_key, model=GROQ_MODEL, temperature=0.4)

    _app_start_time = time.time()
    logger.info(
        f"NEW_RAG Assistant ready | {len(TABLE_SCHEMAS)} total tables | "
        f"schemas={list(schema_counts.keys())} | qdrant={'yes' if qdrant_available else 'no'}"
    )

    yield


# ============================
# FASTAPI APP
# ============================

ALLOWED_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

app = FastAPI(
    root_path="/newchat",
    title="NEW RAG Chat Assistant API",
    description=(
        "Enterprise chat assistant with discrete intent detection, multi-schema SQL generation, "
        "and active Qdrant RAG path. All queries are read-only -- the database cannot be modified."
    ),
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"]
)


# ============================
# EXCEPTION HANDLERS
# ============================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            request_id=str(uuid.uuid4()),
            error=exc.detail,
            timestamp=datetime.utcnow().isoformat()
        ).model_dump(),
        headers=getattr(exc, "headers", None) or {}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            request_id=str(uuid.uuid4()),
            error="Internal server error",
            timestamp=datetime.utcnow().isoformat()
        ).model_dump()
    )


# ============================
# ENDPOINTS
# ============================

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(body: ChatRequest):
    """
    Orchestrated chat endpoint.

    - Detects intent (structured vs unstructured) via rules then LLM fallback.
    - Structured path: generates T-SQL, executes read-only, returns sql_result.
    - Unstructured path: retrieves from Qdrant (k=8), synthesises answer with llm_rag.
    - Pass `session_id` from a previous response to continue a conversation.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()

    cleanup_expired_sessions()

    q = body.question.strip()
    if not q:
        raise HTTPException(status_code=422, detail="Question cannot be empty")
    if len(q) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Question exceeds {MAX_QUESTION_LENGTH} character limit"
        )

    session_id, history = get_or_create_session(body.session_id)
    effective_limit = min(body.limit, MAX_SQL_LIMIT)

    try:
        answer, path, confidence, sql_result, sources = await orchestrate(
            question=q,
            history=history,
            limit=effective_limit,
            offset=body.offset
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Orchestrate error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal processing error")

    append_to_history(session_id, "user", q)
    append_to_history(session_id, "assistant", answer)

    elapsed_ms = int((time.time() - start_time) * 1000)
    logger.info(
        f"[{request_id}] session={session_id} path={path} "
        f"confidence={confidence} elapsed={elapsed_ms}ms"
    )

    response_data = ChatResponse(
        request_id=request_id,
        session_id=session_id,
        answer=answer,
        path=path,
        sql_result=sql_result,
        sources=sources,
        timestamp=datetime.utcnow().isoformat(),
        execution_time_ms=elapsed_ms,
        intent_confidence=confidence
    )

    return JSONResponse(
        content=response_data.model_dump(),
        headers={"X-Request-ID": request_id}
    )


@app.get("/health", response_model=HealthResponse, tags=["Observability"])
async def health_check():
    """Liveness check -- no auth required. Safe for load balancers and monitoring."""
    db_ok = False
    try:
        with engine.connect() as conn:
            conn.execute(sa_text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    qdrant_ok = False
    try:
        client = connect_qdrant()
        client.get_collection(QDRANT_COLLECTION)
        qdrant_ok = True
    except Exception:
        pass

    llm_ready = llm_intent is not None and llm_sql is not None and llm_rag is not None

    schemas_seen: List[str] = []
    for meta in TABLE_SCHEMAS.values():
        s = meta.get("schema", "")
        if s and s not in schemas_seen:
            schemas_seen.append(s)

    overall = "ok" if (db_ok and llm_ready) else "degraded"

    return HealthResponse(
        status=overall,
        db_connected=db_ok,
        qdrant_connected=qdrant_ok,
        llm_ready=llm_ready,
        schema_tables_loaded=len(TABLE_SCHEMAS),
        schemas_loaded=schemas_seen,
        uptime_seconds=round(time.time() - _app_start_time, 2),
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/schema", response_model=SchemaResponse, tags=["Schema"])
async def get_schema():
    """
    Returns all tables across all loaded schemas with qualified names.
    Frontend can use this for schema browsers or table auto-suggest.
    """
    tables = [
        SchemaTable(
            schema_name=meta["schema"],
            table=meta["table"],
            qualified_name=meta["qualified"],
            columns=meta["columns"],
            primary_key=meta.get("pk"),
            row_count=meta.get("count", 0)
        )
        for meta in TABLE_SCHEMAS.values()
    ]

    schemas_seen: List[str] = []
    for t in tables:
        if t.schema_name not in schemas_seen:
            schemas_seen.append(t.schema_name)

    return SchemaResponse(
        tables=tables,
        total_tables=len(tables),
        schemas=schemas_seen,
        timestamp=datetime.utcnow().isoformat()
    )


@app.delete("/session/{session_id}", tags=["Session"])
async def delete_session(session_id: str):
    """
    Clears a conversation session. The next request with this session_id
    will start a fresh conversation.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found or already expired")
    del _sessions[session_id]
    return {"message": "Session deleted", "session_id": session_id}


# ============================
# RUN
# ============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chatbt:app",port=8001, reload=True)
