#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
import json
import math
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from openai import OpenAI
from sqlalchemy import (
    JSON,
    Integer,
    String,
    create_engine,
    func,
    select,
    text as sql_text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Config
# ---------------------------

OPENAI_MODEL_COMPLETION = os.getenv("OPENAI_MODEL_COMPLETION", "gpt-5-mini")
OPENAI_MODEL_EMBEDDING = os.getenv("OPENAI_MODEL_EMBEDDING", "text-embedding-3-large")
DATABASE_URL = os.getenv("STATEMENTS_DB_URL", "sqlite:///data/statements.db")
MAX_CONTEXT_SNIPPETS = int(os.getenv("MAX_CONTEXT_SNIPPETS", "12"))
DEFAULT_TOP_K = int(os.getenv("TOP_K", "8"))
DEFAULT_MIN_SIM = float(os.getenv("MIN_SIM", "0.78"))

client = OpenAI()

# ---------------------------
# SQLAlchemy Models
# ---------------------------


class Base(DeclarativeBase):
    pass


class Statement(Base):
    __tablename__ = "statements"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    line_id: Mapped[str] = mapped_column(
        String, unique=True, nullable=False, index=True
    )
    text: Mapped[str] = mapped_column(String, nullable=False)
    embedding: Mapped[Optional[list]] = mapped_column(JSON, nullable=True, index=False)


# ---------------------------
# Utilities
# ---------------------------


def get_engine():
    return create_engine(DATABASE_URL, future=True)


def init_db():
    engine = get_engine()
    Base.metadata.create_all(engine)


def batched(it: Iterable, n: int) -> Iterable[List]:
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def l2_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (l2_norm(a) * l2_norm(b)) or 1e-12
    return float(np.dot(a, b) / denom)


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    resp = client.embeddings.create(model=OPENAI_MODEL_EMBEDDING, input=list(texts))
    return [d.embedding for d in resp.data]


def llm_clean_query(user_q: str) -> str:
    prompt = (
        "You rewrite user questions into a concise standalone search query.\n"
        "Rules:\n"
        "- Keep the same language.\n"
        "- Remove filler words.\n"
        "- Keep key nouns/verbs and named entities.\n"
        "- 12 words or fewer.\n"
        f'User question: "{user_q}"\n'
        "Return only the cleaned query text, no extra words."
    )
    resp = client.responses.create(
        model=OPENAI_MODEL_COMPLETION,
        input=prompt,
    )
    return resp.output_text.strip()


def llm_answer(question: str, snippets: List[Tuple[int, float, str]]) -> dict:
    # snippets: list of (id, similarity, text)
    system = (
        "You are a careful assistant that answers strictly from provided statements. "
        "Cite statement IDs and explain why each is relevant."
    )
    context_blocks = []
    for sid, sim, txt in snippets[:MAX_CONTEXT_SNIPPETS]:
        context_blocks.append(f"[Statement {sid} | similarity={sim:.3f}]\n{txt}")
    context = "\n\n".join(context_blocks)
    user = (
        "Answer the user using only the statements below.\n"
        "If something is unknown, say so.\n"
        "Return JSON with keys:\n"
        "{\n"
        "  'answer': str,\n"
        "  'evidence': [\n"
        "     {'statement_id': int, 'relevance': float, 'why_relevant': str, 'quote': str}\n"
        "  ]\n"
        "}\n"
        f"Question: {question}\n\n"
        f"STATEMENTS:\n{context}\n"
        "JSON only:"
    )
    resp = client.responses.create(
        model=OPENAI_MODEL_COMPLETION,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(resp.output_text)
    except Exception:
        return {"answer": resp.output_text.strip(), "evidence": []}


# ---------------------------
# Ingestion
# ---------------------------


def ingest_textfile(path: str, batch_size: int = 256):
    engine = get_engine()
    with Session(engine) as sess:
        with open(path, "r", encoding="utf-8") as f:
            rows = []
            for i, line in enumerate(f, start=1):
                m = re.match(r"^([A-Za-z]+\s+\d+:\d+)\s+(.*)$", line)
                line_id, text = [x.strip() for x in m.groups()]
                if not text:
                    continue
                rows.append(Statement(line_id=line_id, text=text))
            sess.add_all(rows)
            sess.commit()

        # Embed any rows missing embeddings
        q = (
            select(Statement)
            .where(Statement.embedding.is_(None))
            .order_by(Statement.id.asc())
        )
        missing = list(sess.scalars(q))
        for chunk in batched(missing, batch_size):
            embs = embed_texts([s.text for s in chunk])
            for s, e in zip(chunk, embs):
                s.embedding = e
            sess.commit()


# ---------------------------
# Semantic Search
# ---------------------------


@dataclass
class SearchResult:
    id: int
    line_id: int
    text: str
    similarity: float


def search_similar(
    sess: Session,
    query_text: str,
    top_k: int = DEFAULT_TOP_K,
    min_sim: float = DEFAULT_MIN_SIM,
) -> List[SearchResult]:
    # Clean -> embed
    cleaned = llm_clean_query(query_text)
    q_vec = np.array(embed_texts([cleaned])[0], dtype=np.float32)

    # Pull embeddings (for larger DBs, consider vector DB or pgvector)
    rows = list(
        sess.execute(
            select(
                Statement.id,
                Statement.line_id,
                Statement.text,
                Statement.embedding,
            ).where(Statement.embedding.is_not(None))
        )
    ).copy()

    sims: List[SearchResult] = []
    for sid, line_id, text_val, emb in rows:
        v = np.array(emb, dtype=np.float32)
        sim = cosine_sim(q_vec, v)
        if sim >= min_sim:
            sims.append(
                SearchResult(id=sid, line_id=line_id, text=text_val, similarity=sim)
            )

    # Always take top_k by similarity; if nothing meets threshold, return best few
    sims.sort(key=lambda r: r.similarity, reverse=True)
    if sims:
        top_score = sims[0].similarity
        dyn_cut = max(min_sim, top_score * 0.92)  # dynamic tightening
        filtered = [s for s in sims if s.similarity >= dyn_cut]
        return (filtered[:top_k]) or sims[:top_k]
    return sims[:top_k]


# ---------------------------
# Orchestrator
# ---------------------------


def ask(question: str) -> dict:
    engine = get_engine()
    with Session(engine) as sess:
        hits = search_similar(
            sess, question, top_k=DEFAULT_TOP_K, min_sim=DEFAULT_MIN_SIM
        )
        snippets = [(h.id, h.similarity, h.text) for h in hits]
        result = llm_answer(question, snippets)
        result["_meta"] = {
            "cleaned_query": llm_clean_query(question),
            "top_k": DEFAULT_TOP_K,
            "min_sim": DEFAULT_MIN_SIM,
            "returned": len(hits),
            "hits": [
                {"id": h.id, "line_id": h.line_id, "similarity": h.similarity}
                for h in hits
            ],
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        return result


# ---------------------------
# CLI
# ---------------------------


def _cmd_init_db(_args):
    init_db()
    print("ok")


def _cmd_ingest(args):
    init_db()
    ingest_textfile(args.path, batch_size=args.batch_size, reembed=args.reembed)
    print("ok")


def _cmd_ask(args):
    out = ask(args.question)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def _cmd_search(args):
    engine = get_engine()
    with Session(engine) as sess:
        hits = search_similar(sess, args.query, top_k=args.top_k, min_sim=args.min_sim)
        for h in hits:
            print(f"{h.id}\t{h.similarity:.3f}\tL{h.line_id}\t{h.text}")


def main():
    parser = argparse.ArgumentParser(
        prog="statements_rag", description="Q&A over line-based statements"
    )
    sub = parser.add_subparsers(required=True)

    p0 = sub.add_parser("init-db", help="initialize database")
    p0.set_defaults(func=_cmd_init_db)

    p1 = sub.add_parser("ingest", help="ingest a text file and build embeddings")
    p1.add_argument("path", help="path to text file with one statement per line")
    p1.add_argument("--batch-size", type=int, default=256)
    p1.add_argument(
        "--reembed", action="store_true", help="drop and re-ingest + re-embed"
    )
    p1.set_defaults(func=_cmd_ingest)

    p2 = sub.add_parser("ask", help="ask a question")
    p2.add_argument("question", help="natural-language question")
    p2.set_defaults(func=_cmd_ask)

    p3 = sub.add_parser("search", help="search similar statements (debug)")
    p3.add_argument("query", help="query text")
    p3.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p3.add_argument("--min-sim", type=float, default=DEFAULT_MIN_SIM)
    p3.set_defaults(func=_cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
