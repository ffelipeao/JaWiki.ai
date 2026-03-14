from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

from core.config import (
    CACHE_EMBEDDING_DIM,
    CACHE_SCHEMA,
    CACHE_TABLE,
    CACHE_THRESHOLD,
    DATABASE_URL,
    RAG_SCHEMA,
    RAG_TABLE,
)


def _as_vector(embedding: Union[list[float], np.ndarray]) -> np.ndarray:
    """Converte para numpy array para o pgvector serializar como tipo vector (não array)."""
    return np.array(embedding, dtype=np.float32) if not isinstance(embedding, np.ndarray) else embedding


@dataclass
class RagRow:
    id: int
    conteudo: str
    # Não expomos o embedding aqui para não trafegar vetor grande desnecessariamente.


class Database:
    """
    Classe de conexão/integração com o PostgreSQL + pgvector.
    Usa a tabela existente (jawiki.jabot_rag com id, conteudo, embedding).
    """

    def __init__(self, dsn: str | None = None) -> None:
        self._dsn = dsn or DATABASE_URL.replace("+psycopg2", "")
        self._conn = None

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self._dsn)
            register_vector(self._conn)
        return self._conn

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            self._conn = None

    def clear_table(self) -> None:
        """Remove todos os registros da tabela de RAG (cuidado, destrutivo)."""
        with self.conn, self.conn.cursor() as cur:
            cur.execute(f'TRUNCATE TABLE "{RAG_SCHEMA}"."{RAG_TABLE}" RESTART IDENTITY')

    def insert_embeddings(
        self,
        textos: Iterable[str],
        embeddings: Iterable[Union[list[float], np.ndarray]],
    ) -> None:
        """Insere lote de (conteudo, embedding) na tabela RAG."""
        rows: List[Tuple[str, np.ndarray]] = [
            (texto, _as_vector(emb)) for texto, emb in zip(textos, embeddings)
        ]
        if not rows:
            return
        with self.conn, self.conn.cursor() as cur:
            cur.executemany(
                f'INSERT INTO "{RAG_SCHEMA}"."{RAG_TABLE}" (conteudo, embedding) '
                f"VALUES (%s, %s)",
                rows,
            )

    def query_similar(
        self,
        query_embedding: Union[list[float], np.ndarray],
        k: int = 3,
    ) -> List[RagRow]:
        """Retorna os k registros mais similares usando distância vetorial (pgvector)."""
        vec = _as_vector(query_embedding)
        with self.conn, self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, conteudo
                FROM "{RAG_SCHEMA}"."{RAG_TABLE}"
                ORDER BY embedding <-> %s
                LIMIT %s
                """,
                (vec, k),
            )
            rows = cur.fetchall()
        return [RagRow(id=row[0], conteudo=row[1]) for row in rows]

    # ---------- cache de perguntas respondidas ----------

    def ensure_cache_table(self) -> None:
        """Cria a tabela de cache de perguntas se não existir (esquema e coluna vector com dimensão fixa)."""
        with self.conn, self.conn.cursor() as cur:
            cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{CACHE_SCHEMA}"')
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS "{CACHE_SCHEMA}"."{CACHE_TABLE}" (
                    id serial PRIMARY KEY,
                    pergunta text NOT NULL,
                    embedding vector({CACHE_EMBEDDING_DIM}) NOT NULL,
                    resposta text NOT NULL,
                    fontes text,
                    created_at timestamp with time zone DEFAULT now()
                )
                """
            )

    def get_cached_answer(
        self,
        query_embedding: Union[list[float], np.ndarray],
        max_distance: float = CACHE_THRESHOLD,
    ) -> Optional[Tuple[str, list]]:
        """
        Busca uma pergunta muito similar no cache. Retorna (resposta, fontes_list) ou None.
        """
        vec = _as_vector(query_embedding)
        with self.conn, self.conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT resposta, fontes, (embedding <-> %s) AS dist
                FROM "{CACHE_SCHEMA}"."{CACHE_TABLE}"
                ORDER BY embedding <-> %s
                LIMIT 1
                """,
                (vec, vec),
            )
            row = cur.fetchone()
            if not row or row[2] is None or float(row[2]) > max_distance:
                return None
            resposta, fontes_raw = row[0], row[1]
            fontes_list = json.loads(fontes_raw) if fontes_raw else []
            return (resposta, fontes_list)

    def add_cached(
        self,
        pergunta: str,
        embedding: Union[list[float], np.ndarray],
        resposta: str,
        fontes: list[dict],
    ) -> None:
        """Salva pergunta, embedding, resposta e fontes no cache para reutilização futura."""
        vec = _as_vector(embedding)
        fontes_json = json.dumps(fontes, ensure_ascii=False)
        with self.conn, self.conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO "{CACHE_SCHEMA}"."{CACHE_TABLE}" (pergunta, embedding, resposta, fontes)
                VALUES (%s, %s, %s, %s)
                """,
                (pergunta, vec, resposta, fontes_json),
            )


__all__ = ["Database", "RagRow"]
