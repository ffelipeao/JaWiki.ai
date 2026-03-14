#!/usr/bin/env python3
"""
Exemplo de busca por linha de comando: pergunta -> trechos mais similares no RAG.
"""
from __future__ import annotations

from typing import List

from core.db import Database, RagRow
from core.vectorstore import get_embeddings


def query_top_k(question: str, k: int = 3) -> List[RagRow]:
    """Recebe uma pergunta, vetoriza com Ollama e busca os k documentos mais próximos no banco."""
    embeddings_model = get_embeddings()
    query_embedding = embeddings_model.embed_query(question)
    db = Database()
    try:
        return db.query_similar(query_embedding, k=k)
    finally:
        db.close()


def _print_results(results: List[RagRow]) -> None:
    for i, row in enumerate(results, start=1):
        print("=" * 80)
        print(f"Resultado #{i} - id={row.id}")
        print("-" * 80)
        print(row.conteudo)
        print()


if __name__ == "__main__":
    pergunta = input("Digite sua pergunta: ")
    docs = query_top_k(pergunta, k=3)
    _print_results(docs)
