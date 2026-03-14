from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from core.config import OLLAMA_BASE_URL, OLLAMA_MODEL


def get_embeddings() -> OllamaEmbeddings:
    """
    Retorna o objeto de embeddings conectado ao Ollama local.
    Certifique-se de que o serviço está rodando em OLLAMA_BASE_URL.
    """
    return OllamaEmbeddings(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
