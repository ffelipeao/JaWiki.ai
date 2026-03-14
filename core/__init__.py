"""
Pacote core: ingestão RAG, banco de dados, embeddings e limpeza de texto.
Separa a lógica de ingestão e geração RAG da aplicação web.
"""
from core.config import DATA_DIR, validate_paths
from core.db import Database, RagRow
from core.vectorstore import get_embeddings
from core.text_cleaning import clean_for_display, strip_dokuwiki_markup

__all__ = [
    "DATA_DIR",
    "validate_paths",
    "Database",
    "RagRow",
    "get_embeddings",
    "clean_for_display",
    "strip_dokuwiki_markup",
]
