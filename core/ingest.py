"""
Pipeline de ingestão RAG: lê .txt do manual, limpa, chunka, gera embeddings e grava no PostgreSQL.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from core.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    validate_paths,
)
from core.text_cleaning import clean_for_display, strip_dokuwiki_markup
from core.db import Database
from core.vectorstore import get_embeddings


def iter_txt_files(root_dir: Path) -> Iterable[Path]:
    """Percorre recursivamente root_dir retornando apenas arquivos .txt."""
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(".txt"):
                yield Path(dirpath) / name


def extract_document_title(raw_text: str) -> str | None:
    """Se a primeira linha for cabeçalho DokuWiki (====== Título ======), retorna o título limpo."""
    first_line = (raw_text.split("\n")[0] or "").strip()
    if first_line.startswith("=") and first_line.endswith("="):
        return first_line.strip("= ").strip()
    return None


def path_to_theme(relative_path: Path) -> str:
    """Deriva o tema a partir do caminho (ex.: validar_planilha.txt -> validar_planilha)."""
    s = str(relative_path).replace("\\", "/")
    if s.lower().endswith(".txt"):
        s = s[:-4]
    return s.strip("/") or "geral"


def humanize_theme(theme: str) -> str:
    """Tema legível para o embedding (ex.: adicionar_nova_determinacao -> adicionar nova determinacao)."""
    return theme.replace("_", " ").strip()


def chunk_text(text: str, source: str, theme: str, assunto_label: str) -> List[Document]:
    """Divide o texto em chunks e prefixa cada um com o assunto."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    prefix = f"Assunto: {assunto_label}.\n\n"
    return [
        Document(
            page_content=prefix + chunk,
            metadata={"source": source, "theme": theme, "chunk_index": idx},
        )
        for idx, chunk in enumerate(chunks)
    ]


def build_documents_from_folder(root_dir: Path) -> List[Document]:
    """Lê todos os .txt de root_dir, limpa, deriva tema/título e gera chunks."""
    documents: List[Document] = []
    for path in iter_txt_files(root_dir):
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        if not raw_text.strip():
            continue
        title = extract_document_title(raw_text)
        cleaned = strip_dokuwiki_markup(raw_text)
        cleaned = clean_for_display(cleaned)
        if not cleaned.strip():
            continue
        relative = path.relative_to(root_dir)
        source = str(relative)
        theme = path_to_theme(relative)
        assunto_label = title if title else humanize_theme(theme)
        docs = chunk_text(cleaned, source=source, theme=theme, assunto_label=assunto_label)
        documents.extend(docs)
    return documents


def ingest(limpar_base: bool = False) -> None:
    """
    Pipeline completo de ingestão: lê .txt, limpa, chunka, gera embeddings e persiste no banco RAG.
    limpar_base: se True, apaga a tabela antes de inserir (re-ingestão completa).
    """
    validate_paths()
    print(f"Lendo arquivos .txt de {DATA_DIR} ...")
    documents = build_documents_from_folder(DATA_DIR)
    if not documents:
        print("Nenhum documento válido encontrado para ingestão.")
        return
    print(f"{len(documents)} chunks gerados. Calculando embeddings...")
    embeddings_model = get_embeddings()
    textos = [doc.page_content for doc in documents]
    embeddings = embeddings_model.embed_documents(textos)
    print("Gravando embeddings na tabela RAG...")
    db = Database()
    if limpar_base:
        db.clear_table()
        print("Base limpa. Inserindo documentos...")
    db.insert_embeddings(textos, embeddings)
    db.close()
    print("Ingestão concluída com sucesso.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingestão de documentos .txt para o banco RAG (jawiki.jabot_rag)."
    )
    parser.add_argument(
        "--limpar",
        action="store_true",
        help="Limpar a tabela RAG antes de inserir (re-ingestão completa).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ingest(limpar_base=args.limpar)
