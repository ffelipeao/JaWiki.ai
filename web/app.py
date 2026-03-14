"""
Aplicação web local: chat que consulta o banco RAG e opcionalmente gera resposta com Ollama.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request

from core.config import OLLAMA_BASE_URL
from core.db import Database, RagRow
from core.text_cleaning import clean_for_display
from core.vectorstore import get_embeddings

# Flask procura templates em web/templates/ (ao lado deste módulo)
_web_dir = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(_web_dir / "templates"))

CHAT_TOP_K = int(os.getenv("CHAT_TOP_K", "5"))
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "")


def _query_rag(pergunta: str, k: int = CHAT_TOP_K) -> list[RagRow]:
    """Busca os k trechos mais relevantes no banco."""
    embeddings = get_embeddings()
    query_embedding = embeddings.embed_query(pergunta)
    db = Database()
    try:
        return db.query_similar(query_embedding, k=k)
    finally:
        db.close()


def _build_answer_from_context(pergunta: str, sources: list[RagRow]) -> str:
    """Monta resposta a partir dos trechos ou usa LLM se OLLAMA_CHAT_MODEL estiver definido."""
    if not sources:
        return "Não encontrei informações relevantes no banco para essa pergunta."
    cleaned_parts = [clean_for_display(s.conteudo) for s in sources]
    context = "\n\n---\n\n".join(cleaned_parts)

    if not OLLAMA_CHAT_MODEL:
        return (
            "Com base na base de conhecimento:\n\n"
            + context
            + "\n\n_(Defina OLLAMA_CHAT_MODEL no .env para gerar respostas com IA.)_"
        )

    try:
        import requests
        prompt = f"""Você é um assistente que responde dúvidas sobre o sistema com base na documentação disponível.

FLUXO:
1. Leia a PERGUNTA DO USUÁRIO abaixo.
2. O texto em "CONTEÚDO BUSCADO NO BANCO" foi recuperado por similaridade; pode conter trechos que não têm relação com a pergunta.
3. INTERPRETE o conteúdo: use apenas as partes que realmente respondem à dúvida do usuário. IGNORE trechos que não tenham relação com a pergunta.
4. Responda ao usuário com base na pergunta dele, usando só o que for relevante.

REGRAS:
- Os trechos podem começar com "Assunto: <tema>". Priorize informações do mesmo assunto e evite misturar temas diferentes.
- Sua resposta deve ser dirigida à pergunta. Use APENAS informações do conteúdo abaixo que tenham relação com a pergunta.
- Escreva em PASSO A PASSO numerado (1., 2., 3., ...) quando for explicar como fazer algo no sistema.
- Se nada do conteúdo for relevante, responda: "Não encontrei informação sobre isso na base de conhecimento."

PERGUNTA DO USUÁRIO:
{pergunta}

CONTEÚDO BUSCADO NO BANCO (use apenas o que for relevante):
---
{context}
---

Responda ao usuário com base na pergunta, usando somente o que for pertinente:"""

        r = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json={"model": OLLAMA_CHAT_MODEL, "prompt": prompt, "stream": False},
            timeout=90,
        )
        r.raise_for_status()
        out = r.json()
        return (out.get("response") or "").strip() or context
    except Exception as e:
        return "Erro ao gerar resposta com o modelo. Mostrando os trechos:\n\n" + context + f"\n\n(Erro: {e})"


@app.route("/")
def index():
    """Página principal do chat."""
    return render_template("chat.html")


def _build_answer_from_context_with_history(
    pergunta: str,
    sources: list[RagRow],
    history: list[dict[str, str]],
) -> str:
    """Como _build_answer_from_context, mas inclui o histórico da conversa no prompt do LLM."""
    if not sources:
        return "Não encontrei informações relevantes no banco para essa pergunta."
    cleaned_parts = [clean_for_display(s.conteudo) for s in sources]
    context = "\n\n---\n\n".join(cleaned_parts)

    if not OLLAMA_CHAT_MODEL:
        return (
            "Com base na base de conhecimento:\n\n"
            + context
            + "\n\n_(Defina OLLAMA_CHAT_MODEL no .env para gerar respostas com IA.)_"
        )

    history_text = ""
    if history:
        lines = []
        for h in history[-10:]:  # últimas 10 mensagens (5 pares user/bot)
            role = "Usuário" if (h.get("role") == "user") else "Assistente"
            lines.append(f"{role}: {h.get('content', '').strip()}")
        history_text = "HISTÓRICO RECENTE DA CONVERSA:\n" + "\n".join(lines) + "\n\n"

    try:
        import requests
        prompt = f"""Você é um assistente que responde dúvidas sobre o sistema com base na documentação disponível.

{history_text}PERGUNTA ATUAL DO USUÁRIO:
{pergunta}

CONTEÚDO BUSCADO NO BANCO (use apenas o que for relevante):
---
{context}
---

REGRAS: Use apenas informações do conteúdo acima. Responda em passo a passo quando for "como fazer". Se nada for relevante, diga que não encontrou. Considere o histórico acima para dar continuidade à conversa (ex.: "como disse antes", "o passo 2 é...")."""

        r = requests.post(
            f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate",
            json={"model": OLLAMA_CHAT_MODEL, "prompt": prompt, "stream": False},
            timeout=90,
        )
        r.raise_for_status()
        out = r.json()
        return (out.get("response") or "").strip() or context
    except Exception as e:
        return "Erro ao gerar resposta. Mostrando os trechos:\n\n" + context + f"\n\n(Erro: {e})"


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Recebe uma pergunta (e opcionalmente histórico). Body: { "message": "...", "history": [{ "role": "user"|"bot", "content": "..." }] }"""
    data: dict[str, Any] = request.get_json() or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"ok": False, "error": "Campo 'message' é obrigatório."}), 400
    history = data.get("history") or []
    if not isinstance(history, list):
        history = []
    try:
        sources = _query_rag(message, k=CHAT_TOP_K)
        if history:
            answer = _build_answer_from_context_with_history(message, sources, history)
        else:
            answer = _build_answer_from_context(message, sources)
        sources_cleaned = []
        for s in sources:
            c = clean_for_display(s.conteudo)
            sources_cleaned.append({"id": s.id, "conteudo": c[:500] + ("..." if len(c) > 500 else "")})
        return jsonify({"ok": True, "answer": answer, "sources": sources_cleaned})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


def run_app() -> None:
    """Ponto de entrada para rodar o servidor Flask."""
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "1").lower() in ("1", "true", "yes")
    app.run(host=host, port=port, debug=debug)
