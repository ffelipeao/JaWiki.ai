import re
from typing import Iterable


DKW_LINK_PATTERN = re.compile(r"\[\[(?P<link>[^\]|]+)(\|(?P<label>[^\]]+))?\]\]")
DKW_BOLD_PATTERN = re.compile(r"\*\*(?P<text>[^*]+)\*\*")
DKW_ITALIC_PATTERN = re.compile(r"//(?P<text>[^/]+)//")
DKW_HEADERS_PATTERN = re.compile(r"^={2,6}\s*(?P<text>.+?)\s*={2,6}\s*$", re.MULTILINE)
DKW_MONO_PATTERN = re.compile(r"''(?P<text>[^']+)''")  # texto monoespaçado

DKW_IMAGE_DOUBLE = re.compile(r"\{\{::[^}|]+\|?\}\}")
DKW_IMAGE_SINGLE = re.compile(r"\{\{:[^}|]+\|?\}\}")
DKW_MEDIA_ANY = re.compile(r"\{\{[^}]*\|?\}\}")
DKW_LINEBREAK_BACKSLASH = re.compile(r"\\\\+")
DKW_HORIZONTAL_RULE = re.compile(r"^-{4,}\s*$", re.MULTILINE)


def _replace_link(match: re.Match) -> str:
    label = match.group("label")
    link = match.group("link")
    return (label or link).strip()


def strip_dokuwiki_markup(text: str) -> str:
    """Remove/normaliza marcações comuns do DokuWiki para embeddings."""
    text = DKW_LINK_PATTERN.sub(_replace_link, text)
    text = DKW_BOLD_PATTERN.sub(lambda m: m.group("text"), text)
    text = DKW_ITALIC_PATTERN.sub(lambda m: m.group("text"), text)
    text = DKW_MONO_PATTERN.sub(lambda m: m.group("text"), text)
    text = DKW_HEADERS_PATTERN.sub(lambda m: m.group("text"), text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_texts(texts: Iterable[str]) -> list[str]:
    """Aplica limpeza em lote."""
    return [strip_dokuwiki_markup(t) for t in texts]


def clean_for_display(text: str) -> str:
    """Remove artefatos DokuWiki para exibição/LLM (imagens, \\\\, ----)."""
    if not text or not text.strip():
        return text
    text = DKW_IMAGE_DOUBLE.sub("[Imagem]", text)
    text = DKW_IMAGE_SINGLE.sub("[Imagem]", text)
    text = DKW_MEDIA_ANY.sub("[Mídia]", text)
    text = DKW_LINEBREAK_BACKSLASH.sub("\n", text)
    text = DKW_HORIZONTAL_RULE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n ", "\n", text)
    return text.strip()
