import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv


# Carrega variáveis de ambiente a partir de um arquivo .env, se existir
load_dotenv()


# Caminho base dos arquivos .txt exportados do DokuWiki
DATA_DIR: Path = Path(os.getenv("DATA_DIR", "./manual_txt")).resolve()

# Configurações do banco PostgreSQL + pgvector
# Use variáveis separadas no .env ou DATABASE_URL completa
DB_HOST: str = os.getenv("DB_HOST", "localhost")
DB_PORT: str = os.getenv("DB_PORT", "5432")
DB_USER: str = os.getenv("DB_USER", "postgres")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "postgres")
DB_NAME: str = os.getenv("DB_NAME", "rag_db")

# Se DATABASE_URL estiver definida, ela tem prioridade; senão montamos a partir das variáveis acima
_raw_url = os.getenv("DATABASE_URL")
if _raw_url:
    DATABASE_URL: str = _raw_url.replace("+psycopg2", "")
else:
    _pw = quote_plus(DB_PASSWORD)  # senhas com caracteres especiais
    DATABASE_URL = f"postgresql://{DB_USER}:{_pw}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Esquema e tabela já existentes para o RAG
RAG_SCHEMA: str = os.getenv("RAG_SCHEMA", "jawiki")
RAG_TABLE: str = os.getenv("RAG_TABLE", "jabot_rag")

# Modelo de embedding rodando no Ollama
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mxbai-embed-large")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Parâmetros de chunking
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# Cache de perguntas já respondidas (reutiliza respostas para perguntas muito parecidas)
CACHE_ENABLED: bool = os.getenv("CACHE_PERGUNTAS", "1").lower() in ("1", "true", "yes")
CACHE_SCHEMA: str = os.getenv("CACHE_SCHEMA", RAG_SCHEMA)
CACHE_TABLE: str = os.getenv("CACHE_TABLE", "jabot_rag_perguntas")
# Distância máxima (pgvector <->) para considerar "mesma" pergunta. Quanto menor, mais exigente (ex.: 0.2 a 0.4).
CACHE_THRESHOLD: float = float(os.getenv("CACHE_THRESHOLD", "0.35"))
# Reutilizar cache apenas quando a resposta tiver avaliação média >= este valor (1-5). Ex.: 4 = só 4 ou 5 estrelas.
CACHE_MIN_STARS: float = float(os.getenv("CACHE_MIN_STARS", "4"))
# Dimensão do embedding (mxbai-embed-large = 1024). Deve bater com o modelo.
CACHE_EMBEDDING_DIM: int = int(os.getenv("CACHE_EMBEDDING_DIM", "1024"))


def validate_paths() -> None:
    """
    Valida pastas configuradas.
    Lança erro claro se a pasta de dados não existir.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Pasta de dados não encontrada: {DATA_DIR}. "
            f"Crie a pasta e coloque os .txt exportados do DokuWiki nela."
        )
