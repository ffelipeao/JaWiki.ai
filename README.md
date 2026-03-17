# JaWiki.ai (Projeto migrado para o GitServer Institucional)

Sistema de **RAG (Retrieval-Augmented Generation)** local que usa o manual do sistema (arquivos `.txt` exportados do DokuWiki) para responder dúvidas dos usuários em formato de chat. As respostas são baseadas na base de conhecimento e, quando configurado, o Ollama gera respostas em passo a passo.

## O que o projeto faz

1. **Ingestão (ETL)**  
   Lê os `.txt` do manual (com suporte a subpastas), limpa marcações DokuWiki, divide em chunks, gera embeddings com Ollama e grava na tabela PostgreSQL com pgvector.

2. **Chat web**  
   Interface local onde o usuário faz perguntas; o sistema busca os trechos mais relevantes no banco e o Ollama (opcional) formula a resposta em passo a passo. Quando o modelo de chat está ativo, as respostas podem ser exibidas em **streaming** (tempo real via SSE) e o chat mantém um pequeno **histórico da conversa** (últimas mensagens) enviado ao modelo para respostas com mais contexto.

3. **Organização por tema**  
   O tema/assunto é derivado do caminho do arquivo (e do título da página quando existir), para a busca priorizar conteúdo do mesmo assunto.

## Pré-requisitos

- **Python 3.10+**
- **PostgreSQL** com extensão **pgvector**
- **Ollama** rodando localmente (para embeddings e, opcionalmente, para gerar respostas)
- Modelo de embedding no Ollama (ex.: `mxbai-embed-large`)
- Opcional: modelo de chat no Ollama (ex.: `llama3`) para respostas em linguagem natural

## Estrutura do projeto

O projeto está separado em **core** (ingestão e RAG) e **web** (aplicação de chat):

```
JaWiki.ai/
├── core/                     # Ingestão e geração RAG (independente da web)
│   ├── __init__.py
│   ├── config.py             # Configuração (lê .env)
│   ├── db.py                 # Conexão PostgreSQL + pgvector
│   ├── vectorstore.py        # Embeddings via Ollama
│   ├── text_cleaning.py      # Limpeza de marcações DokuWiki
│   └── ingest.py             # Pipeline de ingestão (ETL)
├── web/                      # Aplicação web (usa o core)
│   ├── __init__.py
│   ├── app.py                # Flask: chat e API /api/chat
│   └── templates/
│       └── chat.html         # Interface do chat
├── run_ingest.py             # Entrada: ingestão (python run_ingest.py [--limpar])
├── run_web.py                # Entrada: servidor do chat (python run_web.py)
├── query_example.py          # Exemplo de busca por linha de comando
├── manual_txt/               # .txt do manual (configurável via DATA_DIR)
├── .env                      # Variáveis de ambiente (não versionar)
├── .env.example
└── requirements.txt
```

## Configuração

1. **Clone/baixe o projeto** e crie o ambiente:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure o ambiente** copiando o exemplo e editando:

   ```bash
   cp .env.example .env
   ```

   Principais variáveis no `.env`:

   | Variável | Descrição |
   |----------|-----------|
   | `DATA_DIR` | Pasta dos arquivos `.txt` do manual (ex.: `./manual_txt`) |
   | `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` | Conexão PostgreSQL |
   | `RAG_SCHEMA`, `RAG_TABLE` | Esquema e tabela do RAG (ex.: `jawiki`, `jabot_rag`) |
   | `OLLAMA_MODEL` | Modelo de embedding no Ollama (ex.: `mxbai-embed-large`) |
   | `OLLAMA_BASE_URL` | URL do Ollama (ex.: `http://localhost:11434`) |
   | `OLLAMA_CHAT_MODEL` | (Opcional) Modelo para gerar respostas (ex.: `llama3`). Vazio = só mostra trechos. |
   | `CHAT_TOP_K` | Quantidade de trechos buscados por pergunta (ex.: `5`) |
   | `CACHE_PERGUNTAS` | `1` = ativa cache de perguntas (respostas similares reutilizadas); `0` = desligado |
   | `CACHE_THRESHOLD` | Distância máxima para considerar “mesma” pergunta (ex.: `0.35`). Quanto menor, mais exigente. |
   | `CACHE_MIN_STARS` | Reutilizar cache só para respostas com avaliação média ≥ N estrelas (ex.: `4` = só 4 ou 5 estrelas). |
   | `CACHE_EMBEDDING_DIM` | Dimensão do embedding (ex.: `1024` para mxbai-embed-large). Deve bater com o modelo. |
   | `FLASK_HOST`, `FLASK_PORT` | Host e porta do servidor do chat |

3. **Banco de dados**  
   Crie o banco e a extensão pgvector. A tabela usada deve ter pelo menos: `id`, `conteudo` (text), `embedding` (vector). Exemplo:

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   -- Tabela jawiki.jabot_rag com colunas id, conteudo, embedding
   ```

   Se `CACHE_PERGUNTAS=1`, a tabela de cache (`jawiki.jabot_rag_perguntas` por padrão) é criada automaticamente na primeira requisição do chat.

4. **Ollama**  
   Instale e suba o Ollama, e baixe o modelo de embedding:

   ```bash
   ollama pull mxbai-embed-large
   # Opcional, para respostas em texto:
   ollama pull llama3
   ```

## Uso

### Ingestão (popular a base RAG)

```bash
# Ingestão incremental (adiciona documentos; não apaga o que já existe)
python run_ingest.py

# Re-ingestão completa (limpa a tabela e insere tudo de novo)
python run_ingest.py --limpar
```

A pasta de dados (`DATA_DIR`) é percorrida recursivamente; cada `.txt` é limpo, chunkado e armazenado com tema derivado do caminho e do título da página.

### Chat web

```bash
python run_web.py
```

Acesse no navegador: **http://127.0.0.1:5000** (ou o host/porta definidos em `FLASK_HOST`/`FLASK_PORT`). Na interface você pode iniciar uma **nova conversa** para limpar o histórico e recomeçar; quando o modelo de chat está configurado, as respostas do Ollama são exibidas em **streaming** (texto aparecendo em tempo real).

### Busca por linha de comando (exemplo)

```bash
python query_example.py
# Digite a pergunta quando solicitado; retorna os trechos mais similares.
```

### Cache de perguntas e avaliação por estrelas

Com `CACHE_PERGUNTAS=1` (padrão), cada pergunta respondida é salva no banco (tabela `jabot_rag_perguntas`). Quando uma pergunta **muito parecida** é feita de novo, a resposta anterior é reutilizada. No chat, o usuário pode avaliar cada resposta com **1 a 5 estrelas**; as respostas com melhor avaliação são priorizadas quando há várias similares no cache. Só são reutilizadas respostas com avaliação média ≥ `CACHE_MIN_STARS` (ex.: 4 = só 4 ou 5 estrelas). No chat, a avaliação é feita pela API `/api/rate`. Respostas vindas do cache podem aparecer com o texto “(Resposta reutilizada de pergunta similar)”. A “similaridade” é definida por `CACHE_THRESHOLD` (distância do embedding; ex.: `0.35`).

## Fluxo resumido

```
manual_txt/*.txt  →  run_ingest / core.ingest (limpeza, chunking, embeddings)  →  PostgreSQL (pgvector)
                                                                                        ↓
Pergunta do usuário  →  run_web / web.app  →  embedding da pergunta  →  busca por similaridade
                                           →  (opcional) cache por pergunta similar  →  resposta reutilizada
                                           →  trechos + Ollama (opcional)  →  resposta em passo a passo
                                                                           →  streaming (SSE) quando LLM ativo
```

## Licença e uso

Projeto para uso interno/documentação do sistema. Ajuste o `.env` e a tabela conforme seu ambiente (JBRJ/jawiki, etc.).
