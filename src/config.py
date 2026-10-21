# -*- coding: utf-8 -*-
"""
Configuração central do projeto (Sprint 4).

Todo caminho, nome de modelo e parâmetro "mágico" vive aqui, para que os
módulos de `src/rag/`, a interface (`app/`) e os evals (`evals/`) leiam
a mesma fonte de verdade — e para que cada mudança de parâmetro fique
registrada em um único lugar (importante para o relatório de modelos).

A API key NUNCA é escrita aqui: ela vem da variável de ambiente
`OLLAMA_API_KEY` (carregada do arquivo `.env`, que está no .gitignore).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Raiz do projeto = pasta que contém `src/`
RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")

# --- Pastas ---
PASTA_KB = RAIZ / "data" / "knowledge_base"
PASTA_CHROMA = RAIZ / "chroma_db"
PASTA_PROMPTS = RAIZ / "prompts"
PASTA_EVALS = RAIZ / "evals"
PASTA_DOCS = RAIZ / "docs"

# --- Ollama Cloud ---
# CHAT: sempre no Ollama Cloud.
OLLAMA_BASE_URL = "https://ollama.com"
# EMBEDDING: o diagnóstico mostrou que a conta NÃO enxerga modelos de embedding no
# Ollama Cloud (todo /api/embed devolve 401). O MESMO modelo `nomic-embed-text` roda,
# então, em um servidor Ollama LOCAL. Para voltar à nuvem (se um dia funcionar):
#   defina OLLAMA_EMBED_BASE_URL=https://ollama.com no .env.
OLLAMA_EMBED_BASE_URL = os.environ.get("OLLAMA_EMBED_BASE_URL", "http://localhost:11434")
MODELO_EMBEDDING = "nomic-embed-text"  # único modelo de embedding aprovado
MODELO_CHAT_PRINCIPAL = "gpt-oss:120b"
MODELO_CHAT_COMPARACAO = "gemma4:31b-cloud"  # validado pelo diagnóstico real; qwen3:8b não existe na nuvem

# --- Parâmetros do RAG (valores INICIAIS; cada mudança vira uma iteração
#     registrada em evals/ e no relatório de modelos) ---
COLECAO_CHROMA = "chargegrid_kb"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120  # ~15% do chunk_size
TOP_K = 4
USAR_PREFIXO_NOMIC = False  # nomic-embed-text recomenda "search_document: "/"search_query: ";
#                            começa desligado (baseline) e vira uma iteração medida na T5.
PUBLICOS_MOTORISTA = ("motorista", "geral")  # públicos que podem embasar respostas ao motorista

# --- Variantes do pipeline (T5): cada uma é uma ITERAÇÃO medida em evals/. Os padrões abaixo são o BASELINE;
#     depois da avaliação, o padrão passa a ser a combinação vencedora. ---
LIMPAR_BOILERPLATE = False      # remove o aviso/rodapé repetido dos documentos fictícios antes do chunking
CHUNKING_POR_SECAO = False      # documentos fictícios: um chunk por seção numerada (em vez de corte por tamanho)
COTAS_POR_TIPO = None           # ex.: {"ficticio_chargegrid": 3, "real_goodwe": 2} -> busca por camada

# --- Geração com RAG (T4) ---
VERSAO_PROMPT_RAG = "v2"        # arquivo prompts/prompt_rag_<versao>.md (v1 = baseline, v2 = segurança reforçada)
LIMIAR_DISTANCIA = 0.40         # "porta dura": se o melhor chunk estiver mais longe, o LLM nem é chamado.
#                                 Valor INICIAL (dentro do escopo: 0,236–0,353; fora: 0,416). Recalibrar na T5.
NUM_PREDICT_RAG = 800           # o gpt-oss gasta tokens "pensando": 400 pode zerar a resposta
REASONING_GPT_OSS = "low"       # esforço de raciocínio do gpt-oss (None desliga); só vale para modelos gpt-oss
MAX_TENTATIVAS_FORMATO = 2      # tentativas de correção quando o JSON do LLM vem inválido
TEMPERATURE = 0.0  # recomendado pelo guia da Sprint 4 (respostas fundamentadas)
TOP_P = 0.9
NUM_PREDICT = 400  # equivalente ao max_tokens


def obter_api_key() -> str:
    """Lê a OLLAMA_API_KEY do ambiente; falha com mensagem clara se ausente."""
    chave = os.environ.get("OLLAMA_API_KEY", "").strip()
    if not chave or chave == "coloque_sua_api_key_aqui":
        raise RuntimeError(
            "OLLAMA_API_KEY não configurada. Copie .env.example para .env e "
            "cole sua chave (https://ollama.com/settings/keys)."
        )
    return chave


def cabecalhos_ollama(base_url: str) -> dict:
    """Cabeçalho de autenticação: só o Ollama Cloud exige a chave; o servidor local não."""
    if "ollama.com" in base_url:
        return {"Authorization": f"Bearer {obter_api_key()}"}
    return {}
