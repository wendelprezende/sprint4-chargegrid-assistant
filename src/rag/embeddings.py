# -*- coding: utf-8 -*-
"""
Embeddings (etapa 3): `nomic-embed-text` via Ollama Cloud — o único modelo de
embedding aprovado pelo guia.

`EmbeddingsNomic` envolve o `OllamaEmbeddings` mantendo a interface padrão
(`embed_documents` / `embed_query`) e permite, opcionalmente, aplicar os
prefixos de tarefa que o nomic recomenda:
    documentos -> "search_document: <texto>"      consultas -> "search_query: <texto>"
O uso do prefixo é uma flag (`config.USAR_PREFIXO_NOMIC`) justamente para que
ligá-la seja uma ITERAÇÃO MEDIDA na avaliação (T5), e não uma suposição.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from src import config


class EmbeddingsNomic(Embeddings):
    """Wrapper do OllamaEmbeddings com prefixo de tarefa opcional."""

    def __init__(self, base: Embeddings, usar_prefixo: bool = False) -> None:
        self._base = base
        self.usar_prefixo = usar_prefixo

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.usar_prefixo:
            texts = [f"search_document: {t}" for t in texts]
        return self._base.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if self.usar_prefixo:
            text = f"search_query: {text}"
        return self._base.embed_query(text)


def obter_embeddings(usar_prefixo: bool = config.USAR_PREFIXO_NOMIC) -> EmbeddingsNomic:
    """Cria o embedder real (exige OLLAMA_API_KEY no ambiente/.env)."""
    chave = config.obter_api_key()
    base = OllamaEmbeddings(
        model=config.MODELO_EMBEDDING,
        base_url=config.OLLAMA_BASE_URL,
        client_kwargs={"headers": {"Authorization": f"Bearer {chave}"}},
    )
    return EmbeddingsNomic(base, usar_prefixo=usar_prefixo)
