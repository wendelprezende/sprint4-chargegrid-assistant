# -*- coding: utf-8 -*-
"""
Embeddings (etapa 3): `nomic-embed-text` — o único modelo de embedding aprovado
pelo guia. Por padrão roda em um servidor Ollama LOCAL (o Ollama Cloud da nossa
conta não oferece modelos de embedding; veja `config.OLLAMA_EMBED_BASE_URL`).

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
    """Cria o embedder e já testa a conexão, para falhar cedo e com instrução clara."""
    base_url = config.OLLAMA_EMBED_BASE_URL
    base = OllamaEmbeddings(
        model=config.MODELO_EMBEDDING,
        base_url=base_url,
        client_kwargs={"headers": config.cabecalhos_ollama(base_url)},
    )
    try:
        base.embed_query("ping")
    except Exception as erro:  # noqa: BLE001
        raise RuntimeError(
            f"Não consegui gerar embeddings em {base_url} com '{config.MODELO_EMBEDDING}' "
            f"({type(erro).__name__}: {erro}).\n"
            "Se for o servidor local: 1) abra o aplicativo Ollama (ou rode `ollama serve`); "
            f"2) rode `ollama pull {config.MODELO_EMBEDDING}`; 3) confira com `ollama list`."
        ) from erro
    return EmbeddingsNomic(base, usar_prefixo=usar_prefixo)
