# -*- coding: utf-8 -*-
"""
Banco vetorial (etapa 4): ChromaDB PERSISTENTE em disco (`chroma_db/`).

  * distância de COSSENO (a mais adequada a embeddings de texto): quanto MENOR a
    distância, mais parecido. 0 = idêntico; ~1 = sem relação.
  * ids determinísticos (`chunk_id`): reindexar sobrescreve em vez de duplicar.
  * `recriar=True` apaga a coleção antes de indexar, garantindo que o banco
    reflita exatamente os documentos/parâmetros da execução atual.
"""

from __future__ import annotations

from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src import config

_CONFIG_COSSENO = {"hnsw": {"space": "cosine"}}


def _abrir(embeddings: Embeddings, pasta: Path, colecao: str) -> Chroma:
    return Chroma(
        collection_name=colecao,
        embedding_function=embeddings,
        persist_directory=str(pasta),
        collection_configuration=_CONFIG_COSSENO,
    )


def construir_indice(
    chunks: list[Document],
    embeddings: Embeddings,
    pasta: Path = config.PASTA_CHROMA,
    colecao: str = config.COLECAO_CHROMA,
    recriar: bool = True,
    tamanho_lote: int = 32,
) -> Chroma:
    """Vetoriza os chunks e grava no Chroma persistente."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    if recriar:
        cliente = chromadb.PersistentClient(path=str(pasta))
        try:
            cliente.delete_collection(colecao)
        except Exception:  # noqa: BLE001 — coleção ainda não existia
            pass

    loja = _abrir(embeddings, pasta, colecao)
    # Em lotes: uma chamada de embedding por lote (evita payloads enormes e
    # facilita identificar em qual lote um erro de rede/limite ocorreu).
    for inicio in range(0, len(chunks), tamanho_lote):
        lote = chunks[inicio : inicio + tamanho_lote]
        loja.add_documents(lote, ids=[c.metadata["chunk_id"] for c in lote])
    return loja


def abrir_indice(
    embeddings: Embeddings,
    pasta: Path = config.PASTA_CHROMA,
    colecao: str = config.COLECAO_CHROMA,
) -> Chroma:
    """Abre o índice já persistido; falha com instrução clara se estiver vazio."""
    pasta = Path(pasta)
    if not pasta.exists():
        raise FileNotFoundError(
            f"Índice não encontrado em {pasta}. Rode antes: python -m src.rag.indexar"
        )
    loja = _abrir(embeddings, pasta, colecao)
    if loja._collection.count() == 0:  # noqa: SLF001
        raise RuntimeError(
            f"A coleção '{colecao}' está vazia. Rode antes: python -m src.rag.indexar"
        )
    return loja
