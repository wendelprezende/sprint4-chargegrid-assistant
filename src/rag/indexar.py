# -*- coding: utf-8 -*-
"""
Indexação completa: carrega -> divide -> vetoriza -> grava no Chroma.

    python -m src.rag.indexar                       # parâmetros de src/config.py
    python -m src.rag.indexar --chunk-size 500 --overlap 75
    python -m src.rag.indexar --prefixo-nomic       # liga o prefixo search_document/search_query

Cada execução RECRIA o banco (`chroma_db/`), então ele sempre reflete os
documentos e parâmetros desta execução. Os parâmetros ficam registrados no
arquivo `chroma_db/parametros_indexacao.json` — o eval (T5) lê esse arquivo
para saber com qual configuração cada score foi medido.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from langchain_core.embeddings import Embeddings

from src import config
from src.rag.chunking import dividir_documentos
from src.rag.loader import carregar_documentos
from src.rag.vector_store import construir_indice


def indexar(
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    embeddings: Embeddings | None = None,
    usar_prefixo: bool = config.USAR_PREFIXO_NOMIC,
    pasta_chroma: Path = config.PASTA_CHROMA,
) -> dict:
    """Executa a indexação e devolve estatísticas (usado também pelos testes)."""
    paginas, carga = carregar_documentos()
    chunks = dividir_documentos(paginas, chunk_size, chunk_overlap)
    if embeddings is None:
        from src.rag.embeddings import obter_embeddings

        embeddings = obter_embeddings(usar_prefixo)

    inicio = time.perf_counter()
    loja = construir_indice(chunks, embeddings, pasta=pasta_chroma)
    segundos = time.perf_counter() - inicio

    tamanhos = [len(c.page_content) for c in chunks]
    por_doc: dict[str, int] = {}
    for c in chunks:
        por_doc[c.metadata["fonte_id"]] = por_doc.get(c.metadata["fonte_id"], 0) + 1
    stats = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "usar_prefixo_nomic": usar_prefixo,
        "modelo_embedding": config.MODELO_EMBEDDING,
        "paginas_indexadas": carga.paginas_indexadas,
        "paginas_excluidas": carga.paginas_excluidas,
        "arquivos_ausentes": carga.arquivos_ausentes,
        "chunks_total": len(chunks),
        "chunks_no_banco": loja._collection.count(),  # noqa: SLF001
        "chunks_por_documento": por_doc,
        "tamanho_chunk_min_mediana_max": [min(tamanhos), int(statistics.median(tamanhos)), max(tamanhos)],
        "segundos_vetorizacao": round(segundos, 2),
    }
    Path(pasta_chroma).mkdir(parents=True, exist_ok=True)
    (Path(pasta_chroma) / "parametros_indexacao.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return stats


def _imprimir(stats: dict) -> None:
    print("\n=== Indexação concluída ===")
    print(f"  chunk_size={stats['chunk_size']}  overlap={stats['chunk_overlap']}  "
          f"prefixo_nomic={stats['usar_prefixo_nomic']}  modelo={stats['modelo_embedding']}")
    print(f"  páginas indexadas: {stats['paginas_indexadas']}  (excluídas pelo catálogo: {stats['paginas_excluidas']})")
    if stats["arquivos_ausentes"]:
        print(f"  ⚠ PDFs ausentes (ignorados): {', '.join(stats['arquivos_ausentes'])}")
    print(f"  chunks: {stats['chunks_total']} gerados | {stats['chunks_no_banco']} no banco")
    mn, md, mx = stats["tamanho_chunk_min_mediana_max"]
    print(f"  tamanho do chunk (caracteres): mín {mn} | mediana {md} | máx {mx}")
    for doc, n in stats["chunks_por_documento"].items():
        print(f"    {doc:22} {n:4d} chunks")
    print(f"  tempo de vetorização: {stats['segundos_vetorizacao']} s")


def _main() -> None:
    ap = argparse.ArgumentParser(description="Indexa a base de conhecimento no ChromaDB.")
    ap.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    ap.add_argument("--prefixo-nomic", action="store_true", default=config.USAR_PREFIXO_NOMIC)
    args = ap.parse_args()
    _imprimir(indexar(args.chunk_size, args.overlap, usar_prefixo=args.prefixo_nomic))


if __name__ == "__main__":
    _main()
