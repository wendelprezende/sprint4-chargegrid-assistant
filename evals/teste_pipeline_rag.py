# -*- coding: utf-8 -*-
"""
Teste funcional do pipeline RAG (T3): load -> split -> embed -> store -> retrieve.

DOIS MODOS
  --offline   Usa um embedding LÉXICO simples (hash de palavras), sem rede e sem
              API key, e um banco temporário. Valida o ENCANAMENTO (carga, metadados,
              chunks, persistência, filtro de público). NÃO mede qualidade semântica:
              os acertos de busca aqui são só uma pista, não uma métrica.
  (padrão)    Usa o nomic-embed-text REAL via Ollama Cloud e grava o índice
              persistente em chroma_db/. É o teste que VOCÊ roda na sua máquina.

USO (na raiz do projeto, ambiente virtual ativo):
    python -m evals.teste_pipeline_rag --offline
    python -m evals.teste_pipeline_rag
    python -m evals.teste_pipeline_rag --chunk-size 500 --overlap 75

Saída: lista de checagens [OK]/[FALHA] e uma tabela de busca com a distância.
Código de saída 1 se alguma checagem estrutural falhar.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import sys
import tempfile
import unicodedata
from pathlib import Path

from langchain_core.embeddings import Embeddings

from src import config
from src.rag.chunking import dividir_documentos
from src.rag.indexar import indexar
from src.rag.loader import carregar_documentos
from src.rag.retriever import Retriever
from src.rag.vector_store import abrir_indice

# (consulta, fontes aceitas) — fontes aceitas = ids de documento onde a resposta está.
# O sufixo ":p21" restringe também a página.
CONSULTAS = [
    ("Como faço para pagar com PIX?", {"CG-02"}),
    ("Por que o preço está mais caro agora do que ontem de manhã?", {"CG-02", "CG-04"}),
    ("O cabo não está encaixando no meu carro", {"CG-03"}),
    ("Como encerro a sessão e pego o comprovante?", {"CG-01", "CG-02"}),
    ("O que significa a luz azul do carregador?", {"GW-HCAG2-MANUAL:p21"}),
    ("Qual é a potência do carregador de 22 kW?", {"GW-HCAG2-DATASHEET"}),
    ("A tela do totem está preta e não responde", {"CG-03"}),
    ("Posso processar a empresa se meu carro for danificado?", {"CG-05", "CG-04"}),
    ("Posso mexer na fiação da minha casa para instalar um carregador?", {"CG-03", "CG-04", "CG-05"}),
    ("Qual é a capital da França?", set()),  # sem resposta esperada: só observamos a distância
]

_STOP = {"o", "a", "os", "as", "um", "uma", "de", "do", "da", "dos", "das", "e", "em", "no", "na", "nos",
         "nas", "para", "por", "com", "que", "se", "meu", "minha", "eu", "ao", "como", "qual", "quando"}


def _normalizar(texto: str) -> list[str]:
    sem_acento = unicodedata.normalize("NFD", texto.lower())
    sem_acento = "".join(c for c in sem_acento if unicodedata.category(c) != "Mn")
    return [t for t in re.findall(r"[a-z0-9]+", sem_acento) if t not in _STOP and len(t) > 1]


class EmbeddingsLexicaisOffline(Embeddings):
    """Bag-of-words hasheado (512 dims). Só para testar o encanamento sem rede."""

    DIM = 512

    def _vetor(self, texto: str) -> list[float]:
        v = [0.0] * self.DIM
        for token in _normalizar(texto):
            idx = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.DIM
            v[idx] += 1.0
        norma = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norma for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vetor(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vetor(text)


class Checagens:
    def __init__(self) -> None:
        self.falhas = 0

    def ok(self, condicao: bool, descricao: str) -> None:
        print(f"  [{'OK' if condicao else 'FALHA'}] {descricao}")
        if not condicao:
            self.falhas += 1


def _acertou(resultados, aceitas: set[str]) -> bool:
    for r in resultados:
        meta = r.documento.metadata
        for a in aceitas:
            fonte, _, pag = a.partition(":p")
            if meta["fonte_id"] == fonte and (not pag or meta["pagina"] == int(pag)):
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    ap.add_argument("--k", type=int, default=config.TOP_K)
    args = ap.parse_args()
    c = Checagens()
    modo = "OFFLINE (embedding léxico, banco temporário)" if args.offline else "REAL (nomic-embed-text, chroma_db/)"
    print(f"=== Teste do pipeline RAG — modo {modo} ===")

    # ---- 1. Carga -------------------------------------------------------
    print("\n1) Carga dos documentos")
    paginas, carga = carregar_documentos()
    c.ok(carga.paginas_indexadas > 0, f"{carga.paginas_indexadas} páginas indexadas")
    manual_presente = "GW_HCAG2_UserManualPT.pdf" not in carga.arquivos_ausentes
    if manual_presente:
        c.ok(carga.paginas_excluidas == 4, f"capa/direitos/índice excluídos pelo catálogo ({carga.paginas_excluidas} páginas)")
    else:
        print("  [INFO] manual ausente (no .gitignore): testes que dependem dele serão pulados")
    c.ok(all(p.metadata["publico"] != "excluir" for p in paginas), "nenhuma página 'excluir' foi carregada")
    c.ok(all(isinstance(p.metadata["pagina"], int) and p.metadata["pagina"] >= 1 for p in paginas),
         "páginas numeradas a partir de 1")

    # ---- 2. Chunking ----------------------------------------------------
    print("\n2) Chunking")
    chunks = dividir_documentos(paginas, args.chunk_size, args.overlap)
    ids = [ch.metadata["chunk_id"] for ch in chunks]
    c.ok(len(chunks) > 0, f"{len(chunks)} chunks (chunk_size={args.chunk_size}, overlap={args.overlap})")
    c.ok(len(ids) == len(set(ids)), "chunk_id únicos")
    c.ok(max(len(ch.page_content) for ch in chunks) <= args.chunk_size, "nenhum chunk excede o chunk_size")
    obrigatorios = {"fonte_id", "titulo", "tipo", "publico", "pagina", "secao", "chunk_id"}
    c.ok(all(obrigatorios <= set(ch.metadata) for ch in chunks), "todos os chunks têm os metadados obrigatórios")
    cg2 = [ch for ch in chunks if ch.metadata["fonte_id"] == "CG-02"]
    c.ok(any(ch.metadata["secao"].startswith("2") for ch in cg2), "seção numerada detectada em CG-02 (ex.: '2. Tabela tarifária...')")
    print("  exemplos de rótulo de seção detectado:")
    vistos = set()
    for ch in chunks:
        chave = (ch.metadata["fonte_id"], ch.metadata["secao"])
        if chave not in vistos and len(vistos) < 8 and ch.metadata["fonte_id"] in ("CG-01", "CG-03", "GW-HCAG2-MANUAL"):
            vistos.add(chave)
            print(f"    {ch.metadata['fonte_id']:16} p.{ch.metadata['pagina']:<3} -> {ch.metadata['secao']}")

    # ---- 3. Indexação + persistência -----------------------------------
    print("\n3) Indexação e persistência")
    if args.offline:
        embeddings = EmbeddingsLexicaisOffline()
        pasta = Path(tempfile.mkdtemp(prefix="chroma_offline_"))
    else:
        from src.rag.embeddings import obter_embeddings

        embeddings = obter_embeddings()
        pasta = config.PASTA_CHROMA
    stats = indexar(args.chunk_size, args.overlap, embeddings=embeddings, pasta_chroma=pasta)
    c.ok(stats["chunks_no_banco"] == stats["chunks_total"], f"{stats['chunks_no_banco']} chunks gravados no Chroma "
         f"em {stats['segundos_vetorizacao']} s")
    loja = abrir_indice(embeddings, pasta=pasta)  # NOVA instância: prova que persistiu em disco
    c.ok(loja._collection.count() == len(chunks), "índice reaberto do disco tem a mesma contagem")  # noqa: SLF001

    # ---- 4. Filtro de público ------------------------------------------
    print("\n4) Filtro por público (proteção de escopo)")
    pergunta_instalacao = "como instalar o carregador na parede e ligar o cabo de aterramento"
    padrao = Retriever(loja, k=args.k).buscar(pergunta_instalacao)
    c.ok(all(r.documento.metadata["publico"] in ("motorista", "geral") for r in padrao),
         "busca padrão nunca devolve conteúdo 'instalador'")
    if manual_presente:
        so_instalador = Retriever(loja, k=args.k, publicos=("instalador",)).buscar(pergunta_instalacao)
        c.ok(len(so_instalador) > 0 and all(r.documento.metadata["publico"] == "instalador" for r in so_instalador),
             "com publicos=('instalador',) o conteúdo de instalação é recuperável (o filtro é a única barreira)")

    # ---- 5. Busca -------------------------------------------------------
    print(f"\n5) Busca (top-{args.k}) — distância de cosseno: menor = mais parecido")
    ret = Retriever(loja, k=args.k)
    acertos = esperados = 0
    for consulta, aceitas in CONSULTAS:
        res = ret.buscar(consulta)
        marca = "  -  "
        if aceitas:
            esperados += 1
            hit = _acertou(res, aceitas)
            acertos += hit
            marca = " HIT " if hit else "MISS "
        topo = res[0] if res else None
        print(f"  [{marca}] d_min={topo.distancia:.3f} | {consulta[:58]:58} -> "
              f"{', '.join(r.citacao.split(' · seção')[0] + ':p' + str(r.documento.metadata['pagina']) for r in res[:3])}")
    print(f"\n  acertos: {acertos}/{esperados} consultas com fonte esperada no top-{args.k}")
    if args.offline:
        print("  (modo offline: o embedding é léxico — este número NÃO representa a qualidade do nomic-embed-text)")
    print("  A linha 'capital da França' (sem fonte esperada) mostra a distância de uma pergunta fora do escopo:")
    print("  na T5 usaremos esse tipo de valor para calibrar o limiar de 'não encontrei na base'.")

    print(f"\n=== Resultado: {'TODAS as checagens estruturais passaram' if c.falhas == 0 else str(c.falhas) + ' checagem(ns) FALHARAM'} ===")
    return 1 if c.falhas else 0


if __name__ == "__main__":
    sys.exit(main())
