# -*- coding: utf-8 -*-
"""
Recuperação (etapa 5): busca semântica com FILTRO DE METADADOS.

Por padrão só recupera chunks cujo `publico` está em `config.PUBLICOS_MOTORISTA`
("motorista" e "geral"): conteúdo de instalação do manual (`instalador`) NUNCA
chega ao motorista — é uma proteção de escopo aplicada ANTES do LLM.

Cada resultado traz a distância de cosseno e a citação pronta
("CG-02 · seção 2 Tabela tarifária · p.1"), que o prompt RAG (T4) e a interface (T7)
reaproveitam.

Teste rápido pela linha de comando:
    python -m src.rag.retriever "Como pago com PIX?" --k 4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src import config


def formatar_citacao(meta: dict) -> str:
    """Rótulo de fonte exibido ao usuário: documento · seção · página."""
    return f"{meta['fonte_id']} · seção {meta.get('secao', meta.get('secao_catalogo', '?'))} · p.{meta['pagina']}"


@dataclass(frozen=True)
class ResultadoBusca:
    documento: Document
    distancia: float  # cosseno: menor = mais parecido

    @property
    def citacao(self) -> str:
        return formatar_citacao(self.documento.metadata)


def _montar_filtro(publicos: tuple[str, ...] | None, tipos: tuple[str, ...] | None) -> dict | None:
    condicoes = []
    if publicos:
        condicoes.append({"publico": {"$in": list(publicos)}})
    if tipos:
        condicoes.append({"tipo": {"$in": list(tipos)}})
    if not condicoes:
        return None
    return condicoes[0] if len(condicoes) == 1 else {"$and": condicoes}


class Retriever:
    """Busca semântica sobre o índice, com filtros e limiar de distância opcionais."""

    def __init__(
        self,
        loja: Chroma,
        k: int = config.TOP_K,
        publicos: tuple[str, ...] | None = config.PUBLICOS_MOTORISTA,
        tipos: tuple[str, ...] | None = None,
        limiar_distancia: float | None = None,
        cotas: dict[str, int] | None = None,
    ) -> None:
        self.loja = loja
        self.cotas = cotas  # ex.: {"ficticio_chargegrid": 3, "real_goodwe": 2}: busca por camada (T5)
        self.k = k
        self.publicos = publicos
        self.tipos = tipos
        self.limiar_distancia = limiar_distancia

    def buscar(self, consulta: str) -> list[ResultadoBusca]:
        if self.cotas:
            # busca por camada: reserva vagas para cada tipo de documento (o manual real tem 78% dos chunks e,
            # sem cota, espreme as respostas da operação do totem). Resultado final ordenado por distância.
            pares = []
            for tipo, n in self.cotas.items():
                pares += self.loja.similarity_search_with_score(
                    consulta, k=n, filter=_montar_filtro(self.publicos, (tipo,))
                )
            pares.sort(key=lambda p: p[1])
        else:
            pares = self.loja.similarity_search_with_score(
                consulta, k=self.k, filter=_montar_filtro(self.publicos, self.tipos)
            )
        resultados = [ResultadoBusca(doc, float(dist)) for doc, dist in pares]
        if self.limiar_distancia is not None:
            resultados = [r for r in resultados if r.distancia <= self.limiar_distancia]
        return resultados


@lru_cache(maxsize=1)
def _retriever_padrao() -> Retriever:
    from src.rag.embeddings import obter_embeddings
    from src.rag.vector_store import abrir_indice

    return Retriever(abrir_indice(obter_embeddings()))


def buscar(consulta: str) -> list[ResultadoBusca]:
    """Atalho `buscar(consulta)` com o índice persistido (reutilizável como tool no CKP03)."""
    return _retriever_padrao().buscar(consulta)


def _main() -> None:
    ap = argparse.ArgumentParser(description="Consulta o índice RAG e mostra os chunks recuperados.")
    ap.add_argument("consulta")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--publico", nargs="*", default=list(config.PUBLICOS_MOTORISTA),
                    help="públicos permitidos (motorista geral instalador); vazio = sem filtro")
    args = ap.parse_args()

    from src.rag.embeddings import obter_embeddings
    from src.rag.vector_store import abrir_indice

    ret = Retriever(abrir_indice(obter_embeddings()), k=args.k, publicos=tuple(args.publico) or None)
    for i, r in enumerate(ret.buscar(args.consulta), 1):
        trecho = " ".join(r.documento.page_content.split())[:220]
        print(f"\n#{i}  distância={r.distancia:.4f}  [{r.citacao}]  ({r.documento.metadata['tipo']}/{r.documento.metadata['publico']})\n    {trecho}...")


if __name__ == "__main__":
    _main()
