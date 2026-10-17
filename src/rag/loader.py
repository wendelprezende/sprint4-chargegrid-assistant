# -*- coding: utf-8 -*-
"""
Carregamento dos documentos da base de conhecimento (etapa 1 do pipeline RAG).

Lê o `catalogo.json` e, para cada PDF, usa o `PyMuPDFLoader` (exigido pelo guia
da Sprint 4). Cada PÁGINA vira um `Document` com metadados LIMPOS, que depois
acompanham cada chunk até a citação de fonte:

    fonte_id, titulo, arquivo, tipo, publico, secao_catalogo, pagina, versao, origem

`pagina` é base 1 (a mesma numeração do PDF que o usuário vê).

Regras de indexação (vindas do catálogo):
  * página em seção com publico == "excluir"  -> não é indexada (capa, índice);
  * página fora de qualquer seção do catálogo   -> não é indexada;
  * PDF ausente na pasta (ex.: manual no .gitignore) -> aviso, sem derrubar o pipeline.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document

from src import config


@dataclass
class RelatorioCarga:
    """Resumo do que foi (e do que NÃO foi) carregado — impresso pela indexação."""

    paginas_indexadas: int = 0
    paginas_excluidas: int = 0
    paginas_sem_secao: int = 0
    arquivos_ausentes: list[str] = field(default_factory=list)
    por_documento: dict[str, int] = field(default_factory=dict)


def carregar_catalogo(pasta: Path = config.PASTA_KB) -> dict:
    """Lê `catalogo.json` da pasta da base."""
    with open(pasta / "catalogo.json", encoding="utf-8") as arquivo:
        return json.load(arquivo)


def _secao_da_pagina(doc_catalogo: dict, pagina: int) -> dict | None:
    """Devolve a seção do catálogo que cobre a página (base 1), ou None."""
    for secao in doc_catalogo.get("secoes", []):
        inicio, fim = secao["paginas"]
        if inicio <= pagina <= fim:
            return secao
    return None


def carregar_documentos(
    pasta: Path = config.PASTA_KB,
) -> tuple[list[Document], RelatorioCarga]:
    """Carrega todas as páginas indexáveis, com metadados do catálogo."""
    catalogo = carregar_catalogo(pasta)
    relatorio = RelatorioCarga()
    paginas: list[Document] = []

    for doc_cat in catalogo["documentos"]:
        caminho = pasta / doc_cat["arquivo"]
        if not caminho.exists():
            relatorio.arquivos_ausentes.append(doc_cat["arquivo"])
            warnings.warn(f"PDF ausente, ignorado: {doc_cat['arquivo']}", stacklevel=2)
            continue

        for pagina_pdf in PyMuPDFLoader(str(caminho)).load():
            numero = int(pagina_pdf.metadata["page"]) + 1  # PyMuPDFLoader é base 0
            secao = _secao_da_pagina(doc_cat, numero)
            if secao is None:
                relatorio.paginas_sem_secao += 1
                continue
            if secao["publico"] == "excluir":
                relatorio.paginas_excluidas += 1
                continue

            paginas.append(
                Document(
                    page_content=pagina_pdf.page_content,
                    # Somente tipos primitivos: é o que o Chroma aceita em metadados.
                    metadata={
                        "fonte_id": doc_cat["id"],
                        "titulo": doc_cat["titulo"],
                        "arquivo": doc_cat["arquivo"],
                        "tipo": doc_cat["tipo"],
                        "publico": secao["publico"],
                        "secao_catalogo": secao["nome"],
                        "pagina": numero,
                        "versao": doc_cat.get("versao", ""),
                        "origem": doc_cat.get("origem", ""),
                    },
                )
            )
            relatorio.paginas_indexadas += 1
            relatorio.por_documento[doc_cat["id"]] = relatorio.por_documento.get(doc_cat["id"], 0) + 1

    return paginas, relatorio
