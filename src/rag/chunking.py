# -*- coding: utf-8 -*-
"""
Divisão das páginas em chunks (etapa 2 do pipeline RAG).

Usa `RecursiveCharacterTextSplitter` com `separators=["\\n\\n", "\\n", ". ", " ", ""]`
e `chunk_overlap` de ~10–15% do `chunk_size`, como pede o guia.

Decisão de projeto (vai para o relatorio_rag.md): o corte é feito PÁGINA A PÁGINA.
  (+) toda citação aponta uma página exata do PDF;
  (-) um trecho que atravessa a quebra de página é cortado ali.
Como cada chunk herda o metadado da sua página, a citação nunca mente.

Cada chunk recebe:
  * `chunk_id`  — determinístico (reindexar não duplica nem embaralha);
  * `secao`     — rótulo da seção numerada ("2.1 Conectar o cabo..."), quando o
                  texto tem cabeçalho numerado; senão, o nome da seção do catálogo.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

SEPARADORES = ["\n\n", "\n", ". ", " ", ""]

# Cabeçalho numerado em UMA linha curta:
#   "2.1 Conectar o cabo..."  (vários níveis)   ou   "1. Visão geral"  (nível único COM ponto).
# Nível único SEM ponto ("3 Apresentação") fica de fora de propósito: no manual,
# linhas de tabela ("5 Sobretensão ...") seriam confundidas com cabeçalho.
_CABECALHO = re.compile(
    r"^\s*(?P<rotulo>(?:\d{1,2}(?:\.\d{1,2})+|\d{1,2}\.))\s+(?P<titulo>[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][^\n]{2,70})$",
    re.MULTILINE,
)


# Linhas de lista numerada quebradas no meio da frase ("2. Reconecte o conector de")
# terminam em palavra funcional; cabeçalhos de verdade não.
_FINAL_DE_FRASE_CORTADA = {"de", "da", "do", "das", "dos", "o", "a", "os", "as", "e", "ou", "para", "com", "em",
                           "no", "na", "nos", "nas", "que", "ao", "por", "um", "uma"}


def _cabecalhos(texto: str, aceitar_nivel_unico: bool) -> list[tuple[int, str]]:
    """Lista (posição, "2.1 Título") dos cabeçalhos numerados do texto da página.

    `aceitar_nivel_unico` ("1. Visão geral") só vale para os documentos fictícios, cujo
    formato controlamos. No manual real, passos numerados ("1. Verifique se o EV...")
    seriam confundidos com cabeçalho; lá só aceitamos x.y (e mais níveis).
    """
    achados = []
    for m in _CABECALHO.finditer(texto):
        if not aceitar_nivel_unico and "." not in m.group("rotulo").rstrip("."):
            continue
        titulo = m.group("titulo").strip()
        if titulo.endswith("."):  # frase de lista numerada, não cabeçalho
            continue
        if titulo.split()[-1].lower() in _FINAL_DE_FRASE_CORTADA:  # frase cortada pela quebra de linha
            continue
        achados.append((m.start(), f"{m.group('rotulo').rstrip('.')} {titulo}"))
    return achados


def dividir_documentos(
    paginas: list[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[Document]:
    """Divide as páginas em chunks com metadados `secao`, `chunk_id` e `start_index`."""
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARADORES,
        add_start_index=True,
    )

    chunks: list[Document] = []
    ultima_secao: dict[str, str] = {}  # seção "em vigor" por documento, entre páginas

    for pagina in paginas:  # as páginas chegam em ordem de documento e de página
        fonte = pagina.metadata["fonte_id"]
        cabecalhos = _cabecalhos(pagina.page_content, pagina.metadata["tipo"] == "ficticio_chargegrid")
        for ordem, chunk in enumerate(divisor.split_documents([pagina])):
            inicio = chunk.metadata.get("start_index", 0)
            # última seção cujo cabeçalho começa antes do chunk; senão a que vinha da página anterior
            vigente = [rotulo for pos, rotulo in cabecalhos if pos <= inicio]
            if vigente:
                ultima_secao[fonte] = vigente[-1]
            # Sem cabeçalho numerado: usa a seção do catálogo — ou, se ela for o
            # documento inteiro, o título do documento (cita melhor que "Documento inteiro").
            base = chunk.metadata["secao_catalogo"]
            if base == "Documento inteiro":
                base = chunk.metadata["titulo"]
            chunk.metadata["secao"] = ultima_secao.get(fonte, base)
            chunk.metadata["chunk_id"] = f"{fonte}-p{chunk.metadata['pagina']:02d}-c{ordem:02d}"
            chunks.append(chunk)
    return chunks
