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


MIN_CHARS_BLOCO = 120  # bloco de seção menor que isso (ex.: só o título "2. Como iniciar...") é fundido ao seguinte


def _blocos_por_secao(texto: str) -> list[tuple[str | None, str]]:
    """Quebra o texto da página nos cabeçalhos numerados: [(rótulo_da_seção | None, texto_do_bloco)]."""
    cabs = _cabecalhos(texto, True)
    if not cabs:
        return [(None, texto)]
    limites = [pos for pos, _ in cabs]
    blocos: list[tuple[str | None, str]] = []
    if limites[0] > 0 and texto[: limites[0]].strip():
        blocos.append((None, texto[: limites[0]].strip()))  # título do documento ou continuação da página anterior
    for i, (pos, rotulo) in enumerate(cabs):
        fim = limites[i + 1] if i + 1 < len(limites) else len(texto)
        blocos.append((rotulo, texto[pos:fim].strip()))
    # funde blocos minúsculos com o seguinte (o rótulo passa a ser o do bloco mais específico)
    fundidos: list[tuple[str | None, str]] = []
    pendente: tuple[str | None, str] | None = None
    for rotulo, corpo in blocos:
        if pendente is not None:
            rotulo, corpo = rotulo or pendente[0], pendente[1] + "\n" + corpo
            pendente = None
        if len(corpo) < MIN_CHARS_BLOCO:
            pendente = (rotulo, corpo)
        else:
            fundidos.append((rotulo, corpo))
    if pendente is not None:
        if fundidos:
            fundidos[-1] = (fundidos[-1][0], fundidos[-1][1] + "\n" + pendente[1])
        else:
            fundidos.append(pendente)
    return fundidos


def _dividir_por_secao(paginas: list[Document], divisor: RecursiveCharacterTextSplitter) -> list[Document]:
    """Documentos FICTÍCIOS: um chunk por seção numerada (um assunto por chunk). Os reais seguem por tamanho."""
    chunks: list[Document] = []
    ultima: dict[str, str] = {}
    for pagina in paginas:
        fonte = pagina.metadata["fonte_id"]
        if pagina.metadata["tipo"] != "ficticio_chargegrid":
            chunks.extend(_dividir_por_tamanho([pagina], divisor, ultima))
            continue
        ordem = 0
        for rotulo, corpo in _blocos_por_secao(pagina.page_content):
            if rotulo:
                ultima[fonte] = rotulo
            base = pagina.metadata["titulo"] if pagina.metadata["secao_catalogo"] == "Documento inteiro" else pagina.metadata["secao_catalogo"]
            partes = [corpo] if len(corpo) <= divisor._chunk_size else divisor.split_text(corpo)
            for parte in partes:
                meta = dict(pagina.metadata)
                meta["secao"] = rotulo or ultima.get(fonte, base)
                meta["chunk_id"] = f"{fonte}-p{meta['pagina']:02d}-c{ordem:02d}"
                chunks.append(Document(page_content=parte, metadata=meta))
                ordem += 1
    return chunks


def dividir_documentos(
    paginas: list[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    por_secao: bool = config.CHUNKING_POR_SECAO,
) -> list[Document]:
    """Divide as páginas em chunks com metadados `secao` e `chunk_id`.

    `por_secao=True` (iteração da T5): documentos fictícios viram um chunk por seção numerada.
    """
    divisor = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARADORES,
        add_start_index=True,
    )

    if por_secao:
        return _dividir_por_secao(paginas, divisor)
    return _dividir_por_tamanho(paginas, divisor, {})


def _dividir_por_tamanho(paginas: list[Document], divisor: RecursiveCharacterTextSplitter, ultima_secao: dict[str, str]) -> list[Document]:
    chunks: list[Document] = []
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
