# -*- coding: utf-8 -*-
"""
Prompt RAG (T4): carrega o prompt VERSIONADO de `prompts/prompt_rag_<versao>.md` e monta o
contexto com marcação XML (context engineering da Aula 04).

Mensagem enviada ao modelo:
    system : prompt versionado + instruções de formato do schema `RespostaRAG`
    human  : <contexto> <dados_sessao> <historico> <pergunta>   (tudo tratado como DADO)

Todo texto de origem externa (chunks, dados da sessão, histórico, pergunta) passa por
`neutralizar_texto` antes de entrar nas tags: ele não consegue fechar nem abrir tags do prompt.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

from langchain_core.prompts import ChatPromptTemplate

from src import config
from src.guardrails.documentos import neutralizar_texto

MARCADOR_INICIO = "<!-- INICIO_DO_PROMPT -->"
MARCADOR_CANARIO = "{{CANARIO}}"

_HUMANO = """<contexto>
{contexto}
</contexto>

<dados_sessao>
{dados_sessao}
</dados_sessao>

<historico>
{historico}
</historico>

<pergunta>
{pergunta}
</pergunta>"""


def carregar_prompt_sistema(versao: str = config.VERSAO_PROMPT_RAG, canario: str = "CANARIO-INDEFINIDO") -> str:
    """Lê `prompts/prompt_rag_<versao>.md`, descarta a documentação e injeta o canário."""
    caminho = Path(config.PASTA_PROMPTS) / f"prompt_rag_{versao}.md"
    if not caminho.exists():
        raise FileNotFoundError(f"Prompt RAG '{versao}' não encontrado em {caminho}")
    texto = caminho.read_text(encoding="utf-8")
    if MARCADOR_INICIO not in texto:
        raise ValueError(f"{caminho.name} não contém o marcador {MARCADOR_INICIO}")
    return texto.split(MARCADOR_INICIO, 1)[1].strip().replace(MARCADOR_CANARIO, canario)


def montar_prompt() -> ChatPromptTemplate:
    """Template com variáveis: sistema, formato, contexto, dados_sessao, historico, pergunta."""
    return ChatPromptTemplate.from_messages([("system", "{sistema}\n\n{formato}"), ("human", _HUMANO)])


def _limpar(texto: str) -> str:
    texto = re.sub(r"[ \t]+", " ", texto or "")
    return re.sub(r"\n\s*\n+", "\n", texto).strip()


def _atributo(valor: Any) -> str:
    return neutralizar_texto(str(valor)).replace('"', "'")


def formatar_contexto(resultados: Iterable[Any]) -> str:
    """Um bloco <documento id=... fonte=... tipo=... pagina=... secao=...> por chunk (já sanitizado)."""
    blocos = []
    for r in resultados:
        meta = r.documento.metadata
        blocos.append(
            f'<documento id="{_atributo(meta["chunk_id"])}" fonte="{_atributo(meta["fonte_id"])}" '
            f'tipo="{_atributo(meta["tipo"])}" pagina="{_atributo(meta["pagina"])}" '
            f'secao="{_atributo(meta.get("secao", ""))}">\n{neutralizar_texto(_limpar(r.documento.page_content))}\n</documento>'
        )
    return "\n".join(blocos) if blocos else "(nenhum documento relevante foi recuperado)"


def formatar_dados_sessao(dados: dict | None) -> str:
    if not dados:
        return "(indisponível: não há dados da sessão em andamento)"
    return "\n".join(f"{_atributo(k)}: {neutralizar_texto(str(v))}" for k, v in dados.items())


def formatar_historico(historico: list[tuple[str, str]] | None, max_turnos: int = 3, max_chars: int = 300) -> str:
    if not historico:
        return "(sem histórico)"
    linhas = []
    for pergunta, resposta in historico[-max_turnos:]:
        linhas.append(f"motorista: {neutralizar_texto(_limpar(pergunta))[:max_chars]}")
        linhas.append(f"assistente: {neutralizar_texto(_limpar(resposta))[:max_chars]}")
    return "\n".join(linhas)
