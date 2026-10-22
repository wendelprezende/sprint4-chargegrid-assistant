# -*- coding: utf-8 -*-
"""
Chamada MULTI-PROVIDER (bônus +1 pt da Sprint 4): a mesma pergunta, a mesma recuperação, vários MODELOS e vários PROMPTS.

    python -m src.rag.multi_provider "Como pago com PIX?"
    python -m src.rag.multi_provider "Por que o preço subiu?" --modelos gpt-oss:120b gemma4:31b-cloud --prompts v1 v2

Cada combinação (modelo x prompt) passa pelo MESMO pipeline completo (guardrails, porta de distância, validação de citação),
então as diferenças vistas na tabela vêm só do modelo e do prompt. Também está na aba "Comparar modelos e prompts" da interface.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any, Callable

from src import config
from src.rag.cadeia_rag import ChargeGridRAG, criar_llm

MODELOS_PADRAO = [config.MODELO_CHAT_PRINCIPAL, config.MODELO_CHAT_COMPARACAO]
PROMPTS_PADRAO = ["v1", "v2"]


@dataclass
class RespostaComparada:
    modelo: str
    prompt: str
    status: str
    resposta: str
    fontes: list[str] = field(default_factory=list)
    escalar_suporte: bool = False
    latencia_s: float = 0.0
    tokens_prompt: int | None = None
    detalhe: str = ""


def comparar(
    pergunta: str,
    retriever: Any,
    modelos: list[str] | None = None,
    prompts: list[str] | None = None,
    dados_sessao: dict | None = None,
    llm_factory: Callable[[str], Any] = criar_llm,
) -> list[RespostaComparada]:
    """Executa todas as combinações (modelo x prompt). Erro em uma combinação não derruba as outras."""
    resultados: list[RespostaComparada] = []
    for modelo in modelos or MODELOS_PADRAO:
        try:
            llm = llm_factory(modelo)
        except Exception as erro:  # noqa: BLE001 — modelo inexistente / sem chave
            resultados += [RespostaComparada(modelo, p, "erro_llm", "", detalhe=f"{type(erro).__name__}: {str(erro)[:120]}") for p in prompts or PROMPTS_PADRAO]
            continue
        for prompt in prompts or PROMPTS_PADRAO:
            try:
                r = ChargeGridRAG(retriever, llm, versao_prompt=prompt).perguntar(pergunta, dados_sessao=dados_sessao)
                resultados.append(RespostaComparada(modelo, prompt, r.status, r.resposta, [f.citacao for f in r.fontes],
                                                    r.escalar_suporte, r.latencia_total_s, r.tokens_prompt, r.detalhe))
            except Exception as erro:  # noqa: BLE001
                resultados.append(RespostaComparada(modelo, prompt, "erro_llm", "", detalhe=f"{type(erro).__name__}: {str(erro)[:120]}"))
    return resultados


def tabela_markdown(resultados: list[RespostaComparada]) -> str:
    linhas = ["| Modelo | Prompt | Status | Latência | Tokens | Resposta | Fontes |", "|---|---|---|---|---|---|---|"]
    for r in resultados:
        texto = " ".join(r.resposta.split()).replace("|", "/")[:260] or f"({r.detalhe})"
        linhas.append(f"| {r.modelo} | {r.prompt} | {r.status} | {r.latencia_s:.2f} s | {r.tokens_prompt or '-'} | {texto} | {'; '.join(r.fontes) or '-'} |")
    return "\n".join(linhas)


def _main() -> None:
    ap = argparse.ArgumentParser(description="Compara modelos e prompts na mesma pergunta.")
    ap.add_argument("pergunta")
    ap.add_argument("--modelos", nargs="*", default=MODELOS_PADRAO)
    ap.add_argument("--prompts", nargs="*", default=PROMPTS_PADRAO, choices=["v1", "v2"])
    ap.add_argument("--sessao", action="store_true")
    args = ap.parse_args()
    from src.rag.embeddings import obter_embeddings
    from src.rag.indexar import garantir_indice
    from src.rag.retriever import Retriever
    from src.rag.vector_store import abrir_indice

    emb = obter_embeddings()
    garantir_indice(emb)
    sessao = {"bateria_percentual": 67, "energia_consumida_kwh": 18.4, "valor_acumulado_reais": 22.10, "tempo_restante_min": 23} if args.sessao else None
    print(tabela_markdown(comparar(args.pergunta, Retriever(abrir_indice(emb), cotas=config.COTAS_POR_TIPO), args.modelos, args.prompts, sessao)))


if __name__ == "__main__":
    _main()
