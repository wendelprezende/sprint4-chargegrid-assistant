# -*- coding: utf-8 -*-
"""
Reexecuta o eval set (evals/eval_set.json) sobre a versão refatorada em
LangChain (Sprint 03), medindo:
  - tokens do prompt (system prompt escolhido)
  - tokens por turno (entrada + saída), via tiktoken
  - latência por turno
  - taxa de "acerto" nos casos de jailbreak/out_of_scope (verificável
    automaticamente, pois os guardrails são determinísticos)
  - para happy_path/edge_case, o texto da resposta é salvo para revisão
    humana (comportamento de linguagem natural não é 100% verificável
    por regex sem um "juiz" -- isso é uma limitação documentada).

USO:
    export OLLAMA_API_KEY="sua_api_key"
    python evals/run_evals.py --prompt v2 --model gpt-oss:120b

    # Sem API key -> roda em modo offline com um LLM "fake", só para
    # validar que o pipeline (guardrails + chain + medição) funciona:
    python evals/run_evals.py --prompt v2 --offline

IMPORTANTE: os números salvos em `sprint3_results.json` quando rodado
com `--offline` são marcados como "modo": "offline_simulado" e NÃO devem
ser usados na tabela antes/depois do relatório de evolução -- essa
tabela exige números reais (ver docs/relatorio_evolucao.txt, seção 3).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent

import sys

sys.path.insert(0, str(RAIZ))

from src.chain.builder import ChargeGridAssistant, montar_chain_com_memoria
from src.guardrails.moderation import detectar_tentativa_de_manipulacao
from src.guardrails.scope_validator import validar_escopo


def carregar_system_prompt(versao: str) -> str:
    caminho = RAIZ / "prompts" / f"system_prompt_{versao}.md"
    texto = caminho.read_text(encoding="utf-8")
    # extrai apenas o bloco ```xml ... ``` ou ``` ... ``` do markdown
    inicio = texto.find("```")
    fim = texto.find("```", inicio + 3)
    bloco = texto[inicio:fim]
    # remove a primeira linha (```xml ou ```)
    linhas = bloco.split("\n")[1:]
    return "\n".join(linhas).strip()


def contar_tokens(texto: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(texto))
    except Exception:
        # fallback offline (ambiente sem acesso à tabela de encoding do tiktoken)
        return len(texto.split())


def rodar(versao_prompt: str, model: str, offline: bool) -> dict:
    system_prompt = carregar_system_prompt(versao_prompt)
    tokens_prompt = contar_tokens(system_prompt)

    eval_set = json.loads((RAIZ / "evals" / "eval_set.json").read_text(encoding="utf-8"))

    assistant = ChargeGridAssistant(system_prompt=system_prompt, model=model)

    if offline:
        from langchain_core.language_models.fake_chat_models import FakeListChatModel

        respostas_fake = [
            "Simulação offline: aqui entraria a resposta real do modelo."
        ] * len(eval_set["casos"])
        assistant._chain_com_memoria = montar_chain_com_memoria(
            FakeListChatModel(responses=respostas_fake),
            system_prompt,
            gerenciador=assistant.gerenciador,
        )

    resultados = []
    acertos_guardrail = 0
    total_guardrail = 0
    soma_latencia = 0.0
    soma_tokens_turno = 0

    for caso in eval_set["casos"]:
        session_id = f"eval-{caso['id']}"
        inicio = time.perf_counter()
        resposta = assistant.perguntar(session_id, caso["pergunta"])
        latencia = time.perf_counter() - inicio

        tokens_entrada = contar_tokens(caso["pergunta"])
        tokens_saida = contar_tokens(resposta)
        tokens_turno = tokens_entrada + tokens_saida
        soma_tokens_turno += tokens_turno
        soma_latencia += latencia

        guardrail_disparado = (
            detectar_tentativa_de_manipulacao(caso["pergunta"]).bloqueado
            or not validar_escopo(caso["pergunta"]).dentro_do_escopo
        )

        acerto_automatico = None
        if caso["categoria"] in ("jailbreak", "out_of_scope"):
            total_guardrail += 1
            acerto_automatico = guardrail_disparado == caso["deve_recusar"]
            if acerto_automatico:
                acertos_guardrail += 1

        resultados.append(
            {
                "id": caso["id"],
                "categoria": caso["categoria"],
                "pergunta": caso["pergunta"],
                "resposta": resposta,
                "tokens_turno": tokens_turno,
                "latencia_segundos": round(latencia, 4),
                "guardrail_disparado": guardrail_disparado,
                "acerto_automatico": acerto_automatico,
                "requer_revisao_humana": acerto_automatico is None,
            }
        )

    n_casos = len(eval_set["casos"])
    relatorio = {
        "modo": "offline_simulado" if offline else "real",
        "versao_prompt": versao_prompt,
        "modelo": model,
        "tokens_prompt_sistema": tokens_prompt,
        "media_tokens_por_turno": round(soma_tokens_turno / n_casos, 1),
        "latencia_media_segundos": round(soma_latencia / n_casos, 4),
        "taxa_acerto_guardrails": (
            round(acertos_guardrail / total_guardrail, 3) if total_guardrail else None
        ),
        "casos_avaliados": n_casos,
        "resultados": resultados,
    }
    return relatorio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reexecuta o eval set do ChargeGrid Assistant.")
    parser.add_argument("--prompt", choices=["v1", "v2"], default="v2")
    parser.add_argument("--model", default="gpt-oss:120b")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Roda sem chamar o Ollama de verdade (usa um LLM fake só para validar o pipeline).",
    )
    args = parser.parse_args()

    relatorio = rodar(args.prompt, args.model, args.offline)

    saida = RAIZ / "evals" / "sprint3_results.json"
    saida.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Resultados salvos em {saida}")
    print(f"Modo: {relatorio['modo']}")
    print(f"Tokens do prompt de sistema: {relatorio['tokens_prompt_sistema']}")
    print(f"Média de tokens por turno: {relatorio['media_tokens_por_turno']}")
    print(f"Latência média (s): {relatorio['latencia_media_segundos']}")
    print(f"Taxa de acerto dos guardrails: {relatorio['taxa_acerto_guardrails']}")
    if relatorio["modo"] == "offline_simulado":
        print(
            "\nATENÇÃO: modo offline_simulado -- estes números NÃO substituem "
            "a execução real exigida na tabela antes/depois do relatório de evolução."
        )
