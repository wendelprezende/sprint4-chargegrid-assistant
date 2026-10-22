# -*- coding: utf-8 -*-
"""
T6 — Comparação de MODELOS e PARÂMETROS (Sprint 4: relatorio_modelos.md).

Roda o MESMO eval set (43 casos) e a MESMA recuperação (configuração final) variando só o modelo gerador ou um parâmetro.
O juiz é sempre o mesmo (gpt-oss:120b), para que as notas sejam comparáveis.

    python -m evals.comparar_modelos --rodar                  # todos os experimentos (~20 min; cada um é salvo ao terminar)
    python -m evals.comparar_modelos --rodar --experimentos M1_gptoss_ref M2_gemma4
    python -m evals.comparar_modelos --relatorio              # só regenera docs/relatorio_modelos.md

EXPERIMENTOS (uma variável por vez em relação ao M1):
  M1_gptoss_ref      gpt-oss:120b, temperature 0, top_p 0.9, max_tokens 800, cotas 3+2   (referência = configuração final)
  M2_gemma4          gemma4:31b-cloud, mesmos parâmetros                                   (OUTRO MODELO)
  M3_gptoss_temp03   temperature 0.3                                                       (temperatura)
  M4_gptoss_topp1    top_p 1.0                                                              (top_p; com temperature 0 deve ser neutro)
  M5_gptoss_topk8    cotas 5+3 (8 chunks)                                                   (top-k maior)
  M6_gptoss_np300    max_tokens 300                                                         (orçamento de tokens menor)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from src import config
from evals import avaliar_rag as av

PASTA = av.PASTA_RESULTADOS / "modelos"
ARQ = av.PASTA_EVALS / "modelos_resultados.json"
COTAS_BASE = {"ficticio_chargegrid": 3, "real_goodwe": 2}
COTAS_MAIOR = {"ficticio_chargegrid": 5, "real_goodwe": 3}

EXPERIMENTOS = {
    "M1_gptoss_ref": dict(modelo=config.MODELO_CHAT_PRINCIPAL, temperature=0.0, top_p=0.9, num_predict=800, cotas=COTAS_BASE),
    "M2_gemma4": dict(modelo=config.MODELO_CHAT_COMPARACAO, temperature=0.0, top_p=0.9, num_predict=800, cotas=COTAS_BASE),
    "M3_gptoss_temp03": dict(modelo=config.MODELO_CHAT_PRINCIPAL, temperature=0.3, top_p=0.9, num_predict=800, cotas=COTAS_BASE),
    "M4_gptoss_topp1": dict(modelo=config.MODELO_CHAT_PRINCIPAL, temperature=0.0, top_p=1.0, num_predict=800, cotas=COTAS_BASE),
    "M5_gptoss_topk8": dict(modelo=config.MODELO_CHAT_PRINCIPAL, temperature=0.0, top_p=0.9, num_predict=800, cotas=COTAS_MAIOR),
    "M6_gptoss_np300": dict(modelo=config.MODELO_CHAT_PRINCIPAL, temperature=0.0, top_p=0.9, num_predict=300, cotas=COTAS_BASE),
}


def resumir(nome: str, par: dict, res: dict) -> dict:
    g = res["agregados"]["geral"]
    estados: dict[str, int] = {}
    for e in res["entradas"]:
        estados[e["status"]] = estados.get(e["status"], 0) + 1
    return {"experimento": nome, "parametros": {**par, "k_total": sum(par["cotas"].values())}, "geral": g, "estados": estados,
            "erros_formato_ou_llm": estados.get("erro_formato", 0) + estados.get("erro_llm", 0), "duracao_s": res["duracao_s"]}


def rodar(escolhidos: list[str], refazer: bool) -> None:
    from src.rag.cadeia_rag import criar_llm
    from src.rag.embeddings import obter_embeddings

    doc, cal = av.carregar_json(av.ARQ_EVAL_SET), av.carregar_json(av.ARQ_CALIBRACAO)
    base = av.ITERACOES["it7_candidata"]
    PASTA.mkdir(parents=True, exist_ok=True)
    resumo = json.loads(ARQ.read_text(encoding="utf-8")) if ARQ.exists() else {}
    for nome in escolhidos:
        par = EXPERIMENTOS[nome]
        if nome in resumo and not refazer:
            print(f"[{nome}] já medido (use --refazer).")
            continue
        print(f"\n=== {nome}: {par['modelo']} | temperature={par['temperature']} top_p={par['top_p']} max_tokens={par['num_predict']} k={sum(par['cotas'].values())} ===")
        cfg = replace(base, nome=nome, cotas=par["cotas"], descricao=f"experimento de modelo/parâmetros: {nome}")
        res = av.rodar_iteracao(
            cfg, doc["casos"], cal, emb_factory=lambda p: obter_embeddings(p),
            llm_factory=lambda par=par: criar_llm(par["modelo"], temperature=par["temperature"], top_p=par["top_p"], num_predict=par["num_predict"]),
            juiz_factory=lambda: criar_llm(config.MODELO_CHAT_PRINCIPAL, num_predict=1500), dados_sessao_padrao=doc["dados_sessao_padrao"])
        (PASTA / f"{nome}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        resumo[nome] = resumir(nome, par, res)
        ARQ.write_text(json.dumps(resumo, ensure_ascii=False, indent=1), encoding="utf-8")
        g = resumo[nome]["geral"]
        print(f"  -> comportamento={g['comportamento_correto']} fidelidade={g['fidelidade']} relevância={g['relevancia']} "
              f"hit@k={g['hit_at_k']} tokens={g['tokens_prompt_medio']} lat={g['latencia_total_media_s']} erros={resumo[nome]['erros_formato_ou_llm']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rodar", action="store_true")
    ap.add_argument("--experimentos", nargs="*", default=list(EXPERIMENTOS), choices=list(EXPERIMENTOS))
    ap.add_argument("--refazer", action="store_true")
    ap.add_argument("--relatorio", action="store_true")
    args = ap.parse_args()
    if args.rodar:
        rodar(args.experimentos, args.refazer)
    if args.rodar or args.relatorio:
        from src.utils.gerar_relatorios import gerar_relatorio_modelos

        gerar_relatorio_modelos()
        print("\nRelatório atualizado: docs/relatorio_modelos.md")
    elif not args.rodar:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
