# -*- coding: utf-8 -*-
"""
Checklist de ENTREGA da Sprint 4 (T10): confere cada item exigido pelo guia e a higiene de credenciais.

    python -m evals.checklist_entrega
"""

from __future__ import annotations

import json
import subprocess
import sys

from src import config

RAIZ = config.RAIZ
OBRIGATORIOS = {
    "README com nome, integrantes e como executar": ["README.md"],
    "Base de conhecimento expandida (catálogo + PDFs)": ["data/knowledge_base/catalogo.json", "data/knowledge_base/CG-01_Guia_do_Motorista.pdf", "data/knowledge_base/GW_HCAG2_DatasheetPT.pdf"],
    "Pipeline RAG (src/rag/)": [f"src/rag/{n}.py" for n in ("loader", "chunking", "embeddings", "vector_store", "retriever", "prompt_rag", "cadeia_rag")],
    "Prompt RAG versionado + tabela de versões": ["prompts/prompt_rag_v1.md", "prompts/prompt_rag_v2.md", "prompts/versoes_rag.md"],
    "Interface web (app/) com README de execução": ["app/app.py", "app/README.md"],
    "Avaliação: eval set, rubrica e scores por iteração": ["evals/eval_set_rag.json", "evals/rubrica_avaliacao.md", "evals/scores_por_iteracao.json", "evals/scores_por_iteracao.csv"],
    "Relatórios (RAG, modelos, evolução em PDF)": ["docs/relatorio_rag.md", "docs/relatorio_modelos.md", "docs/relatorio_evolucao_sprint4.pdf"],
    "Arquivo .txt de entrega (nome, RM, turma)": ["entrega_sprint4.txt"],
}


def main() -> int:
    falhas = avisos = 0

    def ok(c: bool, d: str, det: str = "", aviso: bool = False) -> None:
        nonlocal falhas, avisos
        marca = "OK" if c else ("AVISO" if aviso else "FALHA")
        falhas += 0 if (c or aviso) else 1
        avisos += 1 if (not c and aviso) else 0
        print(f"  [{marca}] {d}" + (f"  -> {det}" if det and not c else ""))

    print("=== Checklist de entrega — Sprint 4 ===\n\n1) Arquivos exigidos")
    for item, arqs in OBRIGATORIOS.items():
        faltam = [a for a in arqs if not (RAIZ / a).exists()]
        ok(not faltam, item, "faltando: " + ", ".join(faltam))

    print("\n2) Exigências do guia")
    sc_path = RAIZ / "evals/scores_por_iteracao.json"
    sc = {n: d for n, d in (json.loads(sc_path.read_text(encoding="utf-8")) if sc_path.exists() else {}).items() if not d.get("so_recuperacao")}
    base = sc.get("it0_baseline", {}).get("geral", {})
    ganhos = [n for n, d in sc.items() if n not in ("it0_baseline", "it0_sem_rag", "it0_sem_rag_ref", "it0_prompt_v1")
              and any((d["geral"].get(k) or 0) > (base.get(k) or 0) + 0.04 for k in ("hit_at_k", "mrr", "comportamento_correto", "citacao_correta"))]
    ok(len(ganhos) >= 2, f"pelo menos 2 iterações com ganho de score sobre o baseline ({len(ganhos)}: {', '.join(ganhos) or '-'})", "rode python -m evals.avaliar_rag --todas")
    ok("it0_sem_rag" in sc or "it0_sem_rag_ref" in sc, "coluna 'antes' (Sprints 1/2) medida no mesmo eval set")
    rel = (RAIZ / "docs/relatorio_modelos.md").read_text(encoding="utf-8") if (RAIZ / "docs/relatorio_modelos.md").exists() else ""
    mm = json.loads((RAIZ / "evals/modelos_resultados.json").read_text(encoding="utf-8")) if (RAIZ / "evals/modelos_resultados.json").exists() else {}
    modelos_medidos = {d["parametros"]["modelo"] for d in mm.values()}
    ok(len(modelos_medidos) >= 2 and "Ainda não medido" not in rel, f"comparação de 2+ modelos MEDIDA e no relatorio_modelos.md (modelos: {', '.join(sorted(modelos_medidos)) or 'nenhum'})",
       "OBRIGATÓRIO (guia §6): rode python -m evals.comparar_modelos --rodar --experimentos M1_gptoss_ref M2_gemma4 e depois python -m src.utils.gerar_relatorios")
    ok(all(t in rel for t in ("temperature", "top_p", "max_tokens")) and "k" in rel, "relatorio_modelos.md documenta temperature, top_p, max_tokens e k")
    pdf = RAIZ / "docs/relatorio_evolucao_sprint4.pdf"
    if pdf.exists():
        try:
            n = next(int(l.split()[-1]) for l in subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout.splitlines() if l.startswith("Pages:"))
            ok(n <= 5, f"relatório de evolução tem {n} página(s) (limite 5)")
        except Exception:  # noqa: BLE001
            ok(False, "contagem de páginas do PDF", "pdfinfo indisponível: abra o PDF e confira que tem no máximo 5 páginas", aviso=True)
    equipe = (RAIZ / "docs/equipe.json").read_text(encoding="utf-8") if (RAIZ / "docs/equipe.json").exists() else ""
    ok("PREENCHER" not in equipe, "equipe e divisão de trabalho preenchidas em docs/equipe.json (OBRIGATÓRIO no relatório de evolução, guia §8)", "edite docs/equipe.json e rode python -m src.utils.gerar_relatorios")
    pend = [n for n, d in sc.items() if (d.get("configuracao") or {}).get("prefixo")
            and (RAIZ / f"evals/resultados/{n}.json").exists()
            and not json.loads((RAIZ / f"evals/resultados/{n}.json").read_text(encoding="utf-8")).get("recalculado")]
    ok(not pend, "relevância das iterações com prefixo recalculada (sem artefato de medição)", "rode python -m evals.avaliar_rag --recalcular e depois gerar_relatorios", aviso=True)
    txt = (RAIZ / "entrega_sprint4.txt").read_text(encoding="utf-8") if (RAIZ / "entrega_sprint4.txt").exists() else ""
    ok(all(rm in txt for rm in ("573308", "573075", "573838", "573126")) and "1CCPK" in txt and "github.com" in txt, "entrega_sprint4.txt traz nome, RM, turma e link do repositório")

    print("\n3) Credenciais e Git (exposição de chave tem penalidade)")
    try:
        dentro = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=RAIZ, capture_output=True, text=True)
        if dentro.returncode != 0:
            raise FileNotFoundError
        rast = subprocess.run(["git", "ls-files"], cwd=RAIZ, capture_output=True, text=True).stdout.splitlines()
        ok(".env" not in rast and not any(f.startswith("chroma_db/") for f in rast) and not any(f.endswith("UserManualPT.pdf") for f in rast),
           "nenhum arquivo proibido versionado (.env, chroma_db/, manual da GoodWe)")
        try:
            chave = config.obter_api_key()
            hist = subprocess.run(["git", "log", "--all", "--oneline", "-S", chave], cwd=RAIZ, capture_output=True, text=True).stdout.strip()
            ok(not hist, "a sua chave NÃO aparece em nenhum commit do histórico", "REVOGUE a chave em ollama.com/settings/keys")
        except RuntimeError:
            ok(True, "busca da chave no histórico (sem chave no .env: nada a procurar)")
        autores = [l for l in subprocess.run(["git", "shortlog", "-sne", "--all"], cwd=RAIZ, capture_output=True, text=True).stdout.splitlines() if l.strip()]
        ok(len(autores) >= 4, f"commits de {len(autores)} autor(es) no histórico (o guia pede commits regulares de CADA integrante)",
           "risco assumido pela equipe: só os commits do Wendel", aviso=True)
        sujo = [l for l in subprocess.run(["git", "status", "--porcelain"], cwd=RAIZ, capture_output=True, text=True).stdout.splitlines()]
        ok(not sujo, f"tudo commitado ({len(sujo)} alteração(ões) pendente(s))", "faça commit e push antes de entregar", aviso=True)
    except FileNotFoundError:
        ok(False, "checagens de Git", "git não encontrado ou pasta não é um repositório", aviso=True)
    print(f"\n>>> {'PRONTO PARA ENTREGAR.' if falhas == 0 else str(falhas) + ' item(ns) obrigatório(s) pendente(s).'}" + (f" ({avisos} aviso(s))" if avisos else ""))
    return 1 if falhas else 0


if __name__ == "__main__":
    sys.exit(main())
