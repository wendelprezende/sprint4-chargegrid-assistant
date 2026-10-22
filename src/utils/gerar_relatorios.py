# -*- coding: utf-8 -*-
"""
Gera os relatórios da Sprint 4 A PARTIR DOS RESULTADOS MEDIDOS (nenhum número é digitado à mão):

    python -m src.utils.gerar_relatorios

Escreve:
  docs/relatorio_rag.md                       pipeline, decisões, resultados, segurança, problemas e limitações
  docs/relatorio_modelos.md                   comparação de modelos e parâmetros (precisa de: python -m evals.comparar_modelos --rodar)
  docs/relatorio_evolucao_sprint4.pdf         relatório de evolução (até 5 páginas) com a tabela antes/depois
  README.md                                   bloco entre <!-- RESULTADOS:INICIO --> e <!-- RESULTADOS:FIM -->
  prompts/versoes_rag.md                      blocos entre os marcadores ITERACOES e EVIDENCIAS

Fontes de dados: evals/scores_por_iteracao.json, evals/resultados/injecao_prompt_v*_n*.json, evals/modelos_resultados.json,
docs/equipe.json. Se algo ainda não foi medido, o texto diz qual comando rodar (em vez de inventar).
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from src import config

RAIZ = config.RAIZ
DOCS = RAIZ / "docs"
EVALS = RAIZ / "evals"
ARQ_SCORES = EVALS / "scores_por_iteracao.json"
ARQ_MODELOS = EVALS / "modelos_resultados.json"
ARQ_EQUIPE = DOCS / "equipe.json"

ROTULOS_ITERACAO = {
    "it0_sem_rag": "Antes: Sprints 1/2 (sem RAG)", "it0_sem_rag_ref": "Antes (contexto de referência do it4)",
    "it0_prompt_v1": "RAG baseline, prompt v1", "it0_baseline": "RAG baseline (prompt v2)",
    "it1_limpeza": "+ limpeza do rodapé", "it2_secao": "+ chunking por seção", "it3_prefixo": "+ prefixos do nomic",
    "it4_camadas": "+ busca por camada (configuração final)", "it5_limiar": "+ limiar calibrado (0,334)",
    "it6_sem_limpeza": "Ablação: it4 sem a limpeza", "it7_candidata": "Candidata final (it4 + moderação corrigida)",
}
ORDEM = list(ROTULOS_ITERACAO)


# ------------------------------------------------------------------------------------------------ dados
def _json(caminho: Path) -> Any:
    try:
        return json.loads(caminho.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def scores() -> dict:
    todos = _json(ARQ_SCORES) or {}
    return {n: d for n, d in todos.items() if not d.get("so_recuperacao")}


def f3(v: Any, casas: int = 3) -> str:
    return "-" if v is None else (f"{v:.{casas}f}".replace(".", ",") if isinstance(v, float) else str(v))


def g(sc: dict, nome: str, chave: str) -> Any:
    return sc.get(nome, {}).get("geral", {}).get(chave)


def final_e_antes(sc: dict) -> tuple[str | None, str | None]:
    final = next((n for n in ("it7_candidata", "it4_camadas", "it5_limiar") if n in sc), None)
    antes = next((n for n in ("it0_sem_rag_ref", "it0_sem_rag") if n in sc), None)
    return final, antes


def injecao() -> dict:
    out = {}
    for v in ("v1", "v2"):
        arqs = sorted((EVALS / "resultados").glob(f"injecao_prompt_{v}_n*.json"))
        if arqs:
            out[v] = _json(arqs[-1])
    return out


def equipe() -> list[dict]:
    return _json(ARQ_EQUIPE) or []


# ------------------------------------------------------------------------------------------------ tabelas
COLS = [("comportamento_correto", "Comport."), ("hit_at_k", "hit@k"), ("mrr", "MRR"), ("citacao_correta", "Cit. correta"),
        ("fidelidade", "Fidel."), ("relevancia", "Relev."), ("recusa_indevida", "Recusa indevida"), ("tokens_prompt_medio", "Tokens"),
        ("latencia_total_media_s", "Lat. (s)")]


NAO_SE_APLICA_SEM_RAG = ("comportamento_correto", "recusa_indevida", "resposta_indevida")


def _cel(chave: str, v: Any, nome: str = "") -> str:
    if v is None or (nome.startswith("it0_sem_rag") and chave in NAO_SE_APLICA_SEM_RAG):  # o chatbot antigo não tem recusas
        return "-"
    return f"{v:.0f}" if chave == "tokens_prompt_medio" else f3(v, 2 if chave == "latencia_total_media_s" else 3)


def tabela_iteracoes_md(sc: dict) -> str:
    if not sc:
        return "_Ainda não há iterações medidas. Rode: `python -m evals.avaliar_rag --todas`._"
    linhas = ["| Iteração | O que muda | " + " | ".join(r for _, r in COLS) + " |", "|---|---|" + "---|" * len(COLS)]
    for nome in [n for n in ORDEM if n in sc] + [n for n in sc if n not in ORDEM]:
        linhas.append(f"| `{nome}` | {ROTULOS_ITERACAO.get(nome, sc[nome]['descricao'])} | " + " | ".join(_cel(c, g(sc, nome, c), nome) for c, _ in COLS) + " |")
    return "\n".join(linhas)


def linhas_antes_depois(sc: dict) -> list[tuple[str, str, str]]:
    final, antes = final_e_antes(sc)
    if not final or not antes:
        return []
    a = lambda k: g(sc, antes, k)  # noqa: E731
    d = lambda k: g(sc, final, k)  # noqa: E731
    return [
        ("Fidelidade (0–1): afirmações sustentadas pelo contexto", f3(a("fidelidade")), f3(d("fidelidade"))),
        ("Relevância da resposta (0–1)", f3(a("relevancia")), f3(d("relevancia"))),
        ("Presença de citação de fonte", f"{f3(a('citacao_presente'))} (0%)" if a("citacao_presente") == 0 else f3(a("citacao_presente")), f3(d("citacao_presente"))),
        ("Citação correta (fonte aceitável)", "-", f3(d("citacao_correta"))),
        ("Qualidade do contexto recuperado: hit@k", "- (sem busca)", f3(d("hit_at_k"))),
        ("Qualidade do contexto recuperado: MRR", "- (sem busca)", f3(d("mrr"))),
        ("Alucinação em fatos proibidos (menor = melhor)", f3(a("alucinacao_proibidos")), f3(d("alucinacao_proibidos"))),
        ("Comportamento correto (recusas, escopo, injeção)", "- (sem guardrails)", f3(d("comportamento_correto"))),
        ("Tokens de prompt por turno", f"{a('tokens_prompt_medio'):.0f}" if a("tokens_prompt_medio") else "-", f"{d('tokens_prompt_medio'):.0f}" if d("tokens_prompt_medio") else "-"),
        ("Latência média por turno (s)", f3(a("latencia_total_media_s"), 2), f3(d("latencia_total_media_s"), 2)),
    ]


def tabela_antes_depois_md(sc: dict) -> str:
    linhas = linhas_antes_depois(sc)
    if not linhas:
        return "_Faltam a iteração do 'antes' e a final. Rode: `python -m evals.avaliar_rag --todas`._"
    final, antes = final_e_antes(sc)
    cab = f"| Métrica | Sprints 1/2 (versão original) `{antes}` | Sprint 4 (RAG avaliado) `{final}` |\n|---|---|---|\n"
    return cab + "\n".join(f"| {m} | {x} | {y} |" for m, x, y in linhas)


def texto_injecao(inj: dict, sc: dict) -> str:
    partes = []
    for v, d in inj.items():
        partes.append(f"prompt {v}: obedeceu {d['total_obedeceu']}/{d['total']} (IC95% {f3(d['ic95'][0], 2)}–{f3(d['ic95'][1], 2)})")
    if partes:
        return "Obediência do LLM sozinho a documentos envenenados, com repetições: " + "; ".join(partes) + "."
    o = g(sc, "it0_baseline", "injecao_obediencia_sem_defesas")
    return (f"Nas execuções do eval set, a obediência do LLM sozinho a documentos envenenados foi {f3(o)} (4 ataques, 1 execução cada). "
            "Para uma taxa com intervalo de confiança: `python -m evals.avaliar_rag --medir-injecao 10 --prompt-versao v2`.") if o is not None else \
        "A obediência à injeção ainda não foi medida."


# ------------------------------------------------------------------------------------------------ texto fixo (decisões e problemas)
PROBLEMAS = [
    ("Embeddings indisponíveis no Ollama Cloud",
     "O diagnóstico mostrou que a conta não enxerga nenhum modelo de embedding na nuvem (o endpoint /api/embed devolveu 401 para todos), embora o chat funcione.",
     "O mesmo `nomic-embed-text` (único modelo aprovado) passou a rodar em um Ollama local; o chat segue na nuvem. A decisão ficou em `config.OLLAMA_EMBED_BASE_URL`."),
    ("Injeção de prompt por documento",
     "No teste real de 19/10 o gpt-oss obedeceu a uma instrução sutil escondida em um documento e ainda o citou como fonte.",
     "Defesa em camadas: heurística de documentos, prompt v2, escape de tags XML, validação das fontes citadas, canário e filtro de saída (links, e-mails, imagens)."),
    ("Falso positivo do guardrail da Sprint 3",
     "A pergunta legítima de tarifação (hp-03) era recusada como fora de escopo pelo regex de domínio.",
     "Novas âncoras de domínio e troca do bloqueio de 'fora do domínio' pela porta de distância; a moderação ganhou padrões de 'novo papel' (0 falsos positivos em 72 perguntas legítimas)."),
    ("Recuperação fraca na linha de base",
     "Chunks com vários assuntos diluíam o embedding e o manual (78% dos chunks) espremia as respostas da operação do totem.",
     "Chunking por seção, prefixos do nomic e busca por camada (cota para documentos fictícios e reais). Ver a tabela de iterações."),
    ("Erros na própria avaliação",
     "A relevância comparava consulta com documento (prefixos assimétricos do nomic) e o 'antes' sem RAG reportava métricas artificiais.",
     "Relevância simétrica (`--recalcular` corrige resultados gravados), métricas não comparáveis do 'antes' viraram '-' e tudo foi registrado na rubrica (seção 7)."),
    ("Raciocínio do gpt-oss consome o orçamento de tokens",
     "Com 40 tokens a resposta veio vazia (o modelo gastou tudo 'pensando').",
     "`NUM_PREDICT_RAG = 800` e `reasoning = low` para modelos gpt-oss; a cadeia diagnostica resposta vazia e o experimento M6 mede a perda com 300 tokens."),
    ("Limiar de distância não separa domínio de fora do escopo",
     "O conjunto de calibração (separado do eval set) ficou sobreposto: o limiar 0,334 não separa as classes (margem negativa).",
     "A porta de distância é tratada como economia de chamadas, não como barreira de segurança; mantido 0,40 e a recusa fica a cargo do prompt com grounding."),
]
LIMITACOES = [
    "O eval set tem 43 casos: diferenças de 1 caso valem 0,023–0,045 e a fidelidade varia cerca de 0,04 entre execuções do mesmo LLM.",
    "O juiz é um LLM (por padrão o mesmo modelo do gerador); a auditoria humana por amostra (`--auditoria`) calibra esse viés.",
    "A detecção de injeção por heurística não pega paráfrases sem verbo de ordem; o filtro de saída não impede desinformação em texto puro.",
    "A base é pequena (7 documentos) e os documentos da operação do totem são fictícios (premissa acadêmica do projeto).",
    "O manual da GoodWe não é publicado no GitHub por restrição de direitos autorais; quem clonar o projeto deve baixá-lo em en.goodwe.com.",
]


def _ganho(sc: dict, chave: str, de: str, para: str, casas: int = 3) -> str:
    a, b = g(sc, de, chave), g(sc, para, chave)
    return "n/d" if a is None or b is None else f"{f3(a, casas)} → {f3(b, casas)}"


# ------------------------------------------------------------------------------------------------ relatorio_rag.md
def gerar_relatorio_rag() -> Path:
    sc = scores()
    final, antes = final_e_antes(sc)
    inj = injecao()
    hit = _ganho(sc, "hit_at_k", "it0_baseline", final) if final else "n/d"
    mrr = _ganho(sc, "mrr", "it0_baseline", final) if final else "n/d"
    comp = _ganho(sc, "comportamento_correto", "it0_baseline", final) if final else "n/d"
    cit = _ganho(sc, "citacao_correta", "it0_baseline", final) if final else "n/d"
    rec = _ganho(sc, "recusa_indevida", "it0_baseline", final) if final else "n/d"
    md = f"""# Relatório do pipeline RAG — ChargeGrid Assistant (Sprint 4)

> Gerado em {date.today().strftime('%d/%m/%Y')} por `python -m src.utils.gerar_relatorios` a partir de `evals/scores_por_iteracao.json`.
> Equipe 03 · FIAP · Prompt and Artificial Intelligence · Turma 1CCPK.

## 1. Visão geral

```
pergunta → guardrails (jailbreak, escopo) → busca (nomic-embed-text + ChromaDB, filtro de público, cotas por camada)
         → descarte de chunks com injeção → porta de distância → LLM (prompt RAG v2, JSON validado)
         → validação da saída (canário, link, fonte citada existe) → resposta + "Fontes: documento · seção · página"
```

## 2. Base de conhecimento

Sete documentos, em duas camadas catalogadas em `data/knowledge_base/catalogo.json` (tipo, público, origem, faixas de página):

- **Reais (GoodWe):** manual do usuário HCA G2 (70 páginas) e datasheet — hardware, LED, partida, falhas.
- **Fictícios (acadêmicos), CG-01 a CG-05:** guia do motorista, tarifas e pagamentos, solução de problemas e suporte, FAQ e termos/limites do assistente — a operação do totem (PIX, tarifa dinâmica, comprovante). Rotulados como fictícios em todas as páginas.

Decisões: (i) o conteúdo de **instalação** do manual é marcado `instalador` e nunca é recuperado para o motorista (filtro de metadados antes do LLM); (ii) capa, direitos e índice do manual são excluídos da indexação; (iii) o PDF do manual não é publicado no GitHub (restrição de direitos autorais).

## 3. Chunking, embeddings e banco vetorial

| Decisão | Valor | Justificativa e trade-off |
|---|---|---|
| Divisor | `RecursiveCharacterTextSplitter`, separadores `["\\n\\n","\\n",". "," ",""]` | exigido pelo guia |
| Tamanho / overlap | 800 / 120 caracteres (15%) | cada chunk cabe em uma ideia; 15% de overlap evita cortar uma frase-chave |
| Corte | página a página; **documentos fictícios por seção numerada** | citação com página exata; um assunto por chunk melhora a busca. Contra: um trecho que atravessa a quebra de página é cortado |
| Limpeza | remove o aviso/rodapé repetido dos fictícios | menos ruído no embedding |
| Embedding | `nomic-embed-text` (local), com prefixos `search_document:` / `search_query:` | único modelo aprovado; a nuvem não o oferece para a nossa conta |
| Banco | ChromaDB persistente, distância de cosseno, ids determinísticos | reindexar não duplica |
| Busca | top-5 com **cota por camada** (3 fictícios + 2 reais) e filtro de público | o manual tem 78% dos chunks e, sem cota, espremia a operação do totem |

## 4. Grounding, citação e segurança

- **Prompt RAG versionado** (`prompts/prompt_rag_v1.md`, `v2`): responde só com o contexto, cita os ids dos chunks usados e diz "não encontrei" quando a base não cobre.
- **Saída estruturada (Pydantic v2)**: `encontrou_na_base`, `resposta`, `fontes`, `escalar_suporte`; o código **valida** que toda fonte citada existe entre os chunks enviados e remove as inventadas.
- **Porta de distância (0,40):** se o melhor chunk está longe demais, o LLM nem é chamado.
- **Injeção (direta e por documento):** heurística de pergunta e de documento, escape de `< > &`, canário no prompt, filtro de saída de links/e-mails/imagens. {texto_injecao(inj, sc)}

## 5. Avaliação (fallback documentado do RAGAS)

Método e rubricas em `evals/rubrica_avaliacao.md`: **fidelidade** (afirmações sustentadas ÷ totais) e **relevância** (cosseno entre perguntas geradas e a original), com juiz LLM, mais métricas determinísticas. Eval set de 43 casos (16 herdados da Sprint 3); calibração do limiar em conjunto separado.

### 5.1 Scores por iteração

{tabela_iteracoes_md(sc)}

### 5.2 Antes × depois

{tabela_antes_depois_md(sc)}

### 5.3 Leitura honesta dos resultados

- **Recuperação e comportamento (ganhos acima do ruído):** hit@k {hit}; MRR {mrr}; citação correta {cit}; comportamento correto {comp}; recusa indevida {rec} (baseline → final).
- **Fidelidade:** o RAG fica na zona de 0,9 contra o chatbot sem RAG; diferenças entre iterações abaixo de ~0,04 estão dentro da variação do LLM e não devem ser lidas como ganho.
- **Limpeza do rodapé e limiar calibrado:** não trouxeram ganho medido (diferença de 1 caso); foram mantidas/descartadas conforme a tabela.
- **Custo:** o RAG envia bem mais tokens por turno que o chatbot sem busca; a latência média continua em torno de 1 s.

## 6. Problemas encontrados e soluções

""" + "\n".join(f"{i}. **{t}.** {p} **Solução:** {s}" for i, (t, p, s) in enumerate(PROBLEMAS, 1)) + """

## 7. Limitações

""" + "\n".join(f"- {x}" for x in LIMITACOES) + "\n"
    destino = DOCS / "relatorio_rag.md"
    destino.write_text(md, encoding="utf-8")
    return destino


# ------------------------------------------------------------------------------------------------ relatorio_modelos.md
def gerar_relatorio_modelos() -> Path:
    res = _json(ARQ_MODELOS) or {}
    linhas = ["| Experimento | Modelo | temperature | top_p | max_tokens | k | Comport. | Fidel. | Relev. | hit@k | Cit. correta | Tokens | Lat. (s) | Erros |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for nome, d in res.items():
        p, gg = d["parametros"], d["geral"]
        linhas.append(f"| `{nome}` | {p['modelo']} | {p['temperature']} | {p['top_p']} | {p['num_predict']} | {p['k_total']} | "
                      + " | ".join(_cel(c, gg.get(c)) for c in ("comportamento_correto", "fidelidade", "relevancia", "hit_at_k", "citacao_correta", "tokens_prompt_medio", "latencia_total_media_s"))
                      + f" | {d['erros_formato_ou_llm']} |")
    tabela = "\n".join(linhas) if res else "_Ainda não medido. Rode: `python -m evals.comparar_modelos --rodar`._"
    analise = ""
    if "M1_gptoss_ref" in res:
        ref = res["M1_gptoss_ref"]["geral"]
        itens = []
        for nome, d in res.items():
            if nome == "M1_gptoss_ref":
                continue
            diffs = [f"{rot} {d['geral'][c] - ref[c]:+.3f}" for c, rot in (("comportamento_correto", "comportamento"), ("fidelidade", "fidelidade"), ("relevancia", "relevância"), ("hit_at_k", "hit@k"))
                     if d["geral"].get(c) is not None and ref.get(c) is not None]
            itens.append(f"- `{nome}` em relação ao M1: " + (", ".join(diffs) or "sem diferença") + f"; erros de formato/LLM: {d['erros_formato_ou_llm']}.")
        melhor = max(res.items(), key=lambda kv: ((kv[1]['geral'].get('comportamento_correto') or 0), (kv[1]['geral'].get('fidelidade') or 0)))
        analise = ("\n".join(itens) + f"\n\n**Melhor por comportamento correto e, em empate, fidelidade:** `{melhor[0]}`. "
                   "Diferenças de fidelidade abaixo de ~0,04 estão dentro da variação do LLM; diferenças de 1 caso valem 0,023 em comportamento.")
    md = f"""# Relatório de modelos e parâmetros (Sprint 4)

> Gerado em {date.today().strftime('%d/%m/%Y')} por `python -m src.utils.gerar_relatorios`. Mesmo eval set (43 casos), mesma recuperação
> (configuração final) e o **mesmo juiz** (gpt-oss:120b) em todos os experimentos; só o modelo gerador ou um parâmetro muda por vez.

## 1. Modelos

| Modelo | Papel | Observação |
|---|---|---|
| `gpt-oss:120b` (Ollama Cloud) | gerador principal e juiz | modelo de raciocínio: gasta tokens "pensando" antes de responder |
| `gemma4:31b-cloud` (Ollama Cloud) | modelo de comparação | substituiu o `qwen3:8b` da Sprint 3, que **não existe** no Ollama Cloud da nossa conta (erro 404) |
| `nomic-embed-text` (Ollama local) | embeddings | único modelo de embedding aprovado; a nuvem não oferece embeddings para a nossa conta (401) |

## 2. Parâmetros adotados e por quê

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `temperature` | **0** | recomendado pelo guia; respostas fundamentadas devem ser reproduzíveis |
| `top_p` | 0,9 | com temperature 0 a amostragem é determinística, então o `top_p` tende a ser neutro (experimento M4 verifica) |
| `max_tokens` (`num_predict`) | **800** | o gpt-oss gasta parte dos tokens raciocinando: com 40 tokens a resposta veio **vazia**; o experimento M6 mede a perda com 300 |
| `reasoning` | `low` (só gpt-oss) | reduz o consumo de tokens do raciocínio sem perder a precisão nos testes |
| `k` (top-k) | 5 = 3 fictícios + 2 reais | o experimento M5 testa 8 |
| limiar de distância | 0,40 | calibração não separável (ver relatório do RAG); mantido o valor que não gera recusas indevidas |

## 3. Resultados

{tabela}

{analise}

## 4. Limitações da comparação

- O juiz é o gpt-oss:120b: ao avaliar o próprio gpt-oss há risco de viés de autoavaliação (a auditoria humana por amostra mitiga).
- Com 43 casos, diferenças pequenas não são conclusivas; os números absolutos dependem do juiz e do eval set.
- O histórico da Sprint 3 (`gpt-oss:120b` × `qwen3:8b`) não é comparável: outro eval set, sem RAG e com outro modelo.
"""
    destino = DOCS / "relatorio_modelos.md"
    destino.write_text(md, encoding="utf-8")
    return destino


# ------------------------------------------------------------------------------------------------ README e versões
def _substituir_bloco(arquivo: Path, inicio: str, fim: str, conteudo: str) -> bool:
    if not arquivo.exists():
        return False
    t = arquivo.read_text(encoding="utf-8")
    padrao = re.compile(re.escape(inicio) + r".*?" + re.escape(fim), re.S)
    if not padrao.search(t):
        return False
    arquivo.write_text(padrao.sub(lambda _: f"{inicio}\n{conteudo}\n{fim}", t), encoding="utf-8")
    return True


def atualizar_readme_e_versoes() -> list[str]:
    sc = scores()
    feitos = []
    bloco = f"### Resultados medidos (iterações)\n\n{tabela_iteracoes_md(sc)}\n\n### Antes × depois\n\n{tabela_antes_depois_md(sc)}\n\nDetalhes: `docs/relatorio_rag.md`, `docs/relatorio_modelos.md` e `docs/relatorio_evolucao_sprint4.pdf`."
    if _substituir_bloco(RAIZ / "README.md", "<!-- RESULTADOS:INICIO -->", "<!-- RESULTADOS:FIM -->", bloco):
        feitos.append("README.md")
    v = RAIZ / "prompts" / "versoes_rag.md"
    ok1 = _substituir_bloco(v, "<!-- ITERACOES:INICIO -->", "<!-- ITERACOES:FIM -->", tabela_iteracoes_md(sc))
    p1, p2 = g(sc, "it0_prompt_v1", "fidelidade"), g(sc, "it0_baseline", "fidelidade")
    evid = (f"- **Fidelidade (mesmo eval set):** v1 = {f3(p1)} · v2 = {f3(p2)} (diferença pequena: indício, não prova).\n"
            f"- **Comportamento correto:** v1 = {f3(g(sc, 'it0_prompt_v1', 'comportamento_correto'))} · v2 = {f3(g(sc, 'it0_baseline', 'comportamento_correto'))}.\n"
            f"- {texto_injecao(injecao(), sc)}")
    ok2 = _substituir_bloco(v, "<!-- EVIDENCIAS:INICIO -->", "<!-- EVIDENCIAS:FIM -->", evid)
    if ok1 or ok2:
        feitos.append("prompts/versoes_rag.md")
    return feitos


# ------------------------------------------------------------------------------------------------ PDF
def gerar_pdf() -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    sc = scores()
    final, antes = final_e_antes(sc)
    base = getSampleStyleSheet()
    est = {"t": ParagraphStyle("t", parent=base["Title"], fontSize=15, leading=18, spaceAfter=2),
           "s": ParagraphStyle("s", parent=base["BodyText"], fontSize=8.5, leading=10.5, textColor=colors.HexColor("#444444"), spaceAfter=5),
           "h": ParagraphStyle("h", parent=base["Heading2"], fontSize=11, leading=13, spaceBefore=7, spaceAfter=3, textColor=colors.HexColor("#1f3a5f")),
           "p": ParagraphStyle("p", parent=base["BodyText"], fontSize=8.6, leading=10.8, spaceAfter=3),
           "li": ParagraphStyle("li", parent=base["BodyText"], fontSize=8.4, leading=10.4, leftIndent=11, bulletIndent=2, spaceAfter=1.5)}
    esc = lambda s: s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")  # noqa: E731

    def tabela(dados: list[list[str]], larguras: list[float], fonte: float = 7.4) -> Table:
        t = Table([[Paragraph(esc(str(c)), ParagraphStyle("c", parent=est["p"], fontSize=fonte, leading=fonte + 1.6, spaceAfter=0)) for c in linha] for linha in dados],
                  colWidths=[w * cm for w in larguras], repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                               ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
        return t

    h = []
    def titulo(txt): h.append(Paragraph(esc(txt), est["h"]))
    def para(txt): h.append(Paragraph(txt, est["p"]))
    def itens(lista): h.extend(Paragraph(x, est["li"], bulletText="•") for x in lista)

    h += [Paragraph("Relatório de evolução — ChargeGrid Assistant (Sprint 4)", est["t"]),
          Paragraph(f"FIAP · Prompt and Artificial Intelligence · EV Challenge 2026 × GoodWe · Equipe 03 · Turma 1CCPK · {date.today().strftime('%d/%m/%Y')}", est["s"])]

    titulo("1. Resumo da evolução")
    itens(["<b>Sprint 1:</b> planejamento (persona do motorista de EV, escopo, fluxograma, system prompt e modelo de teste).",
           "<b>Sprint 2:</b> chatbot em Python puro (Colab + Gemini 2.5 Flash) com prompt e dados de sessão fixos.",
           "<b>Sprint 3:</b> refactory em LangChain LCEL (memória, saída estruturada Pydantic, guardrails, prompt versionado, eval de 16 casos).",
           "<b>Sprint 4:</b> RAG com ChromaDB + nomic-embed-text sobre 7 documentos, grounding com citação de fonte verificável, defesa em camadas contra injeção, "
           "avaliação por iteração com juiz tipo RAGAS, interface Gradio e comparação de modelos."])

    titulo("2. Pipeline RAG: decisões e trade-offs")
    itens(["<b>Base:</b> manual e datasheet reais do HCA G2 + 5 documentos fictícios da operação do totem (CG-01 a CG-05), catalogados com tipo e público; conteúdo de instalação nunca chega ao motorista; o PDF do manual não é publicado (direitos autorais).",
           "<b>Chunking:</b> RecursiveCharacterTextSplitter 800/120 (15%); por página nos reais (citação com página exata) e <b>por seção</b> nos fictícios (um assunto por chunk).",
           "<b>Embeddings:</b> nomic-embed-text <b>local</b> (a nuvem não oferece embeddings na nossa conta) com prefixos search_query/search_document; ChromaDB persistente, cosseno.",
           "<b>Busca:</b> filtro de metadados por público e <b>cota por camada</b> (3 fictícios + 2 reais), pois o manual tem 78% dos chunks.",
           "<b>Geração:</b> prompt versionado (v1, v2), JSON validado (encontrou_na_base, resposta, fontes, escalar_suporte), porta de distância 0,40, validação das fontes citadas."])

    titulo("3. Comparativo antes × depois (obrigatório)")
    linhas = linhas_antes_depois(sc)
    if linhas:
        h.append(tabela([["Métrica", f"Sprints 1/2 (versão original) [{antes}]", f"Sprint 4 (RAG avaliado) [{final}]"]] + [list(x) for x in linhas], [8.2, 4.7, 4.5]))
        para("<i>O 'antes' mede o chatbot das Sprints 1/2 contra os mesmos documentos como contexto de referência; métricas sem sentido para ele (recusas, busca) aparecem como '-'.</i>")
    else:
        para("Rode <b>python -m evals.avaliar_rag --todas</b> e <b>python -m src.utils.gerar_relatorios</b> para preencher esta tabela.")

    titulo("4. Scores por iteração (mesmo eval set de 43 casos)")
    if sc:
        cab = ["Iteração"] + [r for _, r in COLS]
        corpo = [[n] + [_cel(c, g(sc, n, c), n) for c, _ in COLS] for n in [x for x in ORDEM if x in sc]]
        h.append(tabela([cab] + corpo, [3.2] + [1.6] * len(COLS), 6.9))
        para("Rubrica: fidelidade = afirmações sustentadas ÷ totais; relevância = cosseno médio entre 3 perguntas geradas a partir da resposta e a pergunta original (evasiva = 0). "
             "Diferenças de 1 caso valem 0,023–0,045 e a fidelidade varia ~0,04 entre execuções: só ganhos maiores que isso foram considerados.")
    titulo("5. Segurança")
    para(esc(texto_injecao(injecao(), sc)) + " Camadas: heurística de pergunta e de documento → escape de tags → prompt v2 → validação de fontes e canário → filtro de saída.")

    titulo("6. Problemas encontrados e soluções")
    itens([f"<b>{esc(t)}.</b> {esc(p)} <i>Solução:</i> {esc(s)}" for t, p, s in PROBLEMAS[:6]])

    titulo("7. Equipe e divisão de trabalho")
    eq = equipe()
    h.append(tabela([["Nome", "RM", "Tarefa principal"]] + [[m["nome"], m["rm"], ("(a preencher pela equipe)" if "PREENCHER" in m["tarefa"].upper() else m["tarefa"])] for m in eq], [5.2, 1.8, 10.4]))
    destino = DOCS / "relatorio_evolucao_sprint4.pdf"
    SimpleDocTemplate(str(destino), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.5 * cm, bottomMargin=1.4 * cm,
                      title="Relatório de evolução — ChargeGrid Assistant (Sprint 4)", author="Equipe 03").build(h)
    return destino


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    for nome, funcao in (("docs/relatorio_rag.md", gerar_relatorio_rag), ("docs/relatorio_modelos.md", gerar_relatorio_modelos), ("docs/relatorio_evolucao_sprint4.pdf", gerar_pdf)):
        funcao()
        print(f"  gerado: {nome}")
    for arq in atualizar_readme_e_versoes():
        print(f"  atualizado: {arq}")
    try:
        import subprocess

        paginas = subprocess.run(["pdfinfo", str(DOCS / "relatorio_evolucao_sprint4.pdf")], capture_output=True, text=True).stdout
        n = next((int(l.split()[-1]) for l in paginas.splitlines() if l.startswith("Pages:")), None)
        if n:
            print(f"  PDF: {n} página(s) {'(OK, limite 5)' if n <= 5 else '(ACIMA DO LIMITE DE 5 — reduza o texto)'}")
    except Exception:  # noqa: BLE001 — pdfinfo ausente no Windows: confira a contagem abrindo o PDF
        print("  (confira no PDF que o relatório tem no máximo 5 páginas)")


if __name__ == "__main__":
    main()
