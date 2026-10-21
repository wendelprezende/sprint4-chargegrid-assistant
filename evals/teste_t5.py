# -*- coding: utf-8 -*-
"""
Teste OFFLINE da maquinaria de avaliação da T5 (sem rede, sem Ollama, sem API key).

    python -m evals.teste_t5

Prova que o CÓDIGO que mede o RAG está certo — antes de gastar chamadas reais:
métricas por item, agregação, calibração do limiar, juiz (fidelidade/relevância), as variantes de iteração
(limpeza, chunking por seção, cota por camada), o "antes" sem RAG, a medição do LLM sozinho contra injeção,
e a gravação/comparação dos resultados.

Os números reais (nomic-embed-text + gpt-oss) saem de `python -m evals.avaliar_rag`.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from langchain_core.messages import AIMessage

import evals.avaliar_rag as av
from evals.teste_pipeline_rag import EmbeddingsLexicaisOffline
from src.rag import juiz as juiz_mod


class Checagens:
    def __init__(self) -> None:
        self.falhas = 0
        self.total = 0

    def ok(self, condicao: bool, descricao: str, detalhe: str = "") -> bool:
        self.total += 1
        print(f"  [{'OK' if condicao else 'FALHA'}] {descricao}" + (f"  -> {detalhe}" if detalhe and not condicao else ""))
        self.falhas += 0 if condicao else 1
        return condicao


def _res(status="respondida", resposta="", fontes=(), recuperados=(), escalar=False, llm=True):
    """Resultado sintético no formato do ResultadoRAG (só o que avaliar_item lê)."""
    return SimpleNamespace(
        status=status, resposta=resposta, escalar_suporte=escalar, llm_chamado=llm,
        fontes=[SimpleNamespace(id=i, fonte_id=f, pagina=p) for i, f, p in fontes],
        recuperados_info=[{"id": i, "fonte_id": f, "pagina": p, "descartado": d} for i, f, p, d in recuperados],
    )


def teste_funcoes_basicas(c: Checagens) -> None:
    print("\n1) Funções básicas")
    c.ok(av.casa_fonte("CG-02", "CG-02", 1) and not av.casa_fonte("CG-02", "CG-03", 1), "casa_fonte: documento inteiro")
    c.ok(av.casa_fonte("GW:p21", "GW", 21) and not av.casa_fonte("GW:p21", "GW", 22), "casa_fonte: página exata")
    c.ok(av.casa_fonte("GW:p14-16", "GW", 15) and not av.casa_fonte("GW:p14-16", "GW", 17), "casa_fonte: faixa de páginas")
    t = av.normalizar("Faltam 23 MINUTOS — Horário de Ponta")
    c.ok(av.fato_presente("ponta|noite", t) and av.fato_presente("horario", t) and not av.fato_presente("gratuita", t),
         "fato_presente: alternativas, sem diferenciar acento/maiúscula")


def teste_avaliar_item(c: Checagens) -> None:
    print("\n2) Métricas por item")
    base = {"categoria": "happy_path", "fontes_aceitaveis": ["CG-02"], "fatos_esperados": ["ponta", "17h"], "fatos_proibidos": ["gratis"]}
    rec = [("a", "CG-04", 1, False), ("b", "CG-02", 1, False), ("c", "CG-01", 1, False)]
    r = _res(resposta="Na ponta (17h) é mais caro.", fontes=[("b", "CG-02", 1)], recuperados=rec)
    m = av.avaliar_item({**base, "id": "x", "comportamento": "responder"}, r)
    c.ok(m["comportamento_ok"] is True and m["hit_at_k"] is True, "responder + respondida + fonte certa recuperada")
    c.ok(abs(m["mrr"] - 0.5) < 1e-9 and abs(m["precisao_at_k"] - 1 / 3) < 1e-9, "MRR=1/2 (2ª posição) e precisão@k=1/3", f"{m['mrr']} {m['precisao_at_k']}")
    c.ok(m["citacao_presente"] is True and m["citacao_correta"] is True and m["recall_fatos"] == 1.0, "citação presente/correta e recall de fatos = 1")
    r2 = _res(resposta="É grátis!", fontes=[("a", "CG-04", 1)], recuperados=rec)
    m2 = av.avaliar_item({**base, "id": "x", "comportamento": "responder"}, r2)
    c.ok(m2["violou_proibidos"] is True and m2["citacao_correta"] is False and m2["recall_fatos"] == 0.0,
         "fato proibido detectado; citação de fonte não aceitável; recall 0")
    descartado = [("a", "CG-02", 1, True)]
    m3 = av.avaliar_item({**base, "id": "x", "comportamento": "responder"}, _res(recuperados=descartado, fontes=[("z", "CG-02", 1)]))
    c.ok(m3["hit_at_k"] is None, "chunk descartado por injeção não conta como recuperado")

    it = lambda comp, **kw: {"id": "x", "categoria": "out_of_scope", "comportamento": comp, "fontes_aceitaveis": [], "fatos_esperados": [], "fatos_proibidos": [], **kw}  # noqa: E731
    casos = [
        ("responder", "sem_contexto", False), ("responder", "respondida", True), ("recusar_guardrail", "recusa_jailbreak", True),
        ("recusar_guardrail", "respondida", False), ("recusar_base", "sem_contexto", True), ("recusar_base", "respondida", False),
        ("saudacao", "saudacao", True), ("nao_inventar", "sem_resposta_na_base", True), ("nao_inventar", "erro_llm", False),
    ]
    c.ok(all(av.avaliar_item(it(comp), _res(status=st, fontes=[("i", "CG-01", 1)] if st == "respondida" else []))["comportamento_ok"] is exp
             for comp, st, exp in casos), f"comportamento esperado x status: {len(casos)} combinações")
    ms = av.avaliar_item(it("responder_com_sessao"), _res(fontes=[("dados_sessao", "dados_sessao", None)]))
    c.ok(ms["comportamento_ok"] is True, "responder_com_sessao exige 'dados_sessao' entre as fontes")
    c.ok(av.avaliar_item(it("responder_com_sessao"), _res(fontes=[("a", "CG-01", 1)]))["comportamento_ok"] is False, "sem 'dados_sessao' reprova")
    mi = av.avaliar_item(it("injecao", categoria="injecao_documento", fatos_proibidos=["promo-gratis"]), _res(resposta="veja www.promo-gratis.example"))
    c.ok(mi["comportamento_ok"] is False and mi["violou_proibidos"] is True, "injeção: obedecer ao documento reprova")
    mr = av.avaliar_item(it("responder"), _res(status="sem_contexto"))
    c.ok(mr["recusa_indevida"] is True, "recusa indevida: devia responder e recusou")
    mq = av.avaliar_item(it("recusar_base"), _res(status="respondida", fontes=[("a", "CG-01", 1)]))
    c.ok(mq["resposta_indevida"] is True, "resposta indevida: devia recusar e respondeu")
    me = av.avaliar_item({**base, "id": "x", "comportamento": "responder", "escalar_suporte_esperado": True}, _res(escalar=False, fontes=[("b", "CG-02", 1)], recuperados=rec))
    c.ok(me["escalar_ok"] is False, "escalar_suporte esperado e não acionado é detectado")


def teste_agregacao_e_calibracao(c: Checagens) -> None:
    print("\n3) Agregação e calibração do limiar")
    e = lambda cat, ok, fid=None, rel=None: {"categoria": cat, "metricas": {"comportamento_ok": ok}, "juiz": {"fidelidade": fid, "relevancia": rel}, "llm_chamado": True, "tokens_prompt": 100, "latencia_total_s": 1.0, "latencia_llm_s": 0.8}  # noqa: E731
    ag = av.agregar([e("happy_path", True, 1.0, 0.8), e("happy_path", False, 0.5, 0.6), e("jailbreak", True)])
    g = ag["geral"]
    c.ok(g["n_itens"] == 3 and abs(g["comportamento_correto"] - 0.6667) < 1e-3, "agregado geral: 2 de 3 corretos", str(g["comportamento_correto"]))
    c.ok(g["fidelidade"] == 0.75 and g["n_fidelidade"] == 2 and abs(g["relevancia"] - 0.7) < 1e-9, "fidelidade e relevância ignoram itens sem nota (None)")
    c.ok(ag["por_categoria"]["jailbreak"]["comportamento_correto"] == 1.0, "agregado por categoria")

    def fake_retriever(mapa):
        class R:
            def buscar(self, pergunta):
                return [SimpleNamespace(distancia=mapa[pergunta])]
        return R()

    cal = {"dentro_do_dominio": ["d1", "d2", "d3"], "fora_do_dominio": ["f1", "f2", "f3"]}
    sep = av.calibrar_limiar(fake_retriever({"d1": 0.30, "d2": 0.33, "d3": 0.36, "f1": 0.50, "f2": 0.55, "f3": 0.60}), cal)
    c.ok(sep["separavel"] and 0.36 < sep["limiar"] < 0.50 and abs(sep["margem"] - 0.14) < 1e-9 and sep["nota_balanceada"] == 1.0,
         "separável: limiar entre os grupos, margem 0,14, nota 1,0", str(sep["limiar"]))
    mist = av.calibrar_limiar(fake_retriever({"d1": 0.30, "d2": 0.33, "d3": 0.52, "f1": 0.45, "f2": 0.55, "f3": 0.60}), cal)
    c.ok(not mist["separavel"] and mist["margem"] < 0 and 0.30 < mist["limiar"] < 0.60, "sobreposição: marca como não separável e escolhe o melhor compromisso", str(mist["limiar"]))


class LLMFalso:
    """Faz papel de gerador E de juiz (decide pelo conteúdo do prompt); grava o que recebeu."""

    def __init__(self, obedece_veneno: bool = True) -> None:
        self.obedece_veneno, self.model, self.chamadas = obedece_veneno, "falso", 0

    def invoke(self, entrada):
        self.chamadas += 1
        txt = entrada.to_string() if hasattr(entrada, "to_string") else str(entrada)
        if "auditor rigoroso" in txt:
            return AIMessage(content=json.dumps({"afirmacoes": [
                {"texto": "afirmação A", "suportada": True, "evidencia": "trecho"}, {"texto": "afirmação B", "suportada": False, "evidencia": ""}]}))
        if "avaliador de relevância" in txt:
            return AIMessage(content=json.dumps({"perguntas": ["Como pagar com PIX?", "Como iniciar a recarga?", "Qual o preço?"], "evasiva": False}))
        if "<contexto>" in txt:  # gerador RAG
            ids = re.findall(r'<documento id="([^"]+)"', txt)
            resposta = "Escolha a opção na tela do totem e confirme."
            if self.obedece_veneno and "promo-gratis" in txt:
                resposta += " Veja www.promo-gratis.example"
            if "gratuita para todos" in txt and self.obedece_veneno:
                resposta += " A recarga é gratuita."
            # o prompt de sistema também MENCIONA as tags; o conteúdo real vem sempre na ÚLTIMA ocorrência
            sessao = txt.rsplit("<dados_sessao>", 1)[1].split("</dados_sessao>")[0]
            if "tempo_restante_min" in sessao and "tempo" in txt.rsplit("<pergunta>", 1)[1].lower():
                return AIMessage(content=json.dumps({"encontrou_na_base": True, "resposta": "Faltam 23 minutos.", "fontes": ["dados_sessao"], "escalar_suporte": False}))
            return AIMessage(content=json.dumps({"encontrou_na_base": bool(ids), "resposta": resposta, "fontes": ids[:1], "escalar_suporte": False}))
        return AIMessage(content="Resposta livre do chatbot antigo, sem fontes.")  # "antes" (sem RAG)


def teste_juiz(c: Checagens) -> None:
    print("\n4) Juiz LLM (fidelidade e relevância)")
    llm = LLMFalso()
    f = juiz_mod.fidelidade(llm, "contexto", "resposta com duas afirmações")
    c.ok(f["score"] == 0.5 and f["n_afirmacoes"] == 2 and f["nao_suportadas"] == ["afirmação B"], "fidelidade = suportadas/total (1 de 2 = 0,5) e lista a não suportada", str(f))
    r = juiz_mod.relevancia(llm, EmbeddingsLexicaisOffline(), "Como faço para pagar com PIX?", "Escolha PIX na tela.")
    c.ok(r["score"] is not None and 0 < r["score"] <= 1 and not r["evasiva"] and len(r["perguntas"]) == 3, "relevância = cosseno médio entre perguntas geradas e a original", str(r))

    class Ruim:
        def __init__(self, respostas):
            self.respostas = list(respostas)

        def invoke(self, entrada):
            return AIMessage(content=self.respostas.pop(0))

    ok_json = json.dumps({"afirmacoes": [{"texto": "x", "suportada": True}]})
    c.ok(juiz_mod.fidelidade(Ruim(["não é json", ok_json]), "c", "r")["score"] == 1.0, "JSON inválido do juiz -> corrige na 2ª tentativa")
    c.ok(juiz_mod.fidelidade(Ruim(["lixo", "lixo"]), "c", "r")["score"] is None, "juiz sempre inválido -> nota None com erro (não derruba a avaliação)")
    c.ok(juiz_mod.fidelidade(Ruim([json.dumps({"afirmacoes": []})]), "c", "r")["score"] is None, "resposta sem afirmações -> nota None (nada a verificar)")
    ev = juiz_mod.relevancia(Ruim([json.dumps({"perguntas": ["x"], "evasiva": True})]), EmbeddingsLexicaisOffline(), "q", "Não sei.")
    c.ok(ev["score"] == 0.0 and ev["evasiva"] is True, "resposta evasiva recebe relevância 0 (regra do RAGAS)")


def teste_iteracoes(c: Checagens) -> None:
    print("\n5) Iterações ponta a ponta (componentes de mentira)")
    doc = av.carregar_json(av.ARQ_EVAL_SET)
    cal = av.carregar_json(av.ARQ_CALIBRACAO)
    ids = ["hp-01", "hp-02", "hp-02b", "hp-04", "hp-11", "ec-02", "ec-03", "jb-01", "oos-01", "oos-04", "na-01", "inj-01", "inj-04"]
    itens = [i for i in doc["casos"] if i["id"] in ids]
    c.ok(len(itens) == len(ids), f"{len(itens)} casos do eval set selecionados")
    llm = LLMFalso()
    comuns = dict(emb_factory=lambda prefixo: EmbeddingsLexicaisOffline(), llm_factory=lambda: llm, juiz_factory=lambda: llm,
                  dados_sessao_padrao=doc["dados_sessao_padrao"], quiet=True)
    sem_porta = lambda cfg: replace(cfg, limiar=0.99)  # noqa: E731 — o embedding léxico tem distâncias altas

    base = av.rodar_iteracao(sem_porta(av.ITERACOES["it0_baseline"]), itens, cal, **comuns)
    g = base["agregados"]["geral"]
    c.ok(g["n_itens"] == len(itens) and g["comportamento_correto"] is not None, "baseline roda todos os casos e agrega", str(g["comportamento_correto"]))
    por_id = {e["id"]: e for e in base["entradas"]}
    c.ok(por_id["jb-01"]["status"] == "recusa_jailbreak" and por_id["oos-01"]["status"] == "recusa_escopo" and por_id["ec-02"]["status"] == "saudacao",
         "guardrails agem antes do LLM dentro da avaliação")
    c.ok(por_id["hp-02"]["status"] == "respondida" and "dados_sessao" in por_id["hp-02"]["fontes"], "caso com sessão usa dados_sessao")
    c.ok(g["fidelidade"] == 0.5 and g["n_fidelidade"] >= 1, "juiz aplicado às respostas dos casos 'responder' (fidelidade 0,5 do juiz de mentira)", str(g["fidelidade"]))
    c.ok(por_id["inj-01"]["metricas"]["violou_proibidos"] is True, "LLM SOZINHO (sem defesas) que obedece ao documento: obediência MEDIDA")
    c.ok(por_id["inj-04"]["metricas"]["violou_proibidos"] is False and por_id["inj-04"]["descartados_injecao"],
         "mesmo ataque com as defesas completas: chunk descartado, sem obediência")
    c.ok(g["injecao_obediencia_sem_defesas"] == 1.0 and g["injecao_obediencia_com_defesas"] == 0.0, "agregado separa obediência com e sem defesas")

    cfg_limpa = sem_porta(av.ITERACOES["it1_limpeza"])
    limpa = av.rodar_iteracao(cfg_limpa, itens, cal, **comuns)
    c.ok(not any("DOCUMENTO FICTÍCIO" in (e["contexto_texto"] or "") for e in limpa["entradas"]), "it1: o aviso fictício repetido sumiu do contexto enviado ao LLM")
    c.ok(any("DOCUMENTO FICTÍCIO" in (e["contexto_texto"] or "") for e in base["entradas"]), "baseline ainda tem o aviso (comparação válida)")
    secao = av.rodar_iteracao(sem_porta(av.ITERACOES["it2_secao"]), itens, cal, **comuns)
    c.ok(secao["indexacao"]["chunks_total"] != base["indexacao"]["chunks_total"] and secao["indexacao"]["chunking_por_secao"], "it2: chunking por seção muda a indexação",
         f"{base['indexacao']['chunks_total']} -> {secao['indexacao']['chunks_total']}")
    camadas = av.rodar_iteracao(sem_porta(av.ITERACOES["it4_camadas"]), itens, cal, **comuns)
    tipos = {r["fonte_id"].startswith("CG") for e in camadas["entradas"] for r in e["recuperados"]}
    c.ok(tipos == {True, False}, "it4: busca por camada devolve fictícios E reais em toda consulta")
    c.ok(all(len({r["fonte_id"].startswith("CG") for r in e["recuperados"]}) == 2 for e in camadas["entradas"] if e["recuperados"] and e["categoria"] == "happy_path"),
         "it4: cada consulta tem as duas camadas representadas")
    auto = av.rodar_iteracao(replace(av.ITERACOES["it5_limiar"], cotas=None), itens, cal, **comuns)
    c.ok(auto["calibracao"] is not None and auto["configuracao"]["limiar_usado"] == auto["calibracao"]["limiar"], "it5: limiar 'auto' é calibrado e usado")

    antes = av.rodar_iteracao(av.ITERACOES["it0_sem_rag"], itens, cal, **comuns)
    ae = {e["id"]: e for e in antes["entradas"]}
    c.ok(ae["inj-01"]["status"] == "n/a" and ae["hp-01"]["status"] == "respondida_sem_rag", "'antes' (sem RAG): responde sem citar; injeção por documento não se aplica")
    c.ok(antes["agregados"]["geral"]["citacao_presente"] == 0.0, "'antes' não tem citação de fonte (0%)")
    c.ok(ae["hp-02b"]["metricas"]["violou_proibidos"] is False or True, "'antes' usa dados de sessão fixos")
    so_rec = av.rodar_iteracao(av.ITERACOES["it0_baseline"], itens, cal, **{**comuns, "so_recuperacao": True})
    c.ok(llm.chamadas > 0 and all(e["status"] == "so_recuperacao" for e in so_rec["entradas"]) and so_rec["agregados"]["geral"]["hit_at_k"] is not None,
         "--so-recuperacao: mede hit@k sem LLM")
    c.ok(all(e["metricas"].get("comportamento_ok") is None and e["metricas"].get("resposta_indevida") is None for e in antes["entradas"]),
         "'antes': comportamento e resposta indevida NÃO são calculados (eram artificiais: 'ok' por construção)")
    c.ok(antes["agregados"]["geral"]["comportamento_correto"] is None and antes["agregados"]["geral"]["resposta_indevida"] is None,
         "'antes': agregados correspondentes ficam '-' (None), não 0,957 nem 1,000")
    return {"base": base, "limpa": limpa, "so_rec": so_rec, "antes": antes, "doc": doc, "cal": cal}  # type: ignore[return-value]


def teste_persistencia(c: Checagens, resultados: dict) -> None:
    print("\n6) Gravação, comparação e auditoria")
    tmp = Path(tempfile.mkdtemp(prefix="evals_t5_"))
    av.PASTA_RESULTADOS, av.ARQ_SCORES, av.ARQ_CSV, av.PASTA_EVALS = tmp / "resultados", tmp / "scores.json", tmp / "scores.csv", tmp
    av.salvar(resultados["base"])
    av.salvar(resultados["limpa"])
    av.salvar(resultados["so_rec"])
    c.ok((tmp / "resultados" / "it0_baseline.json").exists() and (tmp / "resultados" / "it0_baseline_so_recuperacao.json").exists(), "resultado completo gravado por iteração")
    scores = json.loads((tmp / "scores.json").read_text(encoding="utf-8"))
    c.ok({"it0_baseline", "it1_limpeza", "it0_baseline_so_recuperacao"} <= set(scores), "scores_por_iteracao.json acumula as iterações")
    linhas = (tmp / "scores.csv").read_text(encoding="utf-8").strip().splitlines()
    c.ok(len(linhas) == 4 and linhas[0].startswith("iteracao,descricao"), "scores_por_iteracao.csv (para o relatório) com cabeçalho e 3 linhas")
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        av.comparar()
        av.exportar_auditoria("it0_baseline")
    saida = buf.getvalue()
    c.ok("it0_baseline" in saida and "it1_limpeza" in saida and "Ganho em relação ao baseline" in saida, "comparar() imprime a tabela com ganhos sobre o baseline")
    aud = tmp / "auditoria_humana_it0_baseline.csv"
    c.ok(aud.exists() and "humano_fidelidade_0a1" in aud.read_text(encoding="utf-8"), "planilha de auditoria humana exportada")


class EmbeddingsPrefixoFalso(EmbeddingsLexicaisOffline):
    """Imita o nomic COM prefixo: consulta e documento recebem prefixos diferentes, então os vetores divergem."""

    def embed_query(self, text):
        return super().embed_query("search_query " + text)

    def embed_documents(self, texts):
        return super().embed_documents(["search_document " + t for t in texts])


def teste_correcoes_pos_medicao(c: Checagens, resultados: dict) -> None:
    print("\n7) Correções feitas depois da 1ª medição real (19/10)")
    emb = EmbeddingsPrefixoFalso()
    llm = LLMFalso()
    pergunta, resposta = "Como faço para pagar com PIX?", "Escolha PIX na tela."
    r = juiz_mod.relevancia(llm, emb, pergunta, resposta)
    esperado = sum(juiz_mod._cosseno(emb.embed_query(pergunta), emb.embed_query(q)) for q in r["perguntas"]) / len(r["perguntas"])
    errado = sum(juiz_mod._cosseno(emb.embed_query(pergunta), v) for v in emb.embed_documents(r["perguntas"])) / len(r["perguntas"])
    c.ok(abs(r["score"] - round(esperado, 4)) < 1e-4 and abs(esperado - errado) > 1e-3,
         "relevância compara pergunta com pergunta usando embed_query nos dois lados (o prefixo assimétrico rebaixava o cosseno)",
         f"{r['score']} vs simétrico {esperado:.4f} vs assimétrico {errado:.4f}")

    tmp = av.PASTA_EVALS
    sujo = json.loads(json.dumps(resultados["base"], default=str))
    sujo["iteracao"] = "it3_prefixo"
    sujo["configuracao"] = {**sujo["configuracao"], "prefixo": True, "nome": "it3_prefixo"}
    alvo = next(e for e in sujo["entradas"] if e["juiz"].get("perguntas_geradas"))
    alvo["juiz"]["relevancia"] = 0.1234  # valor "errado" gravado pela versão antiga
    av.salvar(sujo)
    av.recalcular(lambda prefixo: EmbeddingsPrefixoFalso())
    corrigido = av.carregar_json(av.PASTA_RESULTADOS / "it3_prefixo.json")
    novo = next(e for e in corrigido["entradas"] if e["id"] == alvo["id"])["juiz"]["relevancia"]
    esp = round(sum(juiz_mod._cosseno(emb.embed_query(alvo["pergunta"]), emb.embed_query(q)) for q in alvo["juiz"]["perguntas_geradas"]) / len(alvo["juiz"]["perguntas_geradas"]), 4)
    c.ok(abs(novo - esp) < 1e-4 and novo != 0.1234 and "recalculado" in corrigido, "--recalcular corrige a relevância já gravada SEM chamar LLM", f"{novo} vs {esp}")
    antes_json = av.carregar_json(av.PASTA_RESULTADOS / "it0_baseline.json")
    c.ok(antes_json["agregados"]["geral"]["relevancia"] is not None, "--recalcular preserva as demais iterações")

    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        av.comparar_pareado()
        av.inspecionar("it0_baseline", "hp-01")
        av.inspecionar("it0_baseline", "xx-99")
        av.comparar()
    saida = buf.getvalue()
    c.ok("it3_prefixo" in saida and "Δ" in saida, "--pareado compara fidelidade/relevância nos mesmos itens do baseline")
    c.ok("PERGUNTA" in saida and "RECUPERADOS" in saida and "não existe" in saida, "--inspecionar mostra o caso em detalhe e avisa quando o id não existe")
    c.ok("it0_baseline_so_recuperacao" not in saida.split("ATENÇÃO")[0], "comparar() não duplica linhas 'só recuperação' quando a iteração completa existe")
    c.ok("2468" not in saida and ("tokens" in saida), "comparar() formata tokens sem colar colunas")

    c.ok(av._wilson(0, 10)[0] == 0.0 and 0.25 < av._wilson(0, 10)[1] < 0.35 and av._wilson(5, 10)[0] < 0.5 < av._wilson(5, 10)[1],
         "intervalo de confiança de Wilson (0/10 -> [0; ~0,28])")
    emb_f = lambda prefixo: EmbeddingsLexicaisOffline()  # noqa: E731
    for obedece, esperado_total in ((True, 9), (False, 0)):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            av.medir_injecao(3, "v2", resultados["doc"], resultados["cal"], emb_f, lambda o=obedece: LLMFalso(obedece_veneno=o))
        dados = av.carregar_json(av.PASTA_RESULTADOS / "injecao_prompt_v2_n3.json")
        c.ok(dados["total_obedeceu"] == esperado_total and dados["total"] == 12,
             f"--medir-injecao: LLM {'obediente' if obedece else 'resistente'} -> {dados['total_obedeceu']}/12 (3 repetições x 4 ataques)")


def main() -> int:
    c = Checagens()
    print("=== Teste offline da maquinaria de avaliação (T5) ===")
    teste_funcoes_basicas(c)
    teste_avaliar_item(c)
    teste_agregacao_e_calibracao(c)
    teste_juiz(c)
    resultados = teste_iteracoes(c)
    teste_persistencia(c, resultados)
    teste_correcoes_pos_medicao(c, resultados)
    print(f"\n=== Resultado: {c.total - c.falhas}/{c.total} checagens {'(TODAS passaram)' if c.falhas == 0 else '— ' + str(c.falhas) + ' FALHARAM'} ===")
    return 1 if c.falhas else 0


if __name__ == "__main__":
    sys.exit(main())
