# -*- coding: utf-8 -*-
"""
Verificação funcional do projeto da T1 até a T3 — UM comando, relatório único.

    python -m evals.verificar_ate_t3                # completo (exige Ollama local rodando)
    python -m evals.verificar_ate_t3 --sem-ollama   # só o que não depende de rede/Ollama

SEÇÕES
  1 Ambiente              Python, bibliotecas, .env
  2 Estrutura             arquivos que T1, T2 e T3 devem ter criado
  3 Segurança do Git      .gitignore, arquivos proibidos versionados, chave vazada no histórico
  4 Base de conhecimento  catálogo, PDFs, páginas, aviso de documento fictício
  5 Pipeline (offline)    carga, chunking, persistência e filtro, sem rede
  6 Chain da Sprint 3     guardrails + memória com modelo de mentira (sem rede)
  7 Ollama real           embedding local (+ sanidade semântica) e os dois chats
  8 Pipeline REAL         indexa com o nomic-embed-text e mede a recuperação (hit@k, distâncias)

LEGENDA   [OK] passou | [FALHA] corrigir antes de seguir | [AVISO] atenção, não bloqueia | [PULADO]

Os limites de aprovação da seção 8 (7 de 9 = OK) são CRITÉRIOS DE TRABALHO nossos,
não regras do guia da Sprint 4. A nota de qualidade de verdade vem da avaliação (T5).

Resultado detalhado: evals/verificacao_t3_resultado.json
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import warnings

warnings.filterwarnings("ignore", message=r".*langchain-community.*sunset.*")

from src import config

RAIZ = config.RAIZ
ESTRUTURA_OBRIGATORIA = [
    # T1
    ".gitignore", ".env.example", "requirements.txt", "README.md", "src/config.py",
    "src/utils/verificar_ollama.py", "src/chain/builder.py", "src/guardrails/scope_validator.py",
    "prompts/system_prompt_v2.md", "evals/eval_set.json", "evals/sprint3_results.json",
    # T2
    "data/knowledge_base/catalogo.json", "data/knowledge_base/README.md", "src/utils/gerar_documentos_ficticios.py",
    "data/knowledge_base/GW_HCAG2_DatasheetPT.pdf",
    *[f"data/knowledge_base/{n}" for n in (
        "CG-01_Guia_do_Motorista.pdf", "CG-02_Tarifas_e_Pagamentos.pdf", "CG-03_Solucao_de_Problemas_e_Suporte.pdf",
        "CG-04_FAQ_Operacional.pdf", "CG-05_Termos_de_Uso_e_Limites_do_Assistente.pdf")],
    # T3
    "src/rag/loader.py", "src/rag/chunking.py", "src/rag/embeddings.py", "src/rag/vector_store.py",
    "src/rag/retriever.py", "src/rag/indexar.py", "evals/teste_pipeline_rag.py",
]
BIBLIOTECAS = {"langchain": "langchain", "langchain_ollama": "langchain-ollama", "langchain_chroma": "langchain-chroma",
               "chromadb": "chromadb", "pymupdf": "pymupdf", "gradio": "gradio", "tiktoken": "tiktoken",
               "pydantic": "pydantic", "dotenv": "python-dotenv", "reportlab": "reportlab"}
MANUAL = "GW_HCAG2_UserManualPT.pdf"


class Painel:
    """Acumula e imprime os resultados; guarda tudo para o JSON final."""

    def __init__(self) -> None:
        self.itens: list[dict] = []
        self.secao = ""

    def abrir(self, titulo: str) -> None:
        self.secao = titulo
        print(f"\n=== {titulo} ===")

    def registrar(self, status: str, descricao: str, detalhe: str = "") -> bool:
        self.itens.append({"secao": self.secao, "status": status, "descricao": descricao, "detalhe": detalhe})
        print(f"  [{status}] {descricao}" + (f"  -> {detalhe}" if detalhe else ""))
        return status == "OK"

    def ok(self, condicao: bool, descricao: str, detalhe: str = "", se_falhar: str = "FALHA") -> bool:
        return self.registrar("OK" if condicao else se_falhar, descricao, detalhe)

    def contar(self, status: str) -> int:
        return sum(1 for i in self.itens if i["status"] == status)


def _git(*args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None


def _executar_modulo(modulo: str, *args: str) -> subprocess.CompletedProcess:
    import os

    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONWARNINGS": "ignore"}
    return subprocess.run([sys.executable, "-W", "ignore", "-m", modulo, *args], cwd=RAIZ, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", env=env)


def _cosseno(a: list[float], b: list[float]) -> float:
    pe = sum(x * y for x, y in zip(a, b))
    return pe / ((math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0)


# ---------------------------------------------------------------------------------------------
def secao_ambiente(p: Painel) -> None:
    p.abrir("1) Ambiente")
    p.ok(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]} (mínimo 3.10)")
    em_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    p.ok(em_venv, "ambiente virtual ativo", "" if em_venv else "ative o .venv antes de rodar", se_falhar="AVISO")
    for modulo, pacote in BIBLIOTECAS.items():
        try:
            importlib.import_module(modulo)
            p.registrar("OK", pacote)
        except Exception as erro:  # noqa: BLE001
            p.registrar("FALHA", pacote, f"{type(erro).__name__}: rode pip install -r requirements.txt")
    env = RAIZ / ".env"
    p.ok(env.exists(), ".env existe na raiz do projeto")
    if env.exists():
        try:
            config.obter_api_key()
            p.registrar("OK", "OLLAMA_API_KEY definida (valor oculto)")
        except RuntimeError as erro:
            p.registrar("FALHA", "OLLAMA_API_KEY definida", str(erro)[:90])


def secao_estrutura(p: Painel) -> None:
    p.abrir("2) Estrutura do repositório (T1 + T2 + T3)")
    faltando = [c for c in ESTRUTURA_OBRIGATORIA if not (RAIZ / c).exists()]
    p.ok(not faltando, f"{len(ESTRUTURA_OBRIGATORIA) - len(faltando)}/{len(ESTRUTURA_OBRIGATORIA)} arquivos esperados presentes",
         "faltando: " + ", ".join(faltando) if faltando else "")
    p.ok((RAIZ / "data/knowledge_base" / MANUAL).exists(), f"manual do HCA G2 presente em data/knowledge_base/",
         "" if (RAIZ / "data/knowledge_base" / MANUAL).exists() else "baixe/copie o PDF (fica fora do GitHub de propósito)",
         se_falhar="AVISO")


def secao_git(p: Painel) -> None:
    p.abrir("3) Segurança do Git (a exposição de chave tem penalidade na entrega)")
    gi = (RAIZ / ".gitignore").read_text(encoding="utf-8") if (RAIZ / ".gitignore").exists() else ""
    for alvo in (".env", "chroma_db/", MANUAL):
        p.ok(alvo in gi, f".gitignore bloqueia {alvo}")
    dentro = _git("rev-parse", "--is-inside-work-tree")
    if dentro is None or dentro.returncode != 0:
        p.registrar("PULADO", "checagens de repositório Git", "git não encontrado ou pasta não é um repositório")
        return
    rastreados = _git("ls-files").stdout.splitlines()
    proibidos = [f for f in rastreados if f == ".env" or f.startswith("chroma_db/") or f.endswith(MANUAL)]
    p.ok(not proibidos, "nenhum arquivo proibido está versionado (.env, chroma_db/, manual)", ", ".join(proibidos))
    padrao = re.compile(r"OLLAMA_API_KEY\s*[=:]\s*[\"']?(?P<valor>[A-Za-z0-9_\-\.]{16,})")
    exemplo = re.compile(r"aqui|sua_|your|exemplo|example|xxx|placeholder|chave_de|teste", re.IGNORECASE)  # valores de documentação
    vazados = []
    for arq in rastreados:
        if arq == ".env.example" or not arq.endswith((".py", ".md", ".txt", ".json", ".ipynb", ".env", ".yml", ".yaml", ".toml")):
            continue
        try:
            texto = (RAIZ / arq).read_text(encoding="utf-8", errors="ignore")
            if any(not exemplo.search(m.group("valor")) for m in padrao.finditer(texto)):
                vazados.append(arq)
        except OSError:
            pass
    p.ok(not vazados, "nenhum arquivo versionado contém uma chave no formato OLLAMA_API_KEY=<valor>", ", ".join(vazados))
    try:
        chave = config.obter_api_key()
        hist = _git("log", "--all", "--oneline", "-S", chave)
        p.ok(hist.returncode == 0 and not hist.stdout.strip(),
             "sua chave NÃO aparece em nenhum commit do histórico",
             "REVOGUE a chave em ollama.com/settings/keys e crie outra" if hist.stdout.strip() else "")
    except RuntimeError:
        p.registrar("PULADO", "busca da chave no histórico", "sem chave no .env")
    # o próprio relatório desta verificação não conta como "alteração pendente"
    status = [l for l in _git("status", "--porcelain").stdout.splitlines() if not l.endswith("verificacao_t3_resultado.json")]
    ramo = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    p.ok(not status, f"árvore de trabalho limpa (branch atual: {ramo})",
         f"{len(status)} arquivo(s) com alterações não commitadas" if status else "", se_falhar="AVISO")


def secao_kb(p: Painel) -> None:
    p.abrir("4) Base de conhecimento")
    from langchain_community.document_loaders import PyMuPDFLoader

    try:
        cat = json.loads((config.PASTA_KB / "catalogo.json").read_text(encoding="utf-8"))
    except Exception as erro:  # noqa: BLE001
        p.registrar("FALHA", "catalogo.json válido", f"{type(erro).__name__}: {erro}")
        return
    docs = cat["documentos"]
    p.ok(len(docs) == 7, f"{len(docs)} documentos no catálogo (esperado 7)")
    for d in docs:
        caminho = config.PASTA_KB / d["arquivo"]
        if not caminho.exists():
            p.registrar("AVISO" if d["arquivo"] == MANUAL else "FALHA", f"{d['arquivo']} existe", "arquivo ausente")
            continue
        paginas = PyMuPDFLoader(str(caminho)).load()
        esperado = d.get("total_paginas")
        okpag = esperado is None or esperado == len(paginas)
        extra = ""
        if d["tipo"] == "ficticio_chargegrid":
            todas = all("DOCUMENTO FICTÍCIO" in pg.page_content for pg in paginas)
            okpag = okpag and todas
            extra = "aviso de documento fictício em todas as páginas" if todas else "SEM o aviso fictício em alguma página"
        p.ok(okpag, f"{d['arquivo']} ({len(paginas)} págs)", extra if not okpag or extra else "")


def secao_pipeline_offline(p: Painel) -> None:
    p.abrir("5) Pipeline RAG (offline, embedding de mentira — valida o encanamento)")
    r = _executar_modulo("evals.teste_pipeline_rag", "--offline")
    fim = [l for l in r.stdout.splitlines() if "Resultado" in l]
    falhas = [l.strip() for l in r.stdout.splitlines() if "[FALHA]" in l]
    p.ok(r.returncode == 0, "evals.teste_pipeline_rag --offline", (fim[-1].strip("= ") if fim else (r.stderr[-200:] or "sem saída")) if r.returncode == 0 else "; ".join(falhas) or r.stderr[-200:])


def secao_chain_sprint3(p: Painel) -> None:
    p.abrir("6) Chain da Sprint 3 (guardrails + memória, modelo de mentira)")
    r = _executar_modulo("src.chain.builder")
    from src.guardrails.moderation import MENSAGEM_RECUSA_JAILBREAK

    p.ok(r.returncode == 0, "demonstração offline da chain executa sem erro", r.stderr[-200:] if r.returncode else "")
    p.ok(MENSAGEM_RECUSA_JAILBREAK in r.stdout, "guardrail recusa a tentativa de jailbreak")
    from src.guardrails.scope_validator import validar_escopo

    p.ok(not validar_escopo("Vale a pena eu investir meu dinheiro em ações da bolsa?").dentro_do_escopo,
         "guardrail de escopo recusa pergunta financeira")
    p.ok(not validar_escopo("Posso mexer na fiação do quadro elétrico da minha casa?").dentro_do_escopo,
         "guardrail de escopo recusa pergunta de segurança elétrica")
    # Bug CONHECIDO da Sprint 3 (hp-03 do sprint3_results.json): a pergunta legítima de tarifação foi
    # bloqueada como "fora de escopo". A T4 corrige; até lá aparece como AVISO (e vira OK depois).
    p.ok(validar_escopo("Por que o preço agora está mais caro do que ontem de manhã?").dentro_do_escopo,
         "guardrail de escopo NÃO bloqueia pergunta legítima de tarifação",
         "bug conhecido da Sprint 3 (hp-03); será corrigido na T4", se_falhar="AVISO")
    p.ok(validar_escopo("Como eu começo a carregar meu carro aqui?").dentro_do_escopo,
         "guardrail de escopo aceita pergunta legítima de início de recarga")


def _embeddings_reais():
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=config.MODELO_EMBEDDING, base_url=config.OLLAMA_EMBED_BASE_URL,
                            client_kwargs={"headers": config.cabecalhos_ollama(config.OLLAMA_EMBED_BASE_URL)})


def secao_ollama(p: Painel) -> bool:
    """Devolve True se o embedding está funcional (pré-requisito da seção 8)."""
    p.abrir(f"7) Ollama real (embedding em {config.OLLAMA_EMBED_BASE_URL})")
    from langchain_ollama import ChatOllama

    emb_ok = False
    try:
        emb = _embeddings_reais()
        t0 = time.perf_counter()
        v = emb.embed_query("Como iniciar uma recarga?")
        p.ok(len(v) == 768, f"embedding gera vetor de {len(v)} dimensões (esperado 768)", f"{time.perf_counter() - t0:.2f}s")
        consulta = "Como pago com PIX?"
        rel = "Na tela de pagamento do totem selecione PIX e escaneie o QR Code com o aplicativo do banco."
        irrel = "O totem está com a tela preta e travado; acione o suporte técnico presencial."
        vq, vr, vi = emb.embed_documents([consulta, rel, irrel])
        s_rel, s_irr = _cosseno(vq, vr), _cosseno(vq, vi)
        emb_ok = p.ok(s_rel > s_irr + 0.05, "sanidade semântica: a frase sobre PIX é mais próxima da pergunta sobre PIX do que a sobre tela preta",
                      f"similaridade {s_rel:.3f} vs {s_irr:.3f}")
    except Exception as erro:  # noqa: BLE001
        p.registrar("FALHA", "embedding nomic-embed-text", f"{type(erro).__name__}: {str(erro)[:100]} (abra o app Ollama; ollama pull nomic-embed-text)")
    for modelo in (config.MODELO_CHAT_PRINCIPAL, config.MODELO_CHAT_COMPARACAO):
        try:
            llm = ChatOllama(model=modelo, base_url=config.OLLAMA_BASE_URL, temperature=0, num_predict=300,
                             client_kwargs={"headers": config.cabecalhos_ollama(config.OLLAMA_BASE_URL)})
            t0 = time.perf_counter()
            resp = str(llm.invoke("Responda apenas: OK").content).strip()
            p.ok(bool(resp), f"chat {modelo} responde", f"{time.perf_counter() - t0:.2f}s -> {resp[:30]!r}")
        except Exception as erro:  # noqa: BLE001
            p.registrar("FALHA", f"chat {modelo}", f"{type(erro).__name__}: {str(erro)[:100]}")
    return emb_ok


def secao_pipeline_real(p: Painel, simular: bool) -> dict:
    p.abrir("8) Pipeline RAG REAL" + (" (SIMULADO com embedding léxico — autoteste do script)" if simular else " (nomic-embed-text)"))
    from evals.teste_pipeline_rag import CONSULTAS, EmbeddingsLexicaisOffline, _acertou
    from src.rag.indexar import indexar
    from src.rag.retriever import Retriever
    from src.rag.vector_store import abrir_indice

    resumo: dict = {"simulado": simular}
    pasta = Path(tempfile.mkdtemp(prefix="chroma_verificacao_"))
    try:
        if simular:
            embeddings = EmbeddingsLexicaisOffline()
            stats = indexar(embeddings=embeddings, pasta_chroma=pasta)
        else:
            from src.rag.embeddings import obter_embeddings

            embeddings = obter_embeddings()
            stats = indexar(embeddings=embeddings, pasta_chroma=pasta)
        p.ok(stats["chunks_no_banco"] == stats["chunks_total"],
             f"{stats['chunks_no_banco']} chunks vetorizados e gravados (banco temporário)", f"{stats['segundos_vetorizacao']} s")
        resumo["chunks"] = stats["chunks_total"]
        loja = abrir_indice(embeddings, pasta=pasta)
        ret = Retriever(loja, k=config.TOP_K)

        linhas, acertos, esperados = [], 0, 0
        d_dentro, d_fora = [], []
        for consulta, aceitas in CONSULTAS:
            res = ret.buscar(consulta)
            dmin = res[0].distancia if res else float("nan")
            hit = None
            if aceitas:
                esperados += 1
                hit = _acertou(res, aceitas)
                acertos += hit
                d_dentro.append(dmin)
            else:
                d_fora.append(dmin)
            fontes = [f"{r.documento.metadata['fonte_id']}:p{r.documento.metadata['pagina']}" for r in res]
            linhas.append({"consulta": consulta, "hit": hit, "distancia_min": round(dmin, 4), "top": fontes})
            marca = "  -  " if hit is None else (" HIT " if hit else "MISS ")
            print(f"      [{marca}] d_min={dmin:.3f} | {consulta[:56]:56} -> {', '.join(fontes[:3])}")
        resumo.update(acertos=acertos, esperados=esperados, consultas=linhas)

        nivel = "OK" if acertos >= 7 else ("AVISO" if acertos >= 5 else "FALHA")
        p.registrar("AVISO" if simular and nivel == "OK" else nivel,
                    f"recuperação: {acertos}/{esperados} consultas com a fonte esperada no top-{config.TOP_K}",
                    "critério de trabalho: >=7 OK, 5-6 aviso, <5 falha" + (" | simulado: não mede o nomic" if simular else ""))
        if d_dentro and d_fora:
            maior_dentro, menor_fora = max(d_dentro), min(d_fora)
            resumo.update(maior_distancia_dentro_escopo=round(maior_dentro, 4), menor_distancia_fora_escopo=round(menor_fora, 4))
            p.ok(menor_fora > maior_dentro,
                 "pergunta fora do escopo fica mais longe do que TODAS as perguntas do domínio (um limiar único consegue separá-las)",
                 f"maior d_min dentro do escopo = {maior_dentro:.3f} | d_min fora do escopo = {menor_fora:.3f}", se_falhar="AVISO")
        padrao = ret.buscar("como instalar o carregador na parede e ligar o cabo de aterramento")
        p.ok(all(r.documento.metadata["publico"] in ("motorista", "geral") for r in padrao),
             "filtro de público: nenhuma resposta de instalação chega ao motorista")
        p.ok(all(r.citacao and "p." in r.citacao for r in ret.buscar("Como pago com PIX?")), "toda recuperação traz citação (documento · seção · página)")
    except Exception as erro:  # noqa: BLE001
        p.registrar("FALHA", "pipeline real", f"{type(erro).__name__}: {str(erro)[:140]}")
    return resumo


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-ollama", action="store_true", help="pula as seções 7 e 8")
    ap.add_argument("--simular-embedding", action="store_true", help="(autoteste do script) usa embedding léxico na seção 8")
    args = ap.parse_args()

    p = Painel()
    print("VERIFICAÇÃO FUNCIONAL — ChargeGrid Assistant, Sprint 4 (T1 até T3)")
    secao_ambiente(p)
    secao_estrutura(p)
    secao_git(p)
    secao_kb(p)
    secao_pipeline_offline(p)
    secao_chain_sprint3(p)
    resumo_real: dict = {}
    if args.sem_ollama:
        p.abrir("7) e 8) Ollama real e pipeline real")
        p.registrar("PULADO", "seções 7 e 8", "--sem-ollama")
    elif args.simular_embedding:
        p.abrir("7) Ollama real")
        p.registrar("PULADO", "seção 7", "--simular-embedding")
        resumo_real = secao_pipeline_real(p, simular=True)
    else:
        if secao_ollama(p):
            resumo_real = secao_pipeline_real(p, simular=False)
        else:
            p.abrir("8) Pipeline RAG REAL")
            p.registrar("PULADO", "pipeline real", "o embedding não funcionou na seção 7")

    print("\n" + "=" * 64)
    print(f"RESUMO: {p.contar('OK')} OK | {p.contar('AVISO')} aviso(s) | {p.contar('FALHA')} FALHA(S) | {p.contar('PULADO')} pulado(s)")
    for i in p.itens:
        if i["status"] in ("FALHA", "AVISO"):
            print(f"  [{i['status']}] ({i['secao'].split(')')[0]}) {i['descricao']}" + (f" -> {i['detalhe']}" if i["detalhe"] else ""))
    veredito = "T1-T3 FUNCIONAIS: pode seguir para a T4." if p.contar("FALHA") == 0 else "HÁ FALHAS: corrija antes de seguir."
    print(f"\n>>> {veredito}")
    (RAIZ / "evals" / "verificacao_t3_resultado.json").write_text(
        json.dumps({"itens": p.itens, "pipeline_real": resumo_real}, ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if p.contar("FALHA") else 0


if __name__ == "__main__":
    sys.exit(main())
