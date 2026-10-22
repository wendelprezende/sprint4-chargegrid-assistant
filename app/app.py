# -*- coding: utf-8 -*-
"""
Interface web (Gradio) do ChargeGrid Assistant — Sprint 4, item 5 do escopo.

    python -m app.app                 # abre em http://127.0.0.1:7860
    python -m app.app --porta 7870

Mostra a resposta COM a citação de fonte visível (documento · seção · página, tipo, distância e o trecho recuperado),
o aviso de encaminhamento ao suporte e o diagnóstico (status, distância, tokens, latência, chunks descartados por injeção).
Aba "Comparar modelos e prompts": chamada multi-provider (bônus). Pré-requisitos: app do Ollama aberto (embeddings locais)
e OLLAMA_API_KEY no .env (chat na nuvem). O índice é criado sozinho na primeira execução.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import gradio as gr

from src import config

EXEMPLOS = [
    "Como faço para pagar com PIX?",
    "Por que o preço agora está mais caro do que ontem de manhã?",
    "O cabo não está encaixando no meu carro, o que faço?",
    "O que significa a luz azul do carregador?",
    "Quanto tempo ainda falta para terminar minha recarga?",
    "Qual é a capital da França?",
]
ROTULOS = {
    "respondida": "✅ Respondida com base nos documentos",
    "sem_resposta_na_base": "ℹ️ A base não cobre esta pergunta",
    "sem_contexto": "ℹ️ Nada relevante encontrado na base (o modelo nem foi chamado)",
    "recusa_escopo": "🚫 Fora do que o assistente pode orientar",
    "recusa_jailbreak": "🛡️ Tentativa de alterar as regras do assistente",
    "saudacao": "👋 Saudação",
    "sem_fonte_valida": "⚠️ Resposta descartada: sem fonte válida",
    "bloqueado_saida": "⚠️ Resposta bloqueada pelo filtro de saída",
    "entrada_invalida": "⚠️ Mensagem inválida",
    "erro_formato": "⚠️ Erro técnico (formato)", "erro_llm": "⚠️ Erro técnico (modelo)", "erro_recuperacao": "⚠️ Erro técnico (busca)",
}
_RAG: Any = None
_RETRIEVER: Any = None


def obter_rag() -> Any:
    """Cria o RAG uma única vez (carrega o índice persistido; reindexa se estiver ausente/desatualizado)."""
    global _RAG, _RETRIEVER
    if _RAG is None:
        from src.rag.cadeia_rag import criar_rag

        _RAG = criar_rag()
        _RETRIEVER = _RAG.retriever
    return _RAG


def formatar_fontes(r: Any) -> str:
    if not r.fontes:
        return "_Nenhuma fonte: não há documento da base sustentando esta resposta._"
    linhas = []
    for i, f in enumerate(r.fontes, 1):
        if f.id == "dados_sessao":
            linhas.append(f"**{i}. Dados da sessão em tempo real** (fornecidos pelo sistema)")
            continue
        origem = "documento **fictício** da rede ChargeGrid" if f.tipo == "ficticio_chargegrid" else "documento **real** da GoodWe"
        linhas.append(f"**{i}. {f.citacao}** — {origem} · distância {f.distancia}\n> {f.trecho}…")
    return "\n\n".join(linhas)


def formatar_diagnostico(r: Any) -> str:
    partes = [f"- **Status:** `{r.status}` — {ROTULOS.get(r.status, r.status)}",
              f"- **Melhor distância:** {r.distancia_min} · chunks recuperados: {r.chunks_recuperados}",
              f"- **Latência:** {r.latencia_total_s} s (modelo: {r.latencia_llm_s} s, {r.tentativas_llm} tentativa(s)) · prompt ≈ {r.tokens_prompt or '-'} tokens"]
    if r.descartados_injecao:
        partes.append(f"- **Chunks descartados por suspeita de injeção:** {[i for i, _ in r.descartados_injecao]}")
    if r.fontes_invalidas:
        partes.append(f"- **Fontes inventadas pelo modelo e removidas:** {r.fontes_invalidas}")
    if r.detalhe:
        partes.append(f"- **Detalhe:** {r.detalhe}")
    return "\n".join(partes)


def atender(pergunta: str, usar_sessao: bool, bateria: float, kwh: float, valor: float, minutos: float, historico: list):
    """Callback do botão Enviar. Devolve: resposta, aviso, fontes, diagnóstico, histórico (estado), histórico (texto)."""
    historico = list(historico or [])
    if not (pergunta or "").strip():
        return "Digite uma pergunta sobre o eletroposto.", "", "", "", historico, _texto_historico(historico)
    sessao = {"bateria_percentual": bateria, "energia_consumida_kwh": kwh, "valor_acumulado_reais": valor, "tempo_restante_min": minutos} if usar_sessao else None
    r = obter_rag().perguntar(pergunta, dados_sessao=sessao, historico=historico[-3:])
    aviso = "🔧 **Este caso exige o suporte técnico presencial.** Use o botão “Suporte” do totem ou o número exibido na unidade." if r.escalar_suporte else ""
    historico.append((pergunta, r.resposta))
    cabecalho = f"**{ROTULOS.get(r.status, r.status)}**\n\n"
    return cabecalho + r.resposta_formatada, aviso, formatar_fontes(r), formatar_diagnostico(r), historico, _texto_historico(historico)


def _texto_historico(historico: list) -> str:
    return "\n\n".join(f"**Você:** {p}\n\n**Assistente:** {' '.join(a.split())[:300]}" for p, a in historico[-5:]) or "_Sem conversa ainda._"


def comparar_multi(pergunta: str, modelos: list[str], prompts: list[str], usar_sessao: bool):
    from src.rag import multi_provider as mp

    if not (pergunta or "").strip() or not modelos or not prompts:
        return "Escolha uma pergunta, ao menos um modelo e ao menos um prompt."
    obter_rag()
    sessao = {"bateria_percentual": 67, "energia_consumida_kwh": 18.4, "valor_acumulado_reais": 22.10, "tempo_restante_min": 23} if usar_sessao else None
    return mp.tabela_markdown(mp.comparar(pergunta, _RETRIEVER, modelos, prompts, sessao))


def base_conhecimento_md() -> str:
    try:
        cat = json.loads((config.PASTA_KB / "catalogo.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return "_Catálogo da base não encontrado._"
    linhas = ["| Documento | Tipo | Origem |", "|---|---|---|"]
    for d in cat["documentos"]:
        tipo = "real (GoodWe)" if d["tipo"] == "real_goodwe" else "fictício (acadêmico)"
        linhas.append(f"| **{d['id']}** — {d['titulo']} | {tipo} | {d['origem']} |")
    return "\n".join(linhas) + "\n\nOs documentos **fictícios** materializam a operação do totem (PIX, tarifas, suporte); os **reais** descrevem o hardware do carregador GoodWe HCA G2."


def construir_interface() -> gr.Blocks:
    with gr.Blocks(title="ChargeGrid Assistant") as demo:
        gr.Markdown("# ⚡ ChargeGrid Assistant\nSuporte ao motorista de veículo elétrico — **RAG com citação de fonte** · FIAP EV Challenge 2026 × GoodWe · Sprint 4")
        historico = gr.State([])
        with gr.Tab("Atendimento"):
            with gr.Row():
                with gr.Column(scale=2):
                    pergunta = gr.Textbox(label="Sua pergunta", placeholder="Ex.: Como faço para pagar com PIX?", lines=2)
                    with gr.Row():
                        enviar = gr.Button("Enviar", variant="primary")
                        limpar = gr.Button("Limpar")
                    gr.Examples(EXEMPLOS, inputs=pergunta, label="Exemplos")
                    with gr.Accordion("Dados da sessão em andamento (opcional)", open=False):
                        usar_sessao = gr.Checkbox(label="Usar dados de sessão (simula a leitura em tempo real do carregador)", value=False)
                        bateria = gr.Number(label="Bateria (%)", value=67)
                        kwh = gr.Number(label="Energia consumida (kWh)", value=18.4)
                        valor = gr.Number(label="Valor acumulado (R$)", value=22.10)
                        minutos = gr.Number(label="Tempo restante (min)", value=23)
                    gr.Markdown("**Conversa recente**")
                    texto_hist = gr.Markdown("_Sem conversa ainda._")
                with gr.Column(scale=3):
                    resposta = gr.Markdown(label="Resposta")
                    aviso = gr.Markdown()
                    with gr.Accordion("📚 Fontes citadas", open=True):
                        fontes = gr.Markdown()
                    with gr.Accordion("🔎 Diagnóstico", open=False):
                        diag = gr.Markdown()
            entradas = [pergunta, usar_sessao, bateria, kwh, valor, minutos, historico]
            saidas = [resposta, aviso, fontes, diag, historico, texto_hist]
            enviar.click(atender, entradas, saidas)
            pergunta.submit(atender, entradas, saidas)
            limpar.click(lambda: ("", "", "", "", "", [], "_Sem conversa ainda._"), None, [pergunta, resposta, aviso, fontes, diag, historico, texto_hist])
        with gr.Tab("Comparar modelos e prompts (bônus)"):
            gr.Markdown("A mesma pergunta e a mesma busca, passando por **vários modelos e vários prompts** (chamada multi-provider).")
            p2 = gr.Textbox(label="Pergunta", value="Por que o preço agora está mais caro do que ontem de manhã?", lines=2)
            m2 = gr.CheckboxGroup([config.MODELO_CHAT_PRINCIPAL, config.MODELO_CHAT_COMPARACAO], value=[config.MODELO_CHAT_PRINCIPAL, config.MODELO_CHAT_COMPARACAO], label="Modelos")
            pr2 = gr.CheckboxGroup(["v1", "v2"], value=["v1", "v2"], label="Prompts RAG")
            s2 = gr.Checkbox(label="Com dados de sessão de exemplo", value=False)
            gr.Button("Comparar", variant="primary").click(comparar_multi, [p2, m2, pr2, s2], gr.Markdown())
        with gr.Tab("Base de conhecimento"):
            gr.Markdown(base_conhecimento_md())
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description="Interface web do ChargeGrid Assistant")
    ap.add_argument("--porta", type=int, default=7860)
    ap.add_argument("--compartilhar", action="store_true", help="gera um link público temporário do Gradio")
    args = ap.parse_args()
    print("Preparando o índice e o modelo (primeira vez: ~30 s)...")
    obter_rag()
    construir_interface().launch(server_port=args.porta, share=args.compartilhar)


if __name__ == "__main__":
    main()
