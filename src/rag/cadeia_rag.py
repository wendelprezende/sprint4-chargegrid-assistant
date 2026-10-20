# -*- coding: utf-8 -*-
"""
Cadeia RAG do ChargeGrid Assistant (T4) — da pergunta à resposta fundamentada e citada.

FLUXO (cada etapa pode ENCERRAR o atendimento antes de gastar uma chamada ao LLM):

  pergunta
    1. entrada válida?                         -> status "entrada_invalida"
    2. jailbreak / injeção na pergunta?        -> "recusa_jailbreak"      (guardrails/moderation.py)
    3. saudação?                               -> "saudacao"             (sem busca, sem LLM)
    4. jurídico / financeiro / elétrica?       -> "recusa_escopo"        (guardrails/scope_validator.py)
    5. recuperar chunks (filtro de público)
    6. descartar chunks com injeção indireta   -> registrados em `descartados_injecao`
    7. PORTA DE DISTÂNCIA: melhor chunk longe demais e sem dados de sessão
                                               -> "sem_contexto"         (o LLM NEM É CHAMADO)
    8. LLM -> JSON validado (Pydantic `RespostaRAG`), com correção automática se o JSON vier inválido
                                               -> "erro_formato" / "erro_llm" se não houver jeito
    9. validação da SAÍDA:
         a) canário do prompt, link, e-mail ou imagem na resposta -> "bloqueado_saida"
         b) toda fonte citada existe entre os chunks enviados? Fonte inventada é removida;
            "encontrou_na_base=true" sem nenhuma fonte válida -> "sem_fonte_valida"
   10. -> "respondida" (com citações) ou "sem_resposta_na_base" (o modelo disse que não sabe)

Toda resposta ao usuário traz a linha "Fontes: ..." (documento · seção · página) — ou diz
explicitamente que não há fonte. Isso materializa o "grounding com citação" exigido pela Sprint 4.

USO RÁPIDO (precisa do Ollama local rodando e do índice criado por `python -m src.rag.indexar`):
    python -m src.rag.cadeia_rag "Como pago com PIX?"
    python -m src.rag.cadeia_rag "Quanto tempo falta?" --sessao
"""

from __future__ import annotations

import argparse
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from src import config
from src.guardrails.documentos import analisar_chunk, neutralizar_texto
from src.guardrails.moderation import MENSAGEM_RECUSA_JAILBREAK, detectar_tentativa_de_manipulacao
from src.guardrails.saida import saida_suspeita
from src.guardrails.scope_validator import CategoriaForaDeEscopo, eh_saudacao, validar_escopo
from src.rag.prompt_rag import (
    carregar_prompt_sistema,
    formatar_contexto,
    formatar_dados_sessao,
    formatar_historico,
    montar_prompt,
)
from src.rag.retriever import formatar_citacao
from src.schemas.resposta_rag import RespostaRAG
from src.utils.contagem_tokens import estimar_tokens

MAX_CARACTERES_PERGUNTA = 1000
FONTE_SESSAO = "dados_sessao"

MSG_ENTRADA_INVALIDA = "Não consegui entender a sua mensagem. Pode escrever a sua dúvida sobre o eletroposto?"
MSG_SAUDACAO = (
    "Olá! Eu sou o ChargeGrid Assistant. Posso ajudar com início e encerramento de recarga, tarifas, "
    "pagamento (PIX, crédito e débito), comprovante e suporte técnico. Como posso ajudar?"
)
MSG_SEM_CONTEXTO = (
    "Não encontrei essa informação na minha base de conhecimento e prefiro não arriscar uma resposta errada. "
    'Para te ajudar com segurança, acione o suporte pelo botão "Suporte" do totem ou pelo número exibido na unidade.'
)
MSG_SEM_FONTE_VALIDA = (
    "Não consegui confirmar essa informação nos documentos da base, então prefiro não responder algo que "
    'possa estar errado. Acione o suporte pelo botão "Suporte" do totem ou pelo número exibido na unidade.'
)
MSG_ERRO_TECNICO = (
    "Tive um problema técnico para montar a resposta agora. Pode repetir a pergunta? Se continuar, acione o "
    'suporte pelo botão "Suporte" do totem.'
)
MSG_BLOQUEIO_SAIDA = "Não consegui responder com segurança. Pode reformular a sua dúvida sobre o uso do eletroposto?"

_EXEMPLO_JSON = (
    '{"encontrou_na_base": true, "resposta": "Para pagar com PIX, escolha PIX na tela do totem, escaneie o '
    'QR Code e confirme no app do banco.", "fontes": ["CG-02-p02-c00"], "escalar_suporte": false}'
)


@dataclass
class FonteUsada:
    id: str
    citacao: str
    fonte_id: str
    tipo: str
    pagina: int | None
    secao: str
    distancia: float | None
    trecho: str


@dataclass
class ResultadoRAG:
    pergunta: str
    status: str
    resposta: str  # texto limpo (sem a linha de fontes)
    resposta_formatada: str  # texto + linha "Fontes: ..." (é o que a interface mostra)
    encontrou_na_base: bool = False
    escalar_suporte: bool = False
    fontes: list[FonteUsada] = field(default_factory=list)
    distancia_min: float | None = None
    chunks_recuperados: int = 0
    descartados_injecao: list[tuple[str, str]] = field(default_factory=list)
    fontes_invalidas: list[str] = field(default_factory=list)
    tentativas_llm: int = 0
    llm_chamado: bool = False
    tokens_prompt: int | None = None
    metodo_tokens: str | None = None
    latencia_total_s: float = 0.0
    latencia_llm_s: float = 0.0
    detalhe: str = ""

    def para_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["fontes"] = [f.__dict__ for f in self.fontes]
        return d


def criar_llm(
    modelo: str = config.MODELO_CHAT_PRINCIPAL,
    temperature: float = config.TEMPERATURE,
    top_p: float = config.TOP_P,
    num_predict: int = config.NUM_PREDICT_RAG,
    reasoning: Any = "auto",
) -> ChatOllama:
    """ChatOllama no Ollama Cloud com os parâmetros da T4 (temperature 0, orçamento de tokens maior).

    `reasoning="auto"` aplica `config.REASONING_GPT_OSS` somente a modelos gpt-oss (os demais
    rejeitariam o parâmetro de raciocínio).
    """
    extras: dict[str, Any] = {}
    if reasoning == "auto":
        reasoning = config.REASONING_GPT_OSS if modelo.startswith("gpt-oss") else None
    if reasoning:
        extras["reasoning"] = reasoning
    return ChatOllama(
        model=modelo,
        base_url=config.OLLAMA_BASE_URL,
        client_kwargs={"headers": config.cabecalhos_ollama(config.OLLAMA_BASE_URL)},
        temperature=temperature,
        top_p=top_p,
        num_predict=num_predict,
        **extras,
    )


def _texto(mensagem: Any) -> str:
    conteudo = getattr(mensagem, "content", mensagem)
    return conteudo if isinstance(conteudo, str) else json.dumps(conteudo, ensure_ascii=False)


class ChargeGridRAG:
    """Orquestra guardrails, recuperação, porta de distância, LLM e validação da saída."""

    def __init__(
        self,
        retriever: Any,
        llm: Any,
        versao_prompt: str = config.VERSAO_PROMPT_RAG,
        limiar_distancia: float | None = config.LIMIAR_DISTANCIA,
        max_tentativas: int = config.MAX_TENTATIVAS_FORMATO,
        filtrar_saida: bool = True,
    ) -> None:
        self.filtrar_saida = filtrar_saida  # False só para MEDIR a suscetibilidade do LLM (evals)
        self.retriever = retriever
        self.llm = llm
        self.versao_prompt = versao_prompt
        self.limiar_distancia = limiar_distancia  # None desliga a porta de distância
        self.max_tentativas = max(1, max_tentativas)
        self.canario = f"CANARIO-{secrets.token_hex(4)}"
        self._parser = PydanticOutputParser(pydantic_object=RespostaRAG)
        self._prompt = montar_prompt()
        self._sistema = carregar_prompt_sistema(versao_prompt, self.canario)
        self._formato = (
            self._parser.get_format_instructions()
            + "\n\nExemplo de JSON completo e válido:\n"
            + _EXEMPLO_JSON
        )
        self._prompt_correcao = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Você gerou um JSON inválido para o schema RespostaRAG. Corrija e responda de novo APENAS com o "
                    "JSON completo e válido, mantendo o mesmo conteúdo.\n{formato}",
                ),
                ("human", "JSON inválido gerado anteriormente:\n{json_invalido}\n\nErro de validação:\n{erro}"),
            ]
        )

    # ------------------------------------------------------------------ API pública
    def perguntar(
        self,
        pergunta: str,
        dados_sessao: dict | None = None,
        historico: list[tuple[str, str]] | None = None,
    ) -> ResultadoRAG:
        inicio = time.perf_counter()
        pergunta = (pergunta or "").strip()

        def encerrar(status: str, resposta: str, **campos: Any) -> ResultadoRAG:
            fontes = campos.get("fontes", [])
            linha = (
                "Fontes: " + "; ".join(f.citacao for f in fontes)
                if fontes
                else "Fontes: nenhuma (sem documento da base para sustentar esta resposta)"
            )
            return ResultadoRAG(
                pergunta=pergunta,
                status=status,
                resposta=resposta,
                resposta_formatada=f"{resposta}\n\n{linha}",
                latencia_total_s=round(time.perf_counter() - inicio, 3),
                **campos,
            )

        # 1) entrada
        if not pergunta or len(pergunta) > MAX_CARACTERES_PERGUNTA:
            return encerrar("entrada_invalida", MSG_ENTRADA_INVALIDA, detalhe="vazia ou acima de 1000 caracteres")

        # 2) jailbreak / injeção direta
        moderacao = detectar_tentativa_de_manipulacao(pergunta)
        if moderacao.bloqueado:
            return encerrar("recusa_jailbreak", MENSAGEM_RECUSA_JAILBREAK, detalhe=moderacao.motivo or "")

        # 3) saudação
        if eh_saudacao(pergunta):
            return encerrar("saudacao", MSG_SAUDACAO)

        # 4) categorias sensíveis (a categoria "fora_do_dominio" NÃO bloqueia aqui: quem decide é a
        #    porta de distância — o regex de domínio gerou falso positivo na Sprint 3)
        escopo = validar_escopo(pergunta)
        if not escopo.dentro_do_escopo and escopo.categoria != CategoriaForaDeEscopo.fora_do_dominio:
            return encerrar("recusa_escopo", escopo.orientacao or MSG_SEM_CONTEXTO, detalhe=escopo.categoria.value)

        # 5) recuperação
        try:
            recuperados = self.retriever.buscar(pergunta)
        except Exception as erro:  # noqa: BLE001 — embedding/Chroma fora do ar
            return encerrar("erro_recuperacao", MSG_ERRO_TECNICO, detalhe=f"{type(erro).__name__}: {str(erro)[:160]}")

        # 6) injeção indireta: descarta chunks suspeitos
        aprovados, descartados = [], []
        for r in recuperados:
            analise = analisar_chunk(r.documento.page_content)
            if analise.suspeito:
                descartados.append((r.documento.metadata["chunk_id"], analise.motivo or ""))
            else:
                aprovados.append(r)
        distancia_min = min((r.distancia for r in aprovados), default=None)
        base = {
            "distancia_min": round(distancia_min, 4) if distancia_min is not None else None,
            "chunks_recuperados": len(recuperados),
            "descartados_injecao": descartados,
        }

        # 7) porta de distância
        tem_sessao = bool(dados_sessao)
        if not tem_sessao:
            sem_chunk = not aprovados
            longe = self.limiar_distancia is not None and distancia_min is not None and distancia_min > self.limiar_distancia
            if sem_chunk or longe:
                motivo = "todos os chunks recuperados foram descartados" if sem_chunk else (
                    f"melhor distância {distancia_min:.3f} > limiar {self.limiar_distancia}"
                )
                return encerrar("sem_contexto", MSG_SEM_CONTEXTO, detalhe=motivo, **base)

        # 8) LLM
        entrada = {
            "sistema": self._sistema,
            "formato": self._formato,
            "contexto": formatar_contexto(aprovados),
            "dados_sessao": formatar_dados_sessao(dados_sessao),
            "historico": formatar_historico(historico),
            "pergunta": neutralizar_texto(pergunta),
        }
        valor_prompt = self._prompt.invoke(entrada)
        tokens, metodo = estimar_tokens(valor_prompt.to_string())
        base.update(tokens_prompt=tokens, metodo_tokens=metodo)

        t_llm = time.perf_counter()
        resposta, tentativas, erro_llm, bruto = self._gerar(valor_prompt)
        base.update(tentativas_llm=tentativas, llm_chamado=True, latencia_llm_s=round(time.perf_counter() - t_llm, 3))
        if resposta is None:
            status = "erro_llm" if erro_llm and erro_llm.startswith("erro_llm") else "erro_formato"
            return encerrar(status, MSG_ERRO_TECNICO, detalhe=erro_llm or "", **base)

        # 9a) canário
        if self.canario in bruto or self.canario in resposta.resposta:
            return encerrar("bloqueado_saida", MSG_BLOQUEIO_SAIDA, detalhe="palavra-código do prompt apareceu na saída", **base)

        # 9a') saída com link/e-mail/imagem: efeito típico de injeção que escapou das camadas anteriores
        if self.filtrar_saida:
            motivo_saida = saida_suspeita(resposta.resposta)
            if motivo_saida:
                return encerrar("bloqueado_saida", MSG_BLOQUEIO_SAIDA, detalhe=motivo_saida, **base)

        # 9b) citações
        por_id = {r.documento.metadata["chunk_id"]: r for r in aprovados}
        validas, invalidas = [], []
        for fonte in resposta.fontes:
            (validas if fonte in por_id or (fonte == FONTE_SESSAO and tem_sessao) else invalidas).append(fonte)
        fontes_usadas = [self._fonte_usada(f, por_id.get(f)) for f in validas]
        base["fontes_invalidas"] = invalidas

        if resposta.encontrou_na_base and not validas:
            return encerrar("sem_fonte_valida", MSG_SEM_FONTE_VALIDA, detalhe=f"fontes citadas inexistentes: {invalidas}", **base)

        status = "respondida" if resposta.encontrou_na_base else "sem_resposta_na_base"
        return encerrar(
            status,
            resposta.resposta,
            encontrou_na_base=resposta.encontrou_na_base,
            escalar_suporte=resposta.escalar_suporte,
            fontes=fontes_usadas if resposta.encontrou_na_base else [],
            **base,
        )

    # ------------------------------------------------------------------ internos
    def _gerar(self, valor_prompt: Any) -> tuple[RespostaRAG | None, int, str | None, str]:
        """Chama o LLM e valida o JSON; em caso de JSON inválido, pede correção (até max_tentativas)."""
        bruto, ultimo_erro = "", ""
        for tentativa in range(1, self.max_tentativas + 1):
            entrada = valor_prompt if tentativa == 1 else self._prompt_correcao.invoke(
                {"formato": self._formato, "json_invalido": bruto[:2000] or "(resposta vazia)", "erro": ultimo_erro[:600]}
            )
            try:
                mensagem = self.llm.invoke(entrada)
            except Exception as erro:  # noqa: BLE001 — rede, autenticação, modelo inexistente
                return None, tentativa, f"erro_llm: {type(erro).__name__}: {str(erro)[:200]}", bruto
            bruto = _texto(mensagem)
            try:
                if not bruto.strip():
                    raise ValueError(
                        "resposta vazia: o raciocínio do modelo pode ter consumido todos os tokens "
                        "(aumente NUM_PREDICT_RAG ou reduza REASONING_GPT_OSS)"
                    )
                return self._parser.parse(bruto), tentativa, None, bruto
            except Exception as erro:  # noqa: BLE001 — JSON inválido / schema violado
                ultimo_erro = f"{type(erro).__name__}: {erro}"
        return None, self.max_tentativas, f"JSON inválido após {self.max_tentativas} tentativa(s): {ultimo_erro[:240]}", bruto

    @staticmethod
    def _fonte_usada(id_fonte: str, resultado: Any) -> FonteUsada:
        if resultado is None:  # dados da sessão
            return FonteUsada(FONTE_SESSAO, "Dados da sessão em tempo real", FONTE_SESSAO, "sistema", None, "", None, "")
        meta = resultado.documento.metadata
        trecho = " ".join(resultado.documento.page_content.split())[:240]
        return FonteUsada(
            id=id_fonte,
            citacao=formatar_citacao(meta),
            fonte_id=meta["fonte_id"],
            tipo=meta["tipo"],
            pagina=meta["pagina"],
            secao=meta.get("secao", ""),
            distancia=round(resultado.distancia, 4),
            trecho=trecho,
        )


def criar_rag(modelo: str = config.MODELO_CHAT_PRINCIPAL, **kwargs: Any) -> ChargeGridRAG:
    """Monta tudo com o índice persistido em chroma_db/ (rode `python -m src.rag.indexar` antes)."""
    from src.rag.embeddings import obter_embeddings
    from src.rag.retriever import Retriever
    from src.rag.vector_store import abrir_indice

    retriever = Retriever(abrir_indice(obter_embeddings()))
    return ChargeGridRAG(retriever, criar_llm(modelo), **kwargs)


def _main() -> None:
    ap = argparse.ArgumentParser(description="Faz uma pergunta ao ChargeGrid Assistant (RAG).")
    ap.add_argument("pergunta")
    ap.add_argument("--modelo", default=config.MODELO_CHAT_PRINCIPAL)
    ap.add_argument("--sessao", action="store_true", help="injeta dados de sessão de exemplo (67%%, 18,4 kWh, R$ 22,10, 23 min)")
    ap.add_argument("--sem-porta", action="store_true", help="desliga a porta de distância (para estudar o comportamento)")
    args = ap.parse_args()

    sessao = {"bateria_percentual": 67, "energia_consumida_kwh": 18.4, "valor_acumulado_reais": 22.10,
              "tempo_restante_min": 23} if args.sessao else None
    rag = criar_rag(args.modelo, limiar_distancia=None if args.sem_porta else config.LIMIAR_DISTANCIA)
    r = rag.perguntar(args.pergunta, dados_sessao=sessao)
    print(f"\n[{r.status}] {r.latencia_total_s}s (LLM {r.latencia_llm_s}s, {r.tentativas_llm} tentativa(s)) | "
          f"d_min={r.distancia_min} | ~{r.tokens_prompt} tokens de prompt ({r.metodo_tokens})")
    print(f"\n{r.resposta_formatada}")
    if r.escalar_suporte:
        print("\n>> encaminhar ao suporte técnico presencial")
    if r.descartados_injecao:
        print(f"\n>> chunks descartados por injeção: {r.descartados_injecao}")
    if r.fontes_invalidas:
        print(f"\n>> fontes inventadas pelo modelo e removidas: {r.fontes_invalidas}")
    if r.detalhe:
        print(f"\n>> detalhe: {r.detalhe}")


if __name__ == "__main__":
    _main()
