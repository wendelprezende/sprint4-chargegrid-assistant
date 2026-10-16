# -*- coding: utf-8 -*-
"""
Guardrail de escopo — ChargeGrid Assistant deve permanecer restrito ao
contexto GoodWe/ChargeGrid Intelligence (motorista de EV em eletroposto
comercial).

Regras exigidas pelas instruções da Sprint 03 (bloco §6):
1. Não inventar especificações de produto que não estejam na base de
   conhecimento do assistente (system prompt).
2. Recusar aconselhamento jurídico, financeiro ou de segurança elétrica,
   orientando a procurar um profissional habilitado — em vez de
   simplesmente recusar sem alternativa.
3. Qualquer pergunta fora do domínio de recarga de EV deve ser
   redirecionada, sem tentar responder "por educação".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class CategoriaForaDeEscopo(str, Enum):
    juridico = "juridico"
    financeiro = "financeiro"
    seguranca_eletrica = "seguranca_eletrica"
    fora_do_dominio = "fora_do_dominio"
    dentro_do_escopo = "dentro_do_escopo"


@dataclass
class ResultadoEscopo:
    categoria: CategoriaForaDeEscopo
    dentro_do_escopo: bool
    orientacao: str | None = None


_PADROES_JURIDICO = re.compile(
    r"process[oa]r?|advogad[oa]|indeniza[cç][aã]o|a[cç][aã]o\s+judicial|direito\s+do\s+consumidor|procon\b",
    re.IGNORECASE,
)

_PADROES_FINANCEIRO = re.compile(
    r"investir|investimento|a[cç][oõ]es\s+da\s+bolsa|empr[eé]stimo|financiamento\s+pessoal|"
    r"vale\s+a\s+pena\s+comprar\s+a[cç][oõ]es|criptomoeda",
    re.IGNORECASE,
)

_PADROES_SEGURANCA_ELETRICA = re.compile(
    r"consertar?\s+(a\s+)?fia[cç][aã]o|mexer\s+no\s+disjuntor|abrir\s+o\s+quadro\s+el[eé]trico|"
    r"trocar\s+(o\s+)?cabo\s+eu\s+mesmo|desmontar\s+o\s+carregador|reparo\s+el[eé]trico\s+caseiro",
    re.IGNORECASE,
)

# Termos que ancoram a pergunta no domínio ChargeGrid/GoodWe — usados para
# não sinalizar falso-positivo (ex.: "tarifação" contém "tarifa", que é
# legítimo, mas "vale a pena investir em ações da bolsa" não tem relação).
_PADROES_DOMINIO_EV = re.compile(
    r"recarga|carregar|carregament[oa]|carregando|eletroposto|bateria|kwh|sess[aã]o|pix|cart[aã]o|"
    r"tarifa[cç][aã]o|totem|conector|cabo|chargegrid|goodwe|carro\s+el[eé]trico|"
    r"ve[ií]culo\s+el[eé]trico|\bev\b|comprovante|pagamento|energia",
    re.IGNORECASE,
)

ORIENTACAO_JURIDICO = (
    "Não posso oferecer orientação jurídica. Para questões envolvendo direitos do "
    "consumidor ou uma possível ação judicial relacionada ao uso do eletroposto, "
    "recomendo procurar um advogado ou o Procon da sua região."
)

ORIENTACAO_FINANCEIRO = (
    "Não posso dar orientação financeira ou de investimentos. Posso te ajudar com "
    "as formas de pagamento aceitas aqui no ChargeGrid (PIX, crédito e débito) — "
    "para decisões financeiras mais amplas, recomendo consultar um profissional "
    "certificado (CFP) ou sua instituição financeira."
)

ORIENTACAO_SEGURANCA_ELETRICA = (
    "Por segurança, não posso orientar reparos elétricos ou manuseio de "
    "fiação/quadro de energia — isso deve ser feito exclusivamente por um "
    "eletricista ou técnico habilitado. Se o problema for no eletroposto, "
    "aciono o suporte técnico presencial para você agora."
)

ORIENTACAO_FORA_DO_DOMINIO = (
    "Sou o assistente do ChargeGrid Intelligence e só posso ajudar com assuntos "
    "relacionados ao uso do eletroposto: início e encerramento de sessão, "
    "progresso da recarga, tarifação e pagamento, e suporte técnico. "
    "Consigo te ajudar com algo dentro desse contexto?"
)


def validar_escopo(mensagem_usuario: str) -> ResultadoEscopo:
    """
    Classifica a mensagem do usuário quanto ao escopo GoodWe/ChargeGrid.
    Ordem de checagem importa: primeiro os domínios sensíveis (jurídico,
    financeiro, segurança elétrica), pois mesmo mencionando "cobrança" ou
    "cabo", uma pergunta pode escalar para um desses temas.
    """
    texto = mensagem_usuario or ""

    if _PADROES_JURIDICO.search(texto):
        return ResultadoEscopo(CategoriaForaDeEscopo.juridico, False, ORIENTACAO_JURIDICO)

    if _PADROES_FINANCEIRO.search(texto):
        return ResultadoEscopo(CategoriaForaDeEscopo.financeiro, False, ORIENTACAO_FINANCEIRO)

    if _PADROES_SEGURANCA_ELETRICA.search(texto):
        return ResultadoEscopo(
            CategoriaForaDeEscopo.seguranca_eletrica, False, ORIENTACAO_SEGURANCA_ELETRICA
        )

    if _PADROES_DOMINIO_EV.search(texto):
        return ResultadoEscopo(CategoriaForaDeEscopo.dentro_do_escopo, True, None)

    # Sem nenhuma âncora de domínio EV e sem bater em nenhuma categoria sensível:
    # tratamos como possivelmente fora do domínio (ex.: "qual a capital da França?").
    # Perguntas muito curtas/ambíguas (ex.: "oi", "obrigado") não devem ser bloqueadas.
    saudacoes_e_cortesias = re.compile(
        r"^\s*(oi|ol[aá]|bom\s+dia|boa\s+tarde|boa\s+noite|obrigad[oa]|valeu|tchau)\W*\s*$",
        re.IGNORECASE,
    )
    if saudacoes_e_cortesias.match(texto):
        return ResultadoEscopo(CategoriaForaDeEscopo.dentro_do_escopo, True, None)

    return ResultadoEscopo(
        CategoriaForaDeEscopo.fora_do_dominio, False, ORIENTACAO_FORA_DO_DOMINIO
    )


if __name__ == "__main__":
    testes = [
        "Como eu começo a carregar meu carro aqui?",
        "Posso processar a GoodWe se o cabo estragar meu carro?",
        "Vale a pena eu investir em ações da bolsa esse mês?",
        "Posso abrir o quadro elétrico e consertar a fiação eu mesmo?",
        "Qual a capital da França?",
        "Oi!",
    ]
    for t in testes:
        r = validar_escopo(t)
        print(f"[{r.categoria.value}] dentro_do_escopo={r.dentro_do_escopo} -> {t!r}")
