# -*- coding: utf-8 -*-
"""
Chain de saída estruturada — usada quando o usuário pergunta pelo status
da sessão (bateria, kWh, valor, tempo restante). Em vez de devolver texto
livre, o LLM é instruído a preencher o schema `ConsultaRecarga`
(Pydantic v2), que é validado automaticamente antes de ser exibido ou
repassado a outro sistema (ex.: backend de faturamento).

Arquitetura: prompt (com format_instructions) | llm | PydanticOutputParser
"""

from __future__ import annotations

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from src.schemas.consulta_recarga import ConsultaRecarga

_PARSER = PydanticOutputParser(pydantic_object=ConsultaRecarga)

_EXEMPLO_JSON_COMPLETO = (
    '{"estado_carregador": "carregando", "percentual_bateria": 67, '
    '"energia_consumida_kwh": 18.4, "valor_acumulado_reais": 22.10, '
    '"tempo_restante_min": 23, "tipo_pagamento": "pix", '
    '"necessita_escalonamento_tecnico": false, '
    '"resumo_para_usuario": "Sua recarga está em 67%, faltam cerca de 23 minutos."}'
)


def montar_chain_estruturada(llm: Runnable, system_prompt: str) -> Runnable:
    """
    Monta a chain que força o LLM a responder no formato do schema
    `ConsultaRecarga`. `dados_sessao` deve conter os dados reais vindos
    da integração OCPP (nunca inventados pelo modelo).

    Reforços de prompt (evitam o erro mais comum observado em teste real:
    o modelo omitir o campo `resumo_para_usuario`, que é obrigatório):
      - repetição explícita de que TODOS os campos são obrigatórios;
      - um exemplo completo de JSON válido (few-shot de 1 exemplo).
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt
                + "\n\nResponda ESTRITAMENTE em JSON, sem nenhum texto fora do JSON."
                + "\nTODOS os campos do schema abaixo são OBRIGATÓRIOS, incluindo"
                + " 'resumo_para_usuario' (nunca omita esse campo)."
                + "\n{format_instructions}"
                + "\n\nExemplo de JSON completo e válido:\n{exemplo_json}",
            ),
            (
                "human",
                "Dados reais da sessão (via OCPP): {dados_sessao}\n\n"
                "Pergunta do usuário: {pergunta}",
            ),
        ]
    ).partial(
        format_instructions=_PARSER.get_format_instructions(),
        exemplo_json=_EXEMPLO_JSON_COMPLETO,
    )

    return prompt | llm | _PARSER


def montar_chain_estruturada_com_retry(
    llm: Runnable, system_prompt: str, max_tentativas: int = 2
) -> Runnable:
    """
    Versão resiliente: se o LLM devolver um JSON que falha na validação
    Pydantic (ex.: campo obrigatório ausente), reenvia ao modelo o JSON
    inválido + a mensagem de erro exata, pedindo a correção, até
    `max_tentativas` vezes. Isso não muda o schema (o campo continua
    obrigatório) nem "inventa" o dado — apenas dá ao modelo a chance de
    se autocorrigir, prática recomendada quando não se tem
    `OutputFixingParser` disponível na versão instalada do LangChain.
    """
    chain_base = montar_chain_estruturada(llm, system_prompt)
    prompt_correcao = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Você gerou um JSON inválido para o schema ConsultaRecarga. "
                "Corrija e responda de novo APENAS com o JSON completo e válido, "
                "preenchendo TODOS os campos obrigatórios (incluindo "
                "'resumo_para_usuario').\n{format_instructions}",
            ),
            (
                "human",
                "JSON inválido gerado anteriormente:\n{json_invalido}\n\n"
                "Erro de validação retornado:\n{erro}",
            ),
        ]
    ).partial(format_instructions=_PARSER.get_format_instructions())
    chain_correcao = prompt_correcao | llm | _PARSER

    def _invocar(entrada: dict):
        try:
            return chain_base.invoke(entrada)
        except Exception as primeiro_erro:
            ultimo_erro = primeiro_erro
            # Recupera o texto bruto que falhou, se disponível na exceção do parser.
            json_bruto = getattr(primeiro_erro, "llm_output", None) or str(primeiro_erro)
            for _ in range(max_tentativas):
                try:
                    return chain_correcao.invoke(
                        {"json_invalido": json_bruto, "erro": str(ultimo_erro)}
                    )
                except Exception as novo_erro:
                    ultimo_erro = novo_erro
                    json_bruto = str(novo_erro)
            raise ultimo_erro

    return RunnableLambda(_invocar)


if __name__ == "__main__":
    # Demonstração offline com FakeListChatModel
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    print("--- Teste 1: JSON válido de primeira ---")
    fake_llm_ok = FakeListChatModel(responses=[_EXEMPLO_JSON_COMPLETO])
    chain = montar_chain_estruturada(fake_llm_ok, "Você é o ChargeGrid Assistant.")
    resultado = chain.invoke(
        {
            "dados_sessao": "bateria=67%, kwh=18.4, valor=22.10, tempo_restante=23min, pagamento=pix",
            "pergunta": "Quanto falta para minha recarga terminar?",
        }
    )
    print(resultado.model_dump_json(indent=2))

    print("\n--- Teste 2: primeira resposta inválida (campo faltando), retry corrige ---")
    json_invalido = (
        '{"estado_carregador": "carregando", "percentual_bateria": 67, '
        '"energia_consumida_kwh": 18.4, "valor_acumulado_reais": 22.1, '
        '"tempo_restante_min": 23}'  # falta resumo_para_usuario, igual ao erro real
    )
    fake_llm_com_retry = FakeListChatModel(responses=[json_invalido, _EXEMPLO_JSON_COMPLETO])
    chain_resiliente = montar_chain_estruturada_com_retry(
        fake_llm_com_retry, "Você é o ChargeGrid Assistant."
    )
    resultado2 = chain_resiliente.invoke(
        {
            "dados_sessao": "bateria=67%, kwh=18.4, valor=22.10, tempo_restante=23min, pagamento=pix",
            "pergunta": "Quanto falta para minha recarga terminar?",
        }
    )
    print(resultado2.model_dump_json(indent=2))
    print("\nRetry funcionou: a chain se autocorrigiu após o erro de validação.")
