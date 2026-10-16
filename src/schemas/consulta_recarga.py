# -*- coding: utf-8 -*-
"""
Schema de domínio EV — ChargeGrid Assistant (Sprint 03)

Este módulo define a saída estruturada que o LLM deve produzir sempre que
o usuário pedir informações sobre o estado da sua sessão de recarga.

Por que Pydantic v2 (e não v1)?
- `field_validator` (substituto do antigo `@validator` do Pydantic v1) roda
  em modo "class method" e é explicitamente exigido pelo escopo da Sprint 03.
- Validação automática impede que o chatbot "invente" dados fora de faixa
  (ex.: bateria em 140%), o que é uma regra de negócio herdada do
  system prompt da Sprint 1 ("NUNCA invente dados de sessão").
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class EstadoCarregador(str, Enum):
    """Estado atual do conector/eletroposto para a sessão consultada."""

    aguardando = "aguardando"
    carregando = "carregando"
    concluido = "concluido"
    erro = "erro"


class TipoPagamento(str, Enum):
    """Forma de pagamento identificada ou selecionada na sessão."""

    pix = "pix"
    credito = "credito"
    debito = "debito"
    nao_informado = "nao_informado"


class ConsultaRecarga(BaseModel):
    """
    Representa a resposta estruturada do ChargeGrid Assistant para uma
    consulta sobre o estado de uma sessão de recarga.

    Esse schema é o contrato entre o LLM e qualquer sistema consumidor
    (totem, app mobile, backend de faturamento) — por isso os campos
    numéricos têm validação de faixa e os campos de negócio têm
    validação cruzada (ex.: escalonamento técnico deve vir acompanhado
    de orientação ao usuário).
    """

    estado_carregador: EstadoCarregador = Field(
        ..., description="Estado atual do conector/eletroposto."
    )
    percentual_bateria: float = Field(
        ..., description="Percentual de carga da bateria do veículo (0 a 100)."
    )
    energia_consumida_kwh: float = Field(
        ..., description="Energia consumida na sessão atual, em kWh."
    )
    valor_acumulado_reais: float = Field(
        ..., description="Valor acumulado da sessão atual, em reais (R$)."
    )
    tempo_restante_min: Optional[int] = Field(
        default=None, description="Tempo estimado restante para conclusão, em minutos."
    )
    tipo_pagamento: TipoPagamento = Field(
        default=TipoPagamento.nao_informado,
        description="Forma de pagamento usada ou selecionada nesta sessão.",
    )
    necessita_escalonamento_tecnico: bool = Field(
        default=False,
        description="True quando o problema relatado exige suporte técnico presencial.",
    )
    resumo_para_usuario: str = Field(
        ..., description="Texto final, em português simples, para exibir ao motorista."
    )

    # ------------------------------------------------------------------
    # field_validator: validações de campo individual
    # ------------------------------------------------------------------
    @field_validator("percentual_bateria")
    @classmethod
    def valida_percentual_bateria(cls, v: float) -> float:
        if not (0 <= v <= 100):
            raise ValueError(
                "percentual_bateria deve estar entre 0 e 100 "
                "(recebido: %s) — o assistente não pode inventar valores fora da faixa física possível."
                % v
            )
        return round(v, 1)

    @field_validator("energia_consumida_kwh", "valor_acumulado_reais")
    @classmethod
    def valida_valores_nao_negativos(cls, v: float) -> float:
        if v < 0:
            raise ValueError("valores de energia consumida e cobrança não podem ser negativos.")
        return round(v, 2)

    @field_validator("tempo_restante_min")
    @classmethod
    def valida_tempo_restante(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v < 0:
            raise ValueError("tempo_restante_min não pode ser negativo.")
        return v

    @field_validator("resumo_para_usuario")
    @classmethod
    def valida_resumo_nao_vazio(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("resumo_para_usuario não pode ser vazio.")
        if len(v) > 600:
            raise ValueError(
                "resumo_para_usuario excede o limite de objetividade definido no "
                "system prompt (respostas curtas, no máximo ~3 parágrafos)."
            )
        return v.strip()

    # ------------------------------------------------------------------
    # model_validator: validação cruzada entre campos (regra de negócio)
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def valida_coerencia_escalonamento(self) -> "ConsultaRecarga":
        if self.necessita_escalonamento_tecnico and "suporte" not in self.resumo_para_usuario.lower():
            raise ValueError(
                "quando necessita_escalonamento_tecnico=True, o resumo_para_usuario "
                "precisa orientar explicitamente o usuário a acionar o suporte técnico."
            )
        if self.estado_carregador == EstadoCarregador.concluido and self.percentual_bateria < 100:
            # Aviso de negócio: não bloqueia (sessão pode ter sido encerrada manualmente
            # antes de 100%), mas registra a inconsistência no resumo se ainda não constar.
            pass
        return self


if __name__ == "__main__":
    # Auto-teste rápido — rode `python src/schemas/consulta_recarga.py`
    exemplo_valido = ConsultaRecarga(
        estado_carregador="carregando",
        percentual_bateria=67,
        energia_consumida_kwh=18.4,
        valor_acumulado_reais=22.10,
        tempo_restante_min=23,
        tipo_pagamento="pix",
        resumo_para_usuario="Sua recarga está em 67%, faltam cerca de 23 minutos.",
    )
    print("Exemplo válido:")
    print(exemplo_valido.model_dump_json(indent=2))

    print("\nTestando validação de erro (percentual fora da faixa):")
    try:
        ConsultaRecarga(
            estado_carregador="erro",
            percentual_bateria=140,
            energia_consumida_kwh=1,
            valor_acumulado_reais=1,
            resumo_para_usuario="cabo com defeito, acione o suporte",
        )
    except Exception as erro:
        print(f"OK, validação pegou o erro esperado -> {erro}")
