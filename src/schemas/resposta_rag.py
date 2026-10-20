# -*- coding: utf-8 -*-
"""
Schema (Pydantic v2) da resposta do RAG — decisão A2 da Sprint 4.

O LLM não devolve texto solto: devolve um objeto validado com
  * `encontrou_na_base` — a resposta saiu dos documentos? (False = "não sei, base não cobre")
  * `resposta`          — texto ao motorista (curto, em português)
  * `fontes`            — ids dos chunks realmente usados (ex.: "CG-02-p02-c01") ou "dados_sessao"
  * `escalar_suporte`   — o caso exige suporte técnico presencial?

Por que estruturado: (1) o código consegue VALIDAR as citações (cada id precisa existir entre os
chunks recuperados), (2) a interface mostra as fontes sem "garimpar" o texto, (3) a avaliação
(T5) mede grounding e recusa de forma automática. Os validadores abaixo fazem o Pydantic
rejeitar saídas inconsistentes — e a chain, ao capturar o erro, pede ao modelo que se corrija
(mesma estratégia da Sprint 3 com `ConsultaRecarga`).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TAMANHO_MAXIMO_RESPOSTA = 1800  # ~3 parágrafos curtos, regra herdada da Sprint 1
MAXIMO_DE_FONTES = 6


class RespostaRAG(BaseModel):
    model_config = ConfigDict(extra="ignore")

    encontrou_na_base: bool = Field(
        description="true se a resposta foi montada a partir dos documentos do contexto (ou dos dados "
        "da sessão); false se a base não cobre a pergunta."
    )
    resposta: str = Field(
        description="Texto para o motorista, em português do Brasil, simples e empático, no máximo "
        "3 parágrafos curtos."
    )
    fontes: list[str] = Field(
        default_factory=list,
        description="Ids EXATOS (copiados do atributo id dos blocos <documento>) dos documentos usados; "
        "use 'dados_sessao' para dados da sessão em andamento. Vazio se encontrou_na_base=false.",
    )
    escalar_suporte: bool = Field(
        default=False,
        description="true se o caso é um problema físico/técnico que exige o suporte técnico presencial.",
    )

    @field_validator("resposta")
    @classmethod
    def _resposta_valida(cls, valor: str) -> str:
        valor = (valor or "").strip()
        if not valor:
            raise ValueError("'resposta' não pode ser vazia")
        if len(valor) > TAMANHO_MAXIMO_RESPOSTA:
            raise ValueError(
                f"'resposta' tem {len(valor)} caracteres (máximo {TAMANHO_MAXIMO_RESPOSTA}): resuma em até 3 parágrafos curtos"
            )
        return valor

    @field_validator("fontes")
    @classmethod
    def _fontes_limpas(cls, valor: list[str]) -> list[str]:
        vistos: list[str] = []
        for item in valor:
            item = (item or "").strip()
            if item and item not in vistos:
                vistos.append(item)
        return vistos[:MAXIMO_DE_FONTES]

    @model_validator(mode="after")
    def _coerencia_base_e_fontes(self) -> "RespostaRAG":
        if self.encontrou_na_base and not self.fontes:
            raise ValueError("encontrou_na_base=true exige ao menos uma fonte em 'fontes'")
        if not self.encontrou_na_base and self.fontes:
            self.fontes = []  # "não encontrei" não pode citar fonte: normaliza em vez de rejeitar
        return self
