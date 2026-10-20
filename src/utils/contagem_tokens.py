# -*- coding: utf-8 -*-
"""
Medição de tokens com `tiktoken` — exigida pela Aula 04 (Context
Engineering) para comparar o "custo" de cada versão do system prompt.

Uso:
    python -m src.utils.contagem_tokens prompts/system_prompt_v1.md
    python -m src.utils.contagem_tokens prompts/system_prompt_v2.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import tiktoken


def contar_tokens_arquivo(caminho: str, encoding_name: str = "cl100k_base") -> int:
    texto = Path(caminho).read_text(encoding="utf-8")
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(texto))


def contar_tokens_texto(texto: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(texto))


def estimar_tokens(texto: str) -> tuple[int, str]:
    """
    Contagem tolerante a falhas (T4): usa o tiktoken quando disponível; sem rede/encoding,
    cai numa estimativa por caracteres (~3,6 por token em português). Devolve (tokens, método)
    para que o relatório diga se o número é medido ou estimado.
    """
    try:
        return contar_tokens_texto(texto), "tiktoken"
    except Exception:  # noqa: BLE001 — sem rede para baixar o encoding
        return int(len(texto) / 3.6) + 1, "estimativa_por_caracteres"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python -m src.utils.contagem_tokens <caminho_do_arquivo.md>")
        sys.exit(1)

    caminho = sys.argv[1]
    try:
        total = contar_tokens_arquivo(caminho)
        print(f"{caminho}: {total} tokens (encoding cl100k_base)")
    except Exception as e:
        print(
            f"Não foi possível contar tokens de '{caminho}': {e}\n"
            "Dica: este script precisa de acesso à internet na primeira execução "
            "(o tiktoken baixa a tabela de encoding automaticamente)."
        )
        sys.exit(1)
