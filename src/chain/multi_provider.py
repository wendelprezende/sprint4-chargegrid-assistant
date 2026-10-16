# -*- coding: utf-8 -*-
"""
BÔNUS (+1 pt) — chamada multi-provider: consulta mais de um modelo e mais
de um prompt (versionado) para a mesma pergunta, e devolve os dois
resultados lado a lado. Útil para o relatório de modelos
(docs/relatorio_modelos.md).

Uso real (com API key do Ollama Cloud):
    export OLLAMA_API_KEY="sua_api_key"
    python src/chain/multi_provider.py "Como eu começo a carregar meu carro aqui?"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict

RAIZ = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ))

from src.chain.builder import get_llm, montar_chain


def _carregar_prompt(versao: str) -> str:
    caminho = RAIZ / "prompts" / f"system_prompt_{versao}.md"
    texto = caminho.read_text(encoding="utf-8")
    inicio = texto.find("```")
    fim = texto.find("```", inicio + 3)
    return "\n".join(texto[inicio:fim].split("\n")[1:]).strip()


def consultar_multi_provider(pergunta: str) -> Dict[str, dict]:
    """
    Consulta (modelo x prompt):
      - gpt-oss:120b  + system_prompt v2
      - qwen3:8b      + system_prompt v2
    Isso cumpre "mais de um modelo" (dois modelos Ollama Cloud distintos)
    e "mais de um prompt" (comparando também contra a v1, se desejado)
    exigidos pelo bônus.
    """
    prompt_v2 = _carregar_prompt("v2")
    prompt_v1 = _carregar_prompt("v1")

    combinacoes = [
        ("gpt-oss:120b", "v2", prompt_v2),
        ("qwen3:8b", "v2", prompt_v2),
        ("gpt-oss:120b", "v1", prompt_v1),
    ]

    resultados = {}
    for modelo, versao_prompt, prompt in combinacoes:
        chave = f"{modelo}__{versao_prompt}"
        try:
            llm = get_llm(model=modelo)
            chain = montar_chain(llm, prompt)
            inicio = time.perf_counter()
            resposta = chain.invoke({"pergunta": pergunta, "historico": []})
            latencia = time.perf_counter() - inicio
            resultados[chave] = {"resposta": resposta, "latencia_segundos": round(latencia, 3)}
        except Exception as e:
            resultados[chave] = {"erro": str(e)}

    return resultados


if __name__ == "__main__":
    pergunta = sys.argv[1] if len(sys.argv) > 1 else "Como eu começo a carregar meu carro aqui?"
    resultados = consultar_multi_provider(pergunta)
    for chave, r in resultados.items():
        print(f"\n=== {chave} ===")
        if "erro" in r:
            print(f"[erro] {r['erro']}")
        else:
            print(f"({r['latencia_segundos']}s) {r['resposta']}")
