# -*- coding: utf-8 -*-
"""
Diagnóstico do ambiente (T1 da Sprint 4).

Responde, com testes REAIS, às perguntas que ainda estavam em aberto:
  1. A OLLAMA_API_KEY está configurada e é aceita pelo Ollama Cloud?
  2. O `nomic-embed-text` responde (embeddings, em config.OLLAMA_EMBED_BASE_URL)?
  3. `gpt-oss:120b` e o modelo de comparação respondem (chat, Ollama Cloud)?
  4. As bibliotecas do RAG estão instaladas (chromadb, pymupdf, gradio)?
  5. O RAGAS está instalado? (opcional — senão usamos o fallback manual)

USO (na raiz do projeto, com o ambiente virtual ativo):
    python -m src.utils.verificar_ollama

A chave NUNCA é impressa.
"""

from __future__ import annotations

import importlib
import time

from src import config


def _ok(msg: str) -> None:
    print(f"  [OK]    {msg}")


def _falha(msg: str) -> None:
    print(f"  [FALHA] {msg}")


def checar_bibliotecas() -> None:
    print("\n1) Bibliotecas")
    obrigatorias = {
        "langchain": "langchain",
        "langchain_ollama": "langchain-ollama",
        "langchain_chroma": "langchain-chroma",
        "chromadb": "chromadb",
        "pymupdf": "pymupdf",
        "gradio": "gradio",
        "tiktoken": "tiktoken",
    }
    for modulo, pacote in obrigatorias.items():
        try:
            importlib.import_module(modulo)
            _ok(pacote)
        except Exception as erro:  # noqa: BLE001
            _falha(f"{pacote} -> {erro}")
    try:
        importlib.import_module("ragas")
        _ok("ragas (opcional) instalado")
    except Exception:  # noqa: BLE001
        print("  [INFO]  ragas não instalado — tudo bem, usaremos o fallback manual se necessário")


def checar_chave() -> str | None:
    print("\n2) API key")
    try:
        chave = config.obter_api_key()
        _ok("OLLAMA_API_KEY encontrada (valor oculto)")
        return chave
    except RuntimeError as erro:
        _falha(str(erro))
        return None


def checar_embedding() -> None:
    print(f"\n3) Embedding: {config.MODELO_EMBEDDING} em {config.OLLAMA_EMBED_BASE_URL}")
    from langchain_ollama import OllamaEmbeddings

    try:
        emb = OllamaEmbeddings(
            model=config.MODELO_EMBEDDING,
            base_url=config.OLLAMA_EMBED_BASE_URL,
            client_kwargs={"headers": config.cabecalhos_ollama(config.OLLAMA_EMBED_BASE_URL)},
        )
        inicio = time.perf_counter()
        vetor = emb.embed_query("Como iniciar uma recarga no eletroposto?")
        _ok(f"vetor com {len(vetor)} dimensões em {time.perf_counter() - inicio:.2f}s")
    except Exception as erro:  # noqa: BLE001
        _falha(f"{type(erro).__name__}: {erro}")
        if "ollama.com" not in config.OLLAMA_EMBED_BASE_URL:
            print("          -> Servidor local: abra o app Ollama (ou `ollama serve`) e rode "
                  f"`ollama pull {config.MODELO_EMBEDDING}`; confira com `ollama list`.")


def checar_chat(chave: str, modelo: str) -> None:
    print(f"\n4) Chat: {modelo}")
    from langchain_ollama import ChatOllama

    try:
        llm = ChatOllama(
            model=modelo,
            base_url=config.OLLAMA_BASE_URL,
            client_kwargs={"headers": {"Authorization": f"Bearer {chave}"}},
            temperature=0,
            num_predict=60,
        )
        inicio = time.perf_counter()
        resposta = llm.invoke("Responda apenas: OK").content
        _ok(f"respondeu em {time.perf_counter() - inicio:.2f}s -> {str(resposta).strip()[:60]!r}")
    except Exception as erro:  # noqa: BLE001
        _falha(f"{type(erro).__name__}: {erro}")


def main() -> None:
    print("=== Diagnóstico do ambiente — ChargeGrid Assistant (Sprint 4) ===")
    checar_bibliotecas()
    chave = checar_chave()
    checar_embedding()  # independe da chave do Ollama Cloud quando o servidor é local
    if chave:
        checar_chat(chave, config.MODELO_CHAT_PRINCIPAL)
        checar_chat(chave, config.MODELO_CHAT_COMPARACAO)
    print("\nCopie TODA esta saída e cole no chat para eu validar a T1.")


if __name__ == "__main__":
    main()
