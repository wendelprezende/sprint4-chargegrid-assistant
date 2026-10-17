# -*- coding: utf-8 -*-
"""
Diagnóstico do 401 no embedding (T1). Descobre, com chamadas reais:
  A) quais modelos a SUA chave enxerga no Ollama Cloud (/api/tags);
  B) se o `nomic-embed-text` responde pelo endpoint /api/embed e pelo legado /api/embeddings;
  C) se algum outro modelo de embedding responde (para sabermos se o problema é
     o modelo específico ou o recurso de embeddings na sua conta).

USO (raiz do projeto, ambiente virtual ativo):   python -m src.utils.diagnostico_embedding
A chave NUNCA é impressa.
"""
import httpx

from src import config

CANDIDATOS = ["nomic-embed-text", "nomic-embed-text:latest", "embeddinggemma", "mxbai-embed-large", "bge-m3", "all-minilm"]


def _resumo(resp: httpx.Response) -> str:
    corpo = resp.text.replace("\n", " ")[:110]
    return f"HTTP {resp.status_code} | {corpo}"


def main() -> None:
    cabecalhos = {"Authorization": f"Bearer {config.obter_api_key()}", "Content-Type": "application/json"}
    base = config.OLLAMA_BASE_URL
    with httpx.Client(timeout=30) as http:
        print("A) Modelos visíveis para a sua chave (/api/tags)")
        r = http.get(f"{base}/api/tags", headers=cabecalhos)
        print("  ", _resumo(r) if r.status_code != 200 else f"HTTP 200 | {len(r.json().get('models', []))} modelos")
        if r.status_code == 200:
            nomes = sorted(m["name"] for m in r.json().get("models", []))
            print("   lista completa:", ", ".join(nomes) if nomes else "(vazia)")
            print("   com 'embed'/'nomic' no nome:", [n for n in nomes if "embed" in n or "nomic" in n] or "nenhum")

        print("\nB) POST /api/embed com cada modelo candidato")
        for modelo in CANDIDATOS:
            r = http.post(f"{base}/api/embed", headers=cabecalhos, json={"model": modelo, "input": "teste"})
            ok = r.status_code == 200
            extra = f"vetor de {len(r.json()['embeddings'][0])} dimensões" if ok else _resumo(r)
            print(f"   {'[OK]   ' if ok else '[FALHA]'} {modelo:26} {extra}")

        print("\nC) Endpoint legado /api/embeddings com nomic-embed-text")
        r = http.post(f"{base}/api/embeddings", headers=cabecalhos, json={"model": "nomic-embed-text", "prompt": "teste"})
        print("  ", "[OK]" if r.status_code == 200 else "[FALHA]", _resumo(r) if r.status_code != 200 else "respondeu")

        print("\nD) Controle: o chat funciona com a mesma chave?")
        r = http.post(f"{base}/api/chat", headers=cabecalhos,
                      json={"model": config.MODELO_CHAT_PRINCIPAL, "stream": False,
                            "messages": [{"role": "user", "content": "Responda apenas: OK"}], "options": {"num_predict": 20}})
        print("  ", "[OK]" if r.status_code == 200 else "[FALHA]", "HTTP", r.status_code)
    print("\nCole TODA esta saída no chat.")


if __name__ == "__main__":
    main()
