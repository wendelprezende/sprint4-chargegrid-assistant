# Relatório de modelos e parâmetros (Sprint 4)

> Gerado em 22/10/2026 por `python -m src.utils.gerar_relatorios`. Mesmo eval set (43 casos), mesma recuperação
> (configuração final) e o **mesmo juiz** (gpt-oss:120b) em todos os experimentos; só o modelo gerador ou um parâmetro muda por vez.

## 1. Modelos

| Modelo | Papel | Observação |
|---|---|---|
| `gpt-oss:120b` (Ollama Cloud) | gerador principal e juiz | modelo de raciocínio: gasta tokens "pensando" antes de responder |
| `gemma4:31b-cloud` (Ollama Cloud) | modelo de comparação | substituiu o `qwen3:8b` da Sprint 3, que **não existe** no Ollama Cloud da nossa conta (erro 404) |
| `nomic-embed-text` (Ollama local) | embeddings | único modelo de embedding aprovado; a nuvem não oferece embeddings para a nossa conta (401) |

## 2. Parâmetros adotados e por quê

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `temperature` | **0** | recomendado pelo guia; respostas fundamentadas devem ser reproduzíveis |
| `top_p` | 0,9 | com temperature 0 a amostragem é determinística, então o `top_p` tende a ser neutro (experimento M4 verifica) |
| `max_tokens` (`num_predict`) | **800** | o gpt-oss gasta parte dos tokens raciocinando: com 40 tokens a resposta veio **vazia**; o experimento M6 mede a perda com 300 |
| `reasoning` | `low` (só gpt-oss) | reduz o consumo de tokens do raciocínio sem perder a precisão nos testes |
| `k` (top-k) | 5 = 3 fictícios + 2 reais | o experimento M5 testa 8 |
| limiar de distância | 0,40 | calibração não separável (ver relatório do RAG); mantido o valor que não gera recusas indevidas |

## 3. Resultados

_Ainda não medido. Rode: `python -m evals.comparar_modelos --rodar`._



## 4. Limitações da comparação

- O juiz é o gpt-oss:120b: ao avaliar o próprio gpt-oss há risco de viés de autoavaliação (a auditoria humana por amostra mitiga).
- Com 43 casos, diferenças pequenas não são conclusivas; os números absolutos dependem do juiz e do eval set.
- O histórico da Sprint 3 (`gpt-oss:120b` × `qwen3:8b`) não é comparável: outro eval set, sem RAG e com outro modelo.
