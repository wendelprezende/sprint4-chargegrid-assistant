# Relatório de modelos e parâmetros (Sprint 4)

> Gerado em 23/10/2026 por `python -m src.utils.gerar_relatorios`. Mesmo eval set (43 casos), mesma recuperação
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

| Experimento | Modelo | temperature | top_p | max_tokens | k | Comport. | Fidel. | Relev. | hit@k | Cit. correta | Tokens | Lat. (s) | Erros |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `M1_gptoss_ref` | gpt-oss:120b | 0.0 | 0.9 | 800 | 5 | 0,977 | 0,909 | 0,740 | 1,000 | 0,960 | 2463 | 0,82 | 0 |
| `M2_gemma4` | gemma4:31b-cloud | 0.0 | 0.9 | 800 | 5 | 0,977 | 0,942 | 0,726 | 1,000 | 0,962 | 2462 | 1,35 | 0 |
| `M3_gptoss_temp03` | gpt-oss:120b | 0.3 | 0.9 | 800 | 5 | 0,977 | 0,917 | 0,743 | 1,000 | 0,960 | 2462 | 0,86 | 0 |
| `M4_gptoss_topp1` | gpt-oss:120b | 0.0 | 1.0 | 800 | 5 | 0,977 | 0,917 | 0,744 | 1,000 | 0,960 | 2465 | 0,88 | 0 |
| `M5_gptoss_topk8` | gpt-oss:120b | 0.0 | 0.9 | 800 | 8 | 1,000 | 0,899 | 0,751 | 1,000 | 1,000 | 2990 | 0,89 | 0 |
| `M6_gptoss_np300` | gpt-oss:120b | 0.0 | 0.9 | 300 | 5 | 0,977 | 0,917 | 0,741 | 1,000 | 0,960 | 2463 | 0,83 | 0 |

- `M2_gemma4` em relação ao M1: comportamento +0.000, fidelidade +0.032, relevância -0.014, hit@k +0.000; erros de formato/LLM: 0.
- `M3_gptoss_temp03` em relação ao M1: comportamento +0.000, fidelidade +0.007, relevância +0.003, hit@k +0.000; erros de formato/LLM: 0.
- `M4_gptoss_topp1` em relação ao M1: comportamento +0.000, fidelidade +0.008, relevância +0.005, hit@k +0.000; erros de formato/LLM: 0.
- `M5_gptoss_topk8` em relação ao M1: comportamento +0.023, fidelidade -0.011, relevância +0.011, hit@k +0.000; erros de formato/LLM: 0.
- `M6_gptoss_np300` em relação ao M1: comportamento +0.000, fidelidade +0.008, relevância +0.001, hit@k +0.000; erros de formato/LLM: 0.

**Melhor por comportamento correto e, em empate, fidelidade:** `M5_gptoss_topk8`. Diferenças de fidelidade abaixo de ~0,04 estão dentro da variação do LLM; diferenças de 1 caso valem 0,023 em comportamento.

## 4. Limitações da comparação

- O juiz é o gpt-oss:120b: ao avaliar o próprio gpt-oss há risco de viés de autoavaliação (a auditoria humana por amostra mitiga).
- Com 43 casos, diferenças pequenas não são conclusivas; os números absolutos dependem do juiz e do eval set.
- O histórico da Sprint 3 (`gpt-oss:120b` × `qwen3:8b`) não é comparável: outro eval set, sem RAG e com outro modelo.
