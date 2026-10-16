# Relatório de Uso de Modelos e Parâmetros — ChargeGrid Assistant (Sprint 03)

## Modelos comparados

| Modelo | Uso recomendado | Contexto |
|---|---|---|
| `gpt-oss:120b` | Modelo principal da chain (Aula 01) — raciocínio e coerência mais fortes, indicado para o núcleo conversacional que lida com regras de negócio e guardrails. | 128K tokens |
| `qwen3:8b` | Modelo de comparação (menor, mais rápido/barato) — usado no comparativo de custo-benefício e na chamada multi-provider (bônus). | Menor porte, resposta mais rápida, indicado para tarefas simples (ex.: saudações, FAQ curta). |

## Parâmetros documentados

Os parâmetros abaixo foram herdados do `system_prompt_sprint1.txt`
(Observações Técnicas de Implementação) e mantidos como padrão em
`src/chain/builder.py::get_llm()`, pois já haviam sido justificados na
Sprint 1 e continuam válidos no novo pipeline LCEL:

| Parâmetro | Valor | Justificativa |
|---|---|---|
| `temperature` | `0.3` | Respostas consistentes e controladas — o assistente lida com dados de cobrança e segurança, onde variação excessiva de tom/conteúdo é indesejada. |
| `top_p` | `0.9` | Equilíbrio entre coerência e alguma naturalidade de linguagem. |
| `num_predict` (equivalente ao antigo `max_tokens`) | `400` | Mantém as respostas objetivas, alinhado à regra de "máximo 3 parágrafos curtos" do system prompt. |

> Nota: no Ollama, o parâmetro correspondente ao `max_tokens` da OpenAI
> chama-se `num_predict`. Documentamos essa diferença de nomenclatura
> porque ela não é óbvia para quem vem da API da OpenAI (usada nas
> Sprints 1/2 no código, embora o system prompt já mirasse `gpt-4o`).

## Comparativo qualitativo esperado (a validar com execução real)

| Critério | `gpt-oss:120b` | `qwen3:8b` |
|---|---|---|
| Qualidade de raciocínio em regras de negócio (ex.: quando escalar ao suporte) | Esperado mais consistente, por ser um modelo maior | Pode exigir prompt mais diretivo para manter a mesma consistência |
| Latência | Mais lenta (modelo maior) | Mais rápida |
| Custo/uso de cota | Maior | Menor |
| Indicado para | Conversas completas, casos edge, guardrails | Respostas rápidas e simples, protótipos, chamadas em lote no eval set |

## Bônus — chamada multi-provider (+1 pt)

Implementado em `src/chain/multi_provider.py`: a mesma pergunta é
enviada para `gpt-oss:120b` (prompt v2), `qwen3:8b` (prompt v2) e
`gpt-oss:120b` (prompt v1), cumprindo "mais de um modelo e mais de um
prompt" em uma única chamada de script.
