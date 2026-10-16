# Tabela de Versões do System Prompt — ChargeGrid Assistant

| Versão | O que mudou | Por quê | Ganho medido |
|---|---|---|---|
| **v1** (`system_prompt_v1.md`) | Baseline herdado das Sprints 1/2. Texto corrido em Markdown, sem separação estrutural entre papel, regras e contexto. Sem guardrails explícitos contra jailbreak. Sem contrato de saída estruturada. | Foi o prompt validado nas Sprints anteriores e serve de linha de base para medir evolução real. | Nº de tokens do prompt: **[PREENCHER após rodar `contagem_tokens.py`]**. Nota no eval set (`evals/eval_set.json`) com este prompt, rodando no `gpt-oss:120b`: **[PREENCHER com `evals/run_evals.py --prompt v1`]**. |
| **v2** (`system_prompt_v2.md`) | Reestruturado com **XML tagging** (`<role>`, `<regras_de_negocio>`, `<guardrails>`, `<formato_de_saida>`, etc). Adiciona bloco `<guardrails>` explícito (recusa de jailbreak/prompt injection, escopo jurídico/financeiro/segurança elétrica). Declara contrato de saída estruturada para o schema `ConsultaRecarga`. | Context engineering (Aula 04): XML tagging reduz ambiguidade de interpretação do modelo e facilita auditoria/manutenção de cada regra isoladamente. Guardrails explícitos são exigência do bloco C da rubrica (15 pts) e reduzem a chance de o modelo aceitar uma instrução maliciosa embutida na mensagem do usuário. | Nº de tokens do prompt: **[PREENCHER]** (esperado ser maior que a v1, já que adiciona blocos de guardrails — trade-off documentado no relatório de evolução). Nota no eval set com este prompt: **[PREENCHER com `evals/run_evals.py --prompt v2`]**. Taxa de recusa correta nos casos de jailbreak/out-of-scope do eval set: **[PREENCHER]**. |

## Como preencher as colunas de "ganho medido"

Os números acima **não foram inventados** — propositalmente foram deixados
como `[PREENCHER]`, porque o system prompt da própria Sprint 1 e as regras
da Sprint 3 proíbem dados fictícios ("NUNCA invente dados").

Para obter os números reais:

```bash
export OLLAMA_API_KEY="sua_api_key_aqui"
python evals/run_evals.py --prompt v1 --model gpt-oss:120b
python evals/run_evals.py --prompt v2 --model gpt-oss:120b
```

Cada execução grava um arquivo em `evals/sprint3_results.json` com:
tokens do prompt, tokens por turno, latência média e nota do eval
(percentual de respostas que atenderam ao comportamento esperado).
Basta copiar esses números para esta tabela e para
`docs/relatorio_evolucao.txt` (seção 3, tabela obrigatória antes/depois).
