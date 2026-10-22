# ChargeGrid Assistant — Sprint 4 (RAG medido, confiável e utilizável)

**EV Challenge 2026 | GoodWe | FIAP | Prompt and Artificial Intelligence | 2026.2 | Turma 1CCPK | Equipe 03**

| Integrante | RM |
|---|---|
| Arthur Araujo Massarioli | 573308 |
| Daniel Alejandro Pupo Martinez | 573075 |
| Victor Hugo Lavaqui | 573838 |
| Wendel Pedro Rezende | 573126 |

## O que é

Assistente de suporte ao **motorista de veículo elétrico** dos eletropostos da rede ChargeGrid. Responde **somente com o que está nos documentos**
da base de conhecimento e **cita a fonte** (documento · seção · página) em toda resposta; o que a base não cobre, ele diz que não sabe e encaminha ao suporte.

| Sprint | Entrega |
|---|---|
| 1 | Planejamento: persona, escopo, fluxograma, system prompt e modelo de teste |
| 2 | Chatbot em Python puro (Colab + Gemini 2.5 Flash) |
| 3 | Refactory em LangChain LCEL: memória, saída estruturada Pydantic, guardrails, prompt versionado |
| **4** | **RAG: ChromaDB + nomic-embed-text, grounding com citação, defesa contra injeção, avaliação por iteração e interface Gradio** |

## Como executar

```bash
python -m venv .venv                       # Windows: .venv\Scripts\activate | Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # Windows: copy .env.example .env  -> cole a OLLAMA_API_KEY (nunca versione o .env)
```
Pré-requisitos: **app do Ollama aberto** com `ollama pull nomic-embed-text` (os embeddings rodam localmente: a conta não os oferece na nuvem)
e `GW_HCAG2_UserManualPT.pdf` em `data/knowledge_base/` (baixe em https://en.goodwe.com/; o PDF não é publicado por direitos autorais).

```bash
python -m app.app                                   # interface web (http://127.0.0.1:7860); indexa sozinho na 1ª vez
python -m src.rag.cadeia_rag "Como pago com PIX?"   # pergunta pela linha de comando
python -m src.rag.multi_provider "Por que o preço subiu?"   # bônus: vários modelos e prompts
python -m src.utils.verificar_ollama                # diagnóstico do ambiente
```

**Avaliação e relatórios**
```bash
python -m evals.avaliar_rag --validar-gabarito      # gabarito x base (sem LLM)
python -m evals.avaliar_rag --todas                 # iterações it0 ... it7 (cada uma é salva ao terminar)
python -m evals.avaliar_rag --recalcular            # corrige a relevância gravada (sem LLM)
python -m evals.avaliar_rag --medir-injecao 10 --prompt-versao v2
python -m evals.comparar_modelos --rodar            # modelos e parâmetros -> docs/relatorio_modelos.md
python -m src.utils.gerar_relatorios                # relatórios, PDF, README e versões a partir dos resultados
python -m evals.checklist_entrega                   # checklist final
python -m evals.teste_t5 && python -m evals.teste_t4   # testes offline (sem rede)
```

## Estrutura

```
data/knowledge_base/   7 documentos (reais GoodWe + fictícios CG-01..05) e catalogo.json (tipo, público, páginas)
src/rag/               loader, chunking, embeddings, vector_store, retriever, prompt_rag, cadeia_rag, juiz, multi_provider, indexar
src/guardrails/        moderation (jailbreak), scope_validator, documentos (injeção indireta), saida (links/e-mails)
src/schemas/           Pydantic v2 (RespostaRAG)
prompts/               prompt_rag_v1.md, prompt_rag_v2.md e versoes_rag.md (tabela de versões e ganhos)
app/                   interface Gradio + README de execução
evals/                 eval_set_rag.json (43 casos), calibracao_limiar.json, rubrica_avaliacao.md, resultados/, scores_por_iteracao.*
docs/                  relatorio_rag.md, relatorio_modelos.md, relatorio_evolucao_sprint4.pdf, equipe.json
```

## Configuração final

Chunk 800/120 (fictícios **por seção**), nomic-embed-text com prefixos, busca por camada (3 fictícios + 2 reais) com filtro de público,
porta de distância 0,40, prompt v2, `gpt-oss:120b` (temperature 0, top_p 0,9, max_tokens 800, reasoning low).

## Segurança e grounding

Entrada: jailbreak, escopo (jurídico, financeiro, elétrica). Documentos: heurística de injeção indireta e escape de tags. Geração: JSON validado,
canário no prompt, fonte citada precisa existir entre os chunks enviados. Saída: bloqueio de links, e-mails e imagens. Detalhes em `docs/relatorio_rag.md`.

## Resultados

<!-- RESULTADOS:INICIO -->
### Resultados medidos (iterações)

| Iteração | O que muda | Comport. | hit@k | MRR | Cit. correta | Fidel. | Relev. | Recusa indevida | Tokens | Lat. (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| `it0_sem_rag` | Antes: Sprints 1/2 (sem RAG) | - | 0,636 | 0,553 | - | 0,192 | 0,746 | - | 221 | 1,19 |
| `it0_sem_rag_ref` | Antes (contexto de referência do it4) | - | 1,000 | 0,839 | - | 0,220 | 0,731 | - | 221 | 1,43 |
| `it0_prompt_v1` | RAG baseline, prompt v1 | 0,837 | 0,636 | 0,553 | 0,809 | 0,863 | 0,773 | 0,273 | 2367 | 2,12 |
| `it0_baseline` | RAG baseline (prompt v2) | 0,837 | 0,636 | 0,553 | 0,800 | 0,909 | 0,778 | 0,273 | 2468 | 1,10 |
| `it1_limpeza` | + limpeza do rodapé | 0,814 | 0,591 | 0,538 | 0,842 | 0,917 | 0,766 | 0,318 | 2445 | 1,55 |
| `it2_secao` | + chunking por seção | 0,884 | 0,818 | 0,724 | 0,909 | 0,907 | 0,761 | 0,136 | 2244 | 0,82 |
| `it3_prefixo` | + prefixos do nomic | 0,907 | 0,818 | 0,784 | 0,957 | 0,934 | 0,740 | 0,136 | 2330 | 0,80 |
| `it4_camadas` | + busca por camada (configuração final) | 0,954 | 1,000 | 0,839 | 0,960 | 0,896 | 0,741 | 0,045 | 2467 | 1,79 |
| `it5_limiar` | + limiar calibrado (0,334) | 0,954 | 1,000 | 0,839 | 0,960 | 0,933 | 0,749 | 0,045 | 2462 | 0,79 |
| `it6_sem_limpeza` | Ablação: it4 sem a limpeza | 0,977 | 1,000 | 0,842 | 1,000 | 0,884 | 0,737 | 0,045 | 2489 | 0,85 |
| `it7_candidata` | Candidata final (it4 + moderação corrigida) | 0,954 | 1,000 | 0,839 | 0,958 | 0,968 | 0,749 | 0,091 | 2462 | 0,83 |

### Antes × depois

| Métrica | Sprints 1/2 (versão original) `it0_sem_rag_ref` | Sprint 4 (RAG avaliado) `it7_candidata` |
|---|---|---|
| Fidelidade (0–1): afirmações sustentadas pelo contexto | 0,220 | 0,968 |
| Relevância da resposta (0–1) | 0,731 | 0,749 |
| Presença de citação de fonte | 0,000 (0%) | 1,000 |
| Citação correta (fonte aceitável) | - | 0,958 |
| Qualidade do contexto recuperado: hit@k | - (sem busca) | 1,000 |
| Qualidade do contexto recuperado: MRR | - (sem busca) | 0,839 |
| Alucinação em fatos proibidos (menor = melhor) | 0,333 | 0,000 |
| Comportamento correto (recusas, escopo, injeção) | - (sem guardrails) | 0,954 |
| Tokens de prompt por turno | 221 | 2462 |
| Latência média por turno (s) | 1,43 | 0,83 |

Detalhes: `docs/relatorio_rag.md`, `docs/relatorio_modelos.md` e `docs/relatorio_evolucao_sprint4.pdf`.
<!-- RESULTADOS:FIM -->

## Limitações

Eval set de 43 casos (diferenças pequenas não são conclusivas); juiz LLM (viés mitigado por auditoria humana por amostra); heurística de injeção
não cobre paráfrases; documentos da operação do totem são fictícios (premissa acadêmica). Ver `docs/relatorio_rag.md`.

## Credenciais

A chave vem de variável de ambiente/`.env` (no `.gitignore`). **Nunca** faça commit do `.env`.
