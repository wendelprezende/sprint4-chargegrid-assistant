# Relatório do pipeline RAG — ChargeGrid Assistant (Sprint 4)

> Gerado em 22/10/2026 por `python -m src.utils.gerar_relatorios` a partir de `evals/scores_por_iteracao.json`.
> Equipe 03 · FIAP · Prompt and Artificial Intelligence · Turma 1CCPK.

## 1. Visão geral

```
pergunta → guardrails (jailbreak, escopo) → busca (nomic-embed-text + ChromaDB, filtro de público, cotas por camada)
         → descarte de chunks com injeção → porta de distância → LLM (prompt RAG v2, JSON validado)
         → validação da saída (canário, link, fonte citada existe) → resposta + "Fontes: documento · seção · página"
```

## 2. Base de conhecimento

Sete documentos, em duas camadas catalogadas em `data/knowledge_base/catalogo.json` (tipo, público, origem, faixas de página):

- **Reais (GoodWe):** manual do usuário HCA G2 (70 páginas) e datasheet — hardware, LED, partida, falhas.
- **Fictícios (acadêmicos), CG-01 a CG-05:** guia do motorista, tarifas e pagamentos, solução de problemas e suporte, FAQ e termos/limites do assistente — a operação do totem (PIX, tarifa dinâmica, comprovante). Rotulados como fictícios em todas as páginas.

Decisões: (i) o conteúdo de **instalação** do manual é marcado `instalador` e nunca é recuperado para o motorista (filtro de metadados antes do LLM); (ii) capa, direitos e índice do manual são excluídos da indexação; (iii) o PDF do manual não é publicado no GitHub (restrição de direitos autorais).

## 3. Chunking, embeddings e banco vetorial

| Decisão | Valor | Justificativa e trade-off |
|---|---|---|
| Divisor | `RecursiveCharacterTextSplitter`, separadores `["\n\n","\n",". "," ",""]` | exigido pelo guia |
| Tamanho / overlap | 800 / 120 caracteres (15%) | cada chunk cabe em uma ideia; 15% de overlap evita cortar uma frase-chave |
| Corte | página a página; **documentos fictícios por seção numerada** | citação com página exata; um assunto por chunk melhora a busca. Contra: um trecho que atravessa a quebra de página é cortado |
| Limpeza | remove o aviso/rodapé repetido dos fictícios | menos ruído no embedding |
| Embedding | `nomic-embed-text` (local), com prefixos `search_document:` / `search_query:` | único modelo aprovado; a nuvem não o oferece para a nossa conta |
| Banco | ChromaDB persistente, distância de cosseno, ids determinísticos | reindexar não duplica |
| Busca | top-5 com **cota por camada** (3 fictícios + 2 reais) e filtro de público | o manual tem 78% dos chunks e, sem cota, espremia a operação do totem |

## 4. Grounding, citação e segurança

- **Prompt RAG versionado** (`prompts/prompt_rag_v1.md`, `v2`): responde só com o contexto, cita os ids dos chunks usados e diz "não encontrei" quando a base não cobre.
- **Saída estruturada (Pydantic v2)**: `encontrou_na_base`, `resposta`, `fontes`, `escalar_suporte`; o código **valida** que toda fonte citada existe entre os chunks enviados e remove as inventadas.
- **Porta de distância (0,40):** se o melhor chunk está longe demais, o LLM nem é chamado.
- **Injeção (direta e por documento):** heurística de pergunta e de documento, escape de `< > &`, canário no prompt, filtro de saída de links/e-mails/imagens. Obediência do LLM sozinho a documentos envenenados, com repetições: prompt v1: obedeceu 0/40 (IC95% -0,00–0,09); prompt v2: obedeceu 0/40 (IC95% -0,00–0,09).

## 5. Avaliação (fallback documentado do RAGAS)

Método e rubricas em `evals/rubrica_avaliacao.md`: **fidelidade** (afirmações sustentadas ÷ totais) e **relevância** (cosseno entre perguntas geradas e a original), com juiz LLM, mais métricas determinísticas. Eval set de 43 casos (16 herdados da Sprint 3); calibração do limiar em conjunto separado.

### 5.1 Scores por iteração

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

### 5.2 Antes × depois

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

### 5.3 Leitura honesta dos resultados

- **Recuperação e comportamento (ganhos acima do ruído):** hit@k 0,636 → 1,000; MRR 0,553 → 0,839; citação correta 0,800 → 0,958; comportamento correto 0,837 → 0,954; recusa indevida 0,273 → 0,091 (baseline → final).
- **Fidelidade:** o RAG fica na zona de 0,9 contra o chatbot sem RAG; diferenças entre iterações abaixo de ~0,04 estão dentro da variação do LLM e não devem ser lidas como ganho.
- **Limpeza do rodapé e limiar calibrado:** não trouxeram ganho medido (diferença de 1 caso); foram mantidas/descartadas conforme a tabela.
- **Custo:** o RAG envia bem mais tokens por turno que o chatbot sem busca; a latência média continua em torno de 1 s.

## 6. Problemas encontrados e soluções

1. **Embeddings indisponíveis no Ollama Cloud.** O diagnóstico mostrou que a conta não enxerga nenhum modelo de embedding na nuvem (o endpoint /api/embed devolveu 401 para todos), embora o chat funcione. **Solução:** O mesmo `nomic-embed-text` (único modelo aprovado) passou a rodar em um Ollama local; o chat segue na nuvem. A decisão ficou em `config.OLLAMA_EMBED_BASE_URL`.
2. **Injeção de prompt por documento.** No teste real de 19/10 o gpt-oss obedeceu a uma instrução sutil escondida em um documento e ainda o citou como fonte. **Solução:** Defesa em camadas: heurística de documentos, prompt v2, escape de tags XML, validação das fontes citadas, canário e filtro de saída (links, e-mails, imagens).
3. **Falso positivo do guardrail da Sprint 3.** A pergunta legítima de tarifação (hp-03) era recusada como fora de escopo pelo regex de domínio. **Solução:** Novas âncoras de domínio e troca do bloqueio de 'fora do domínio' pela porta de distância; a moderação ganhou padrões de 'novo papel' (0 falsos positivos em 72 perguntas legítimas).
4. **Recuperação fraca na linha de base.** Chunks com vários assuntos diluíam o embedding e o manual (78% dos chunks) espremia as respostas da operação do totem. **Solução:** Chunking por seção, prefixos do nomic e busca por camada (cota para documentos fictícios e reais). Ver a tabela de iterações.
5. **Erros na própria avaliação.** A relevância comparava consulta com documento (prefixos assimétricos do nomic) e o 'antes' sem RAG reportava métricas artificiais. **Solução:** Relevância simétrica (`--recalcular` corrige resultados gravados), métricas não comparáveis do 'antes' viraram '-' e tudo foi registrado na rubrica (seção 7).
6. **Raciocínio do gpt-oss consome o orçamento de tokens.** Com 40 tokens a resposta veio vazia (o modelo gastou tudo 'pensando'). **Solução:** `NUM_PREDICT_RAG = 800` e `reasoning = low` para modelos gpt-oss; a cadeia diagnostica resposta vazia e o experimento M6 mede a perda com 300 tokens.
7. **Limiar de distância não separa domínio de fora do escopo.** O conjunto de calibração (separado do eval set) ficou sobreposto: o limiar 0,334 não separa as classes (margem negativa). **Solução:** A porta de distância é tratada como economia de chamadas, não como barreira de segurança; mantido 0,40 e a recusa fica a cargo do prompt com grounding.

## 7. Limitações

- O eval set tem 43 casos: diferenças de 1 caso valem 0,023–0,045 e a fidelidade varia cerca de 0,04 entre execuções do mesmo LLM.
- O juiz é um LLM (por padrão o mesmo modelo do gerador); a auditoria humana por amostra (`--auditoria`) calibra esse viés.
- A detecção de injeção por heurística não pega paráfrases sem verbo de ordem; o filtro de saída não impede desinformação em texto puro.
- A base é pequena (7 documentos) e os documentos da operação do totem são fictícios (premissa acadêmica do projeto).
- O manual da GoodWe não é publicado no GitHub por restrição de direitos autorais; quem clonar o projeto deve baixá-lo em en.goodwe.com.
