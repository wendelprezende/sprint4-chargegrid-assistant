# ChargeGrid Assistant — Sprint 4 (RAG medido, confiável e utilizável)

**EV Challenge 2026 | GoodWe | FIAP | Prompt and Artificial Intelligence | 2026.2 | Turma 1CCPK**

> Status: 🚧 em construção — este README é completado ao final de cada tarefa
> (T1 → T10). Seções marcadas com *(a preencher)* ainda não foram feitas.

## Equipe 03

| Nome | RM |
|---|---|
| Arthur Araujo Massarioli | 573308 |
| Daniel Alejandro Pupo Martinez | 573075 |
| Victor Hugo Lavaqui | 573838 |
| Wendel Pedro Rezende | 573126 |

## Evolução do projeto

| Sprint | O que foi entregue |
|---|---|
| 1 | Planejamento: persona (motorista de EV), escopo, fluxograma, system prompt e modelo de teste |
| 2 | Chatbot em Python puro (Colab + Gemini 2.5 Flash) |
| 3 | Refactory em LangChain LCEL (Ollama Cloud): memória por sessão, saída estruturada Pydantic v2, guardrails, prompt versionado e eval set |
| **4** | **RAG com ChromaDB + `nomic-embed-text`, grounding com citação de fonte, avaliação (RAGAS ou fallback manual) e interface web (Gradio)** |

## Estrutura do projeto

```
data/knowledge_base/   base de conhecimento (PDFs)                (T2)
src/
  config.py            caminhos, modelos e parâmetros centrais
  chain/               núcleo LCEL + memória (Sprint 3)
  guardrails/          moderação e escopo (Sprint 3)
  schemas/             Pydantic v2 (Sprint 3)
  rag/                 loader, chunking, embeddings, vector_store, retriever, prompt_rag   (T3/T4)
  utils/               contagem de tokens e diagnóstico do ambiente
prompts/               system prompts versionados (v1, v2 da Sprint 3 + versões RAG)
app/                   interface Gradio                           (T7)
evals/                 eval set, resultados e scores por iteração (T5)
docs/                  relatórios (modelos, RAG, evolução em PDF) (T6/T9)
```

## Como rodar

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     |  Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # Windows: copy .env.example .env
# edite o .env e cole a OLLAMA_API_KEY (https://ollama.com/settings/keys)

python -m src.utils.verificar_ollama    # diagnóstico do ambiente
```

*(instruções de indexação, interface e avaliação: a preencher)*

## Base de conhecimento (T2)

7 documentos em duas camadas, catalogados em `data/knowledge_base/catalogo.json`
(detalhes em `data/knowledge_base/README.md`):

- **`real_goodwe`** — manual (70 p.) e datasheet do carregador GoodWe HCA G2: hardware, LED, partida, falhas.
- **`ficticio_chargegrid`** — 5 documentos fictícios (CG-01 a CG-05) com a operação do totem: pagamento, tarifas, suporte e limites do assistente. Rotulados como fictícios em todas as páginas.

Cada documento declara `tipo`, `publico` (motorista/instalador/geral) e origem; o conteúdo de instalação
do manual é marcado como `instalador` para **não** ser usado em respostas ao motorista.
## Pipeline RAG *(a preencher na T3)*
## Grounding, citação e segurança *(a preencher na T4)*
## Avaliação *(a preencher na T5)*
## Relatórios *(a preencher na T6 e T9)*

## Segurança de credenciais

A chave do Ollama vem de variável de ambiente / `.env` (no `.gitignore`).
**Nunca** faça commit do `.env`.
