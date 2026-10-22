# Interface web — ChargeGrid Assistant (Gradio)

## Como executar

Na **raiz do projeto**, com o ambiente virtual ativo, o app do Ollama aberto e a `OLLAMA_API_KEY` no `.env`:

```bash
pip install -r requirements.txt
python -m app.app
```

Abre em http://127.0.0.1:7860. Na primeira execução o índice (`chroma_db/`) é criado sozinho (~30 s); depois é reaproveitado.
Opções: `--porta 7870` e `--compartilhar` (link público temporário).

## O que a interface mostra

- **Resposta** com o status do atendimento (respondida, base não cobre, recusa, etc.).
- **📚 Fontes citadas** (sempre visíveis): documento · seção · página, se é documento *fictício* ou *real*, a distância de busca e o trecho recuperado.
- **🔧 Aviso de suporte técnico** quando o caso é físico (cabo, conector, tela, cheiro de queimado).
- **🔎 Diagnóstico**: distância, tokens, latência, chunks descartados por suspeita de injeção, fontes inventadas removidas.
- **Dados de sessão (opcional):** simulam a leitura em tempo real do carregador (bateria, kWh, valor, tempo restante).
- **Aba "Comparar modelos e prompts" (bônus):** a mesma pergunta em vários modelos e prompts.
- **Aba "Base de conhecimento":** os 7 documentos usados (reais e fictícios).

## Observações

- Sem os dados de sessão, perguntas como "quanto tempo falta?" são respondidas com "não tenho esse dado" — o assistente não inventa.
- Perguntas fora do assunto (ex.: "capital da França") não chegam ao modelo: a porta de distância responde que não encontrou na base.
