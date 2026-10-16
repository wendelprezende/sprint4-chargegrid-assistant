# -*- coding: utf-8 -*-
"""
Memória conversacional por sessão, com limite de tokens.

NOTA TÉCNICA IMPORTANTE (documentada também no relatório de evolução,
seção "Problemas encontrados e soluções"):

As instruções da Sprint 03 pedem `RunnableWithMessageHistory` +
`ConversationTokenBufferMemory`. Essa segunda classe pertence à API de
memória "legada" do LangChain (`langchain.memory`) e foi REMOVIDA nas
versões atuais da biblioteca (LangChain >= 1.0), que é a versão
disponível via pip hoje e a única compatível com `langchain-ollama`
atual (necessário para falar com o Ollama Cloud).

Solução adotada: implementamos o EQUIVALENTE FUNCIONAL do
`ConversationTokenBufferMemory` usando as ferramentas atuais e
oficialmente recomendadas pelo próprio LangChain:
  - `BaseChatMessageHistory` (mesma interface usada pelo
    `RunnableWithMessageHistory`, exigido no escopo);
  - `trim_messages` (utilitário atual do `langchain_core.messages`)
    para podar o histórico sempre que ele ultrapassar um orçamento de
    tokens, contado com `tiktoken` — exatamente o comportamento que o
    `ConversationTokenBufferMemory` oferecia.

Ou seja: a *funcionalidade* pedida (memória por sessão, com corte por
tokens) é entregue de forma idêntica; apenas o nome da classe interna
mudou porque a classe antiga não existe mais na versão atual da lib.
"""

from __future__ import annotations

from typing import Dict, List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, trim_messages

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")

    def _contar_tokens(mensagens: List[BaseMessage]) -> int:
        return sum(len(_ENCODING.encode(str(m.content))) for m in mensagens)

except Exception:  # pragma: no cover - fallback só para ambientes sem acesso à internet
    def _contar_tokens(mensagens: List[BaseMessage]) -> int:
        # Aproximação grosseira (1 token ~ 1 palavra) usada apenas se o
        # tiktoken não conseguir baixar sua tabela de encoding (ex.: rede
        # bloqueada). Em ambiente normal do aluno, o bloco try acima é usado.
        return sum(len(str(m.content).split()) for m in mensagens)


class MemoriaComLimiteDeTokens(BaseChatMessageHistory):
    """
    Histórico de conversa de UMA sessão, podado automaticamente para
    nunca ultrapassar `max_tokens` tokens — substitui o
    `ConversationTokenBufferMemory` do LangChain legado.
    """

    def __init__(self, max_tokens: int = 800):
        self._mensagens: List[BaseMessage] = []
        self.max_tokens = max_tokens

    @property
    def messages(self) -> List[BaseMessage]:  # nome exigido pela interface do LangChain
        return self._mensagens

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self._mensagens.extend(messages)
        self._podar_historico()

    def _podar_historico(self) -> None:
        self._mensagens = trim_messages(
            self._mensagens,
            token_counter=_contar_tokens,
            max_tokens=self.max_tokens,
            strategy="last",   # mantém os turnos mais RECENTES (comportamento do buffer)
            start_on="human",  # nunca corta no meio de um turno humano/assistente
        )

    def clear(self) -> None:
        self._mensagens = []

    def tokens_atuais(self) -> int:
        return _contar_tokens(self._mensagens)


class GerenciadorDeSessoes:
    """
    Mantém uma `MemoriaComLimiteDeTokens` por `session_id`, para que o
    `RunnableWithMessageHistory` consiga isolar a conversa de cada
    motorista/totem sem misturar sessões diferentes.
    """

    def __init__(self, max_tokens_por_sessao: int = 800):
        self._sessoes: Dict[str, MemoriaComLimiteDeTokens] = {}
        self.max_tokens_por_sessao = max_tokens_por_sessao

    def obter_historico(self, session_id: str) -> MemoriaComLimiteDeTokens:
        if session_id not in self._sessoes:
            self._sessoes[session_id] = MemoriaComLimiteDeTokens(
                max_tokens=self.max_tokens_por_sessao
            )
        return self._sessoes[session_id]

    def limpar_sessao(self, session_id: str) -> None:
        if session_id in self._sessoes:
            self._sessoes[session_id].clear()


if __name__ == "__main__":
    from langchain_core.messages import AIMessage, HumanMessage

    gerenciador = GerenciadorDeSessoes(max_tokens_por_sessao=40)
    historico = gerenciador.obter_historico("totem-01-sessao-abc")

    turnos = [
        ("Como eu começo a carregar meu carro aqui?", "Conecte o cabo e selecione o pagamento."),
        ("Quanto tempo falta para terminar?", "Faltam cerca de 23 minutos, você está em 67%."),
        ("Posso pagar com PIX?", "Pode sim! Selecione PIX na tela e escaneie o QR Code."),
    ]

    for pergunta, resposta in turnos:
        historico.add_messages([HumanMessage(pergunta), AIMessage(resposta)])
        print(f"tokens no histórico após o turno: {historico.tokens_atuais()}")

    print("\nMensagens finais mantidas na memória (após poda por limite de tokens):")
    for m in historico.messages:
        print(f"  [{type(m).__name__}] {m.content}")
