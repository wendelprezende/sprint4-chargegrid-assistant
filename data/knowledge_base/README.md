# Base de conhecimento — ChargeGrid Assistant

Duas camadas, separadas pelo metadado `tipo` em [`catalogo.json`](catalogo.json):

| Camada | Arquivos | Para que serve |
|---|---|---|
| `real_goodwe` | `GW_HCAG2_UserManualPT.pdf`, `GW_HCAG2_DatasheetPT.pdf` | Fatos de **hardware** do carregador GoodWe HCA G2: potência, proteções, conector, indicador LED, partida por app/RFID, falhas |
| `ficticio_chargegrid` | `CG-01` a `CG-05` (gerados por `src/utils/gerar_documentos_ficticios.py`) | **Operação** do totem da rede ChargeGrid: PIX, tarifa dinâmica, comprovante, suporte, limites do assistente |

> Os documentos `CG-*` são **fictícios** (premissa acadêmica do projeto desde a Sprint 1),
> trazem o aviso em todas as páginas e **não** são material oficial da GoodWe.
> Valores de tarifa, descontos e prazos são placeholders editáveis.

## Fontes reais

- Manual do usuário — Carregador CA Série HCA (7 a 22 kW) G2, V1.5-2025-11-11 — https://en.goodwe.com/
- Datasheet — Linha HCA G2, GoodWe-Single page-20241121-PT-V2.1 — https://en.goodwe.com/

## Direitos autorais do manual (decisão pendente)

O manual declara que não pode ser reproduzido em plataforma pública sem autorização
escrita da GoodWe. Por segurança, o PDF do manual está no `.gitignore` e **não** sobe para
o repositório público. Quem clonar o projeto deve baixá-lo em https://en.goodwe.com/ e
colocá-lo nesta pasta com o nome `GW_HCAG2_UserManualPT.pdf`.

## Como adicionar um documento novo

1. Coloque o PDF nesta pasta.
2. Registre-o em `catalogo.json` (`arquivo`, `id`, `titulo`, `tipo`, `origem`, `secoes` com
   `paginas` e `publico`). Páginas sem seção listada não são indexadas.
3. Reindexe (comando será documentado na T3).

## Como regenerar os documentos fictícios

```bash
python -m src.utils.gerar_documentos_ficticios
```

Se alterar valores (ex.: tarifas), **revise o gabarito do eval set** para manter a coerência.

## Escopo (decisão da equipe)

A base atende o **motorista de veículo elétrico em eletropostos comerciais** (rede ChargeGrid), a persona definida na Sprint 1.
O cenário condominial (EV ChargeOps) e os regimentos condominiais de carregamento compartilhado **não** fazem parte da base; perguntas
sobre esse assunto caem na recusa fora do contexto.
