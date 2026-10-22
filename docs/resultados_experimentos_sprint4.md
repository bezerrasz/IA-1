# Resultados dos experimentos — Sprint 4

## Ambiente e referência

- Corpus: `the-verdict.txt` (20.479 caracteres);
- Vocabulário: 1.132 tokens;
- Pesos: inicialização aleatória controlada por `torch.manual_seed`;
- Medição: CPU, três execuções após uma execução de aquecimento;
- O tempo é indicativo e deve ser comparado dentro do mesmo ambiente.

## Experimentos de arquitetura

A coluna `score_elements` estima o tamanho do tensor de scores de atenção para o prompt usado (`B × H × T × T`). A coluna de parâmetros inclui embeddings, blocos, normalização e LM Head.

| Configuração | D | H | Blocos | F | Contexto | Parâmetros | Scores | Forward (ms) | Logits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| base | 32 | 4 | 2 | 64 | 16 | 90.112 | 1024 | 0.865 | `(1, 16, 1132)` |
| mais_blocos | 32 | 4 | 4 | 64 | 16 | 107.200 | 1024 | 1.598 | `(1, 16, 1132)` |
| mais_heads | 32 | 8 | 2 | 64 | 16 | 90.112 | 2048 | 0.781 | `(1, 16, 1132)` |
| embedding_maior | 64 | 4 | 2 | 128 | 16 | 212.992 | 1024 | 1.006 | `(1, 16, 1132)` |
| ffn_maior | 32 | 4 | 2 | 128 | 16 | 98.432 | 1024 | 0.865 | `(1, 16, 1132)` |
| contexto_maior | 32 | 4 | 2 | 64 | 32 | 90.624 | 4096 | 1.246 | `(1, 32, 1132)` |

### Saídas observadas

As continuações abaixo são diagnósticas: o modelo ainda não foi treinado.

| Configuração | Continuação para `The court` |
|---|---|
| base | `The <|unk|> existed Russian among Riviera` |
| mais_blocos | `The <|unk|> predicted aside growing say` |
| mais_heads | `The <|unk|> existed Russian among Riviera` |
| embedding_maior | `The <|unk|> ironic from covered excuse` |
| ffn_maior | `The <|unk|> endless fitting attack its` |
| contexto_maior | `The <|unk|> lips herself sugar sunburnt` |

## Experimentos de geração

Todos usam a mesma arquitetura base, o mesmo prompt e `max_new_tokens=4`.

| Estratégia | Temperatura | Top-k | Resultado |
|---|---:|---:|---|
| greedy (greedy) | 1.0 | — | `The <|unk|> As sex Stroud apparently` |
| amostragem (top_k_temperatura_baixa) | 0.7 | 10 | `The <|unk|> conjugal glory get buying` |
| amostragem (top_k_temperatura_alta) | 1.3 | 10 | `The <|unk|> conjugal glory get buying` |
