# Entregável 3 — Arquitetura GPT simplificada

## Organização no código

O modelo está em `src/gpt.py` e reutiliza `InputEmbedding` de
`src/embeddings.py` e `TransformerBlock` de `src/transformer.py`:

```text
Token IDs [B,T]
    │
    ├── Token Embedding + Positional Embedding
    │                         [B,T,D]
    │
    ├── TransformerBlock 1    [B,T,D]
    ├── TransformerBlock 2    [B,T,D]
    ├── ...                   [B,T,D]
    │
    ├── LayerNorm final        [B,T,D]
    └── Language Model Head    [B,T,V] logits
```

`B` é o tamanho do lote, `T` é o comprimento do contexto, `D` é a dimensão
do embedding e `V` é o tamanho do vocabulário. Cada bloco preserva `[B,T,D]`,
portanto pode ser empilhado em `nn.ModuleList` sem adaptações entre blocos.

## Fluxo de processamento

1. `InputEmbedding` consulta a matriz de Token Embeddings e soma o vetor
   posicional de cada posição;
2. cada `TransformerBlock` normaliza a entrada, aplica atenção causal e soma
   o residual; depois normaliza novamente, aplica a FFN e soma o segundo
   residual;
3. `final_norm` estabiliza a representação após o último bloco;
4. `lm_head`, uma projeção `Linear(D,V)`, calcula os logits para todos os
   tokens possíveis em todas as posições.

Os logits ainda não são probabilidades. No treinamento, podem ser comparados
com os `target_ids` do `GPTDatasetV1` usando cross-entropy. Na inferência, a
última posição é convertida em uma decisão de próximo token pelo método
`generate`.

## Parâmetros e custo

Os parâmetros principais são configuráveis pelo `GPTConfig` ou diretamente
por `GPTModel`:

- `embedding_dim` (`D`) precisa ser divisível por `num_heads` (`H`);
- `num_layers` controla a quantidade de Transformer Blocks empilhados;
- `feed_forward_dim` (`F`) controla a largura da FFN;
- `context_length` (`T`) limita a posição máxima e a máscara causal;
- `dropout` é aplicado na atenção, na FFN e nos caminhos residuais.

A atenção possui custo dominante proporcional a `B × H × T² × Dh`; por isso,
aumentar o contexto tende a ser mais caro do que aumentar linearmente apenas a
dimensão. Os parâmetros e formas efetivas de cada configuração serão medidos
no experimento da Sprint 4.

