# Sprint 4 – Arquitetura GPT e Geração de Texto (Capítulo 4)

## Entregável 1 – Glossário (Capítulo 4)
* **Transformer**: Arquitetura baseada em mecanismos de atenção para processar sequências em paralelo. No GPT, utiliza a variante *decoder-only* causal, conectando embeddings, blocos de transformação e cabeça de saída.
* **Transformer Block**: Unidade central repetida `n_layers` vezes que combina atenção multi-cabeças e rede densa, mantendo a dimensão `(batch_size, seq_len, emb_dim)` constante.
* **LayerNorm**: Normaliza as ativações na última dimensão (`emb_dim`) para média 0 e variância 1, estabilizando os gradientes antes da atenção e da rede densa (*Pre-LayerNorm*).
* **Conexões Residuais**: Atalhos (*shortcuts*) que somam a entrada de uma subcamada diretamente à sua saída (`x + shortcut`), evitando o desaparecimento do gradiente.
* **FeedForward Network**: Sub-rede de camadas lineares aplicada a cada posição individualmente. Expande o vetor de `emb_dim` para `4 * emb_dim` e o comprime de volta.
* **GELU**: Função de ativação suave que permite a passagem de pequenos gradientes negativos no interior da rede `FeedForward`.
* **Dropout**: Regularização que zera aleatoriamente uma fração (`drop_rate`) das ativações durante o treino para prevenir sobreajuste.
* **Logits**: Vetor de pontuações brutas e não normalizadas produzido pela última camada linear para cada token do vocabulário.
* **Language Model Head**: Camada linear final (`out_head`) que projeta os tensores de `emb_dim` (768) para `vocab_size` (50.257).
* **Inferência**: Execução do modelo em modo `model.eval()` e `torch.no_grad()`, desativando o `Dropout` e o cálculo de gradientes para prever tokens.
* **Geração Autoregressiva**: Ciclo iterativo em que o token previsto no passo atual é anexado à entrada para prever o próximo token.

## Documentação Técnica dos Entregáveis 2, 3 e 4
* **Entregável 2 – Fluxo e Dimensões no `TransformerBlock`**: Recebe um tensor `x` de formato `(B, T, 768)`. O `LayerNorm` normaliza os 768 valores de cada token sem alterar o shape; a `MultiHeadAttention` divide a dimensão 768 em 12 cabeças de `head_dim = 64`, calcula os pesos causais `(B, 12, T, T)` e reagrupa para `(B, T, 768)`. Após somar o atalho residual (`x + shortcut`), a `FeedForward` projeta de `(B, T, 768)` para `(B, T, 3072)` com ativação `GELU` e retorna a `(B, T, 768)`.
* **Entregável 3 – Processamento da Sequência no `GPTModel`**: A matriz de entrada `in_idx` de inteiros `(B, T)` passa por `tok_emb` para virar `(B, T, 768)` e soma-se ao `pos_emb` `(T, 768)` via *broadcasting*. Após atravessar os 12 `trf_blocks` e a `final_norm` mantendo `(B, T, 768)`, a camada `out_head` multiplica os vetores pela matriz de projeção `(768, 50257)`, gerando os `logits` finais de formato `(B, T, 50257)`.
* **Entregável 4 – Logits, Probabilidades e Geração Autoregressiva**: Na inferência, extrai-se apenas o vetor do último token `logits[:, -1, :]` com shape `(B, 50257)`. A função `softmax` transforma esses valores reais em uma distribuição de probabilidades que soma `1.0`, o `argmax` (ou `multinomial` quando há `temperature` e `top_k`) seleciona o índice vencedor `(B, 1)`, e `torch.cat` anexa esse ID à sequência original, fazendo a janela crescer de `T` para `T + 1` a cada iteração.

## Entregável 5 – Experimentos de Arquitetura e Análise
* **1. Baseline (GPT-124M)**: `emb_dim=768`, `n_heads=12`, `n_layers=12`, `ff_mult=4` -> ~163.0M parâmetros (~621.8 MB). Possui 163M em vez de 124M porque `tok_emb` e `out_head` têm pesos separados nesta etapa.
* **2. Redução de Camadas**: `n_layers=6` -> ~120.5M parâmetros (~459.7 MB). Reduz o tempo de inferência quase pela metade de forma linear.
* **3. Embedding Compacto**: `emb_dim=384`, `n_heads=6` -> ~64.9M parâmetros (~247.5 MB). Causa queda quadrática no número de parâmetros das camadas internas.
* **4. Mais Heads**: `n_heads=24` -> ~163.0M parâmetros (~621.8 MB). Mantém a mesma quantidade de parâmetros do Baseline, apenas dividindo `emb_dim` em 24 cabeças menores (`head_dim=32`).
* **5. FeedForward Reduzida**: `ff_mult=2` -> ~134.7M parâmetros (~513.8 MB). Economiza mais de 28M de parâmetros, mostrando que a rede `FeedForward` concentra a maior parte dos pesos de cada bloco.