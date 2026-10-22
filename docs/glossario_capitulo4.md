# Glossário Técnico — Capítulo 4

Este glossário complementa os termos dos Capítulos 1 e 2 e registra os
conceitos usados na integração do modelo GPT simplificado. As dimensões usam
`B` para batch size, `T` para comprimento da sequência, `D` para dimensão do
embedding, `H` para número de heads, `Dh = D/H` para a dimensão de cada head,
`F` para a dimensão interna da FeedForward Network e `V` para o tamanho do
vocabulário.

## Transformer

- **Original:** Transformer
- **Português:** Transformer
- **Definição:** Arquitetura baseada em atenção que processa uma sequência de
  representações vetoriais sem depender de recorrência.
- **Função no modelo:** Combina atenção, transformações não lineares,
  normalização e conexões residuais para construir representações contextuais.
- **Relações:** Recebe embeddings, empilha Transformer Blocks e termina com uma
  normalização e um Language Model Head.
- **Exemplo:** O GPT simplificado deste projeto recebe `[B, T]` Token IDs e
  produz logits com forma `[B, T, V]`.

## Transformer Block

- **Original:** Transformer Block
- **Português:** Bloco Transformer
- **Definição:** Unidade repetível formada por LayerNorm, Multi-Head Attention,
  FeedForward Network e conexões residuais.
- **Função no modelo:** Atualiza cada posição usando o contexto permitido e uma
  transformação não linear aplicada independentemente a cada posição.
- **Relações:** A saída de um bloco é a entrada do próximo bloco; o residual
  mantém o caminho da representação original e facilita a otimização.
- **Exemplo:** Na variante pre-norm, `x = x + Attention(LayerNorm(x))` e depois
  `x = x + FeedForward(LayerNorm(x))`.

## LayerNorm

- **Original:** Layer Normalization / LayerNorm
- **Português:** Normalização por camada
- **Definição:** Normaliza os valores na dimensão de características de cada
  posição e aplica escala e deslocamento aprendíveis.
- **Função no modelo:** Mantém as ativações em uma faixa mais estável durante o
  processamento dos blocos.
- **Relações:** É aplicada antes da atenção e da FeedForward na configuração
  pre-norm usada nesta Sprint.
- **Exemplo:** Para `[B, T, D]`, a estatística é calculada ao longo de `D`.

## Conexão residual

- **Original:** Residual connection / skip connection
- **Português:** Conexão residual ou atalho
- **Definição:** Soma a entrada de um sub-bloco à sua transformação, preservando
  um caminho direto para o sinal e para o gradiente.
- **Função no modelo:** Facilita o treinamento de redes profundas e evita que
  cada subcamada precise reaprender a identidade.
- **Relações:** É usada tanto ao redor da atenção quanto da FeedForward Network;
  exige que a transformação preserve a forma `[B, T, D]`.
- **Exemplo:** `saida = entrada + dropout(subcamada(entrada))`.

## FeedForward Network

- **Original:** FeedForward Network / Position-wise FeedForward
- **Português:** Rede FeedForward por posição
- **Definição:** MLP aplicado separadamente a cada posição, normalmente com
  projeção para uma dimensão maior, ativação e projeção de volta para `D`.
- **Função no modelo:** Introduz capacidade não linear depois da mistura de
  informações feita pela atenção.
- **Relações:** Fica dentro de cada Transformer Block e retorna à dimensão do
  residual.
- **Exemplo:** `[B,T,D] → Linear(D,F) → GELU → Dropout → Linear(F,D) → [B,T,D]`.

## GELU

- **Original:** Gaussian Error Linear Unit
- **Português:** Unidade linear de erro gaussiano
- **Definição:** Função de ativação suave que pondera a entrada pela sua
  magnitude, em vez de aplicar um corte abrupto como a ReLU.
- **Função no modelo:** Fornece a não linearidade da FeedForward Network.
- **Relações:** É aplicada entre as duas projeções lineares da FFN.
- **Exemplo:** O projeto usa `nn.GELU()` entre as projeções de dimensões `D` e
  `F`.

## Dropout

- **Original:** Dropout
- **Português:** Abandono aleatório
- **Definição:** Durante o treinamento, zera aleatoriamente parte das
  ativações e reescala as restantes.
- **Função no modelo:** Atua como regularização e reduz a dependência excessiva
  de caminhos específicos.
- **Relações:** Pode ser aplicado aos pesos de atenção, à FFN e ao residual; é
  desativado automaticamente em `model.eval()`.
- **Exemplo:** Com `dropout=0.1`, aproximadamente 10% das ativações são
  descartadas durante o treinamento.

## Multi-Head Attention

- **Original:** Multi-Head Attention
- **Português:** Atenção com múltiplas cabeças
- **Definição:** Divide as projeções de consulta, chave e valor em `H` cabeças
  que calculam relações em subespaços diferentes e depois são concatenadas.
- **Função no modelo:** Permite que cada posição incorpore informação de outras
  posições do contexto.
- **Relações:** É a subcamada contextual do Transformer Block e usa uma máscara
  causal para impedir acesso ao futuro.
- **Exemplo:** `[B,T,D] → [B,H,T,Dh] → atenção → [B,T,D]`.

## Máscara causal

- **Original:** Causal mask / look-ahead mask
- **Português:** Máscara causal
- **Definição:** Máscara triangular que permite a uma posição consultar apenas
  ela mesma e as posições anteriores.
- **Função no modelo:** Preserva o objetivo autorregressivo, evitando que a
  previsão do token `t` veja o token `t+1`.
- **Relações:** É aplicada aos scores antes do softmax dentro da atenção.
- **Exemplo:** Em uma sequência de quatro posições, a primeira linha só pode
  atender à primeira coluna.

## Logits

- **Original:** Logits
- **Português:** Escores não normalizados
- **Definição:** Valores reais produzidos pelo Language Model Head antes do
  softmax.
- **Função no modelo:** Representam a preferência do modelo por cada token do
  vocabulário em cada posição.
- **Relações:** Têm forma `[B,T,V]`, viram probabilidades com softmax e são
  usados diretamente pela cross-entropy durante o treinamento.
- **Exemplo:** Para a última posição, `logits[:, -1, :]` contém um escore para
  cada possível próximo token.

## Language Model Head

- **Original:** Language Model Head / LM head
- **Português:** Cabeça do modelo de linguagem
- **Definição:** Projeção linear que transforma a representação final de cada
  posição em um vetor de tamanho `V`.
- **Função no modelo:** Produz os logits necessários para classificar o próximo
  Token ID.
- **Relações:** É aplicada depois do último LayerNorm e conecta o Transformer ao
  vocabulário.
- **Exemplo:** `Linear(D,V)` transforma `[B,T,D]` em `[B,T,V]`.

## Inferência

- **Original:** Inference
- **Português:** Inferência
- **Definição:** Execução do modelo para obter previsões sem atualizar os
  parâmetros.
- **Função no modelo:** Usa uma sequência inicial para calcular o próximo token
  e pode ser executada com `torch.no_grad()`.
- **Relações:** Converte logits em uma decisão por argmax ou amostragem e
  alimenta essa decisão à próxima etapa.
- **Exemplo:** `model.eval()` e `with torch.no_grad(): logits = model(ids)`.

## Geração autoregressiva

- **Original:** Autoregressive generation
- **Português:** Geração autorregressiva
- **Definição:** Processo iterativo em que cada token gerado é anexado ao
  contexto usado para prever o token seguinte.
- **Função no modelo:** Produz uma continuação de comprimento variável a partir
  de uma sequência inicial.
- **Relações:** Usa logits, temperatura, top-k ou greedy decoding e respeita o
  tamanho máximo de contexto.
- **Exemplo:** `[t0,t1] → t2 → [t0,t1,t2] → t3 → ...`.

## Temperatura

- **Original:** Temperature
- **Português:** Temperatura de amostragem
- **Definição:** Fator que divide os logits antes do softmax para controlar a
  concentração da distribuição.
- **Função no modelo:** Valores menores tornam a escolha mais concentrada;
  valores maiores aumentam a diversidade.
- **Relações:** Só altera a amostragem probabilística, não os parâmetros do
  modelo nem o cálculo dos logits originais.
- **Exemplo:** `softmax(logits / temperature)` com `temperature=0.7`.

## Top-k

- **Original:** Top-k sampling
- **Português:** Amostragem entre os k melhores tokens
- **Definição:** Estratégia que mantém apenas os `k` maiores logits e elimina as
  demais opções antes do softmax.
- **Função no modelo:** Evita amostras de tokens com probabilidade muito baixa
  e mantém alguma diversidade.
- **Relações:** Pode ser combinada com temperatura durante a geração.
- **Exemplo:** Com `top_k=20`, a distribuição é calculada apenas entre os 20
  tokens mais prováveis.

