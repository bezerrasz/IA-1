# Leitura orientada — Capítulo 2: Working with Text Data

## Objetivo da leitura

O Capítulo 2 apresenta a transformação gradual de texto legível por pessoas em
estruturas numéricas que podem ser processadas por uma rede neural. A ideia
central não é apenas tokenizar, mas preservar a ordem e o contexto necessários
para que o modelo aprenda a prever o próximo token.

O estudo foi relacionado ao código do projeto e ao fluxo apresentado na
referência oficial do livro: [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch/tree/main/ch02).

## 1. Texto e tokenização

O modelo não recebe o texto bruto diretamente. Primeiro, o texto é dividido em
unidades menores chamadas tokens. Dependendo do tokenizador, um token pode ser
uma palavra, parte de uma palavra, número, símbolo ou pontuação.

Nesta Sprint, a implementação reproduz o tokenizador simples usado no início
do capítulo: uma expressão regular separa palavras, pontuação e o marcador de
travessão duplo. Espaços são utilizados como separadores, mas não permanecem
como tokens.

Esse tokenizador é didático e baseado em palavras e símbolos. Ele não é um
tokenizador BPE completo. Essa escolha torna visível cada etapa do pipeline,
mas também faz com que palavras desconhecidas precisem ser representadas por
`<|unk|>`.

## 2. Vocabulário e Token IDs

O vocabulário é a tabela que associa cada token distinto a um inteiro. A
conversão pode ser representada assim:

```text
token                 vocabulário             Token ID
"painting"            {"painting": 123}       123
```

O Token ID não é uma medida de significado. O número 123 não é semanticamente
"maior" ou "mais próximo" do número 124. Ele é somente um índice usado para
localizar o vetor correspondente na camada de embedding.

Tokens especiais têm funções de controle. O projeto reserva `<|unk|>` para
tokens ausentes do vocabulário e `<|endoftext|>` para separar documentos ou
indicar o fim de um trecho. O tokenizador deve tratar esses marcadores antes da
separação genérica para que eles não sejam divididos em caracteres isolados.

## 3. Sequências de treinamento

Depois da conversão para IDs, todo o corpus pode ser visto como uma sequência
numérica:

```text
[id_0, id_1, id_2, id_3, id_4, ...]
```

O modelo autorregressivo aprende a prever o próximo token. Por isso, cada
amostra contém uma entrada e um alvo deslocado uma posição:

```text
entrada: [id_0, id_1, id_2, id_3]
alvo:    [id_1, id_2, id_3, id_4]
```

O tamanho da entrada é o `context_length`. Uma janela deslizante percorre a
sequência e cria várias amostras. O `stride` define quanto a janela avança:

- `stride` menor: mais sobreposição e mais amostras;
- `stride` igual ao contexto: janelas sem sobreposição;
- `stride` maior: menos amostras e menor custo de armazenamento.

Se o corpus tem `N` tokens, o contexto tem tamanho `L` e o alvo precisa de um
token adicional, a quantidade de janelas é:

```text
floor((N - (L + 1)) / stride) + 1
```

quando `N >= L + 1`.

## 4. Embeddings

Os Token IDs são índices discretos e não devem ser enviados diretamente para
as camadas do modelo como se fossem valores contínuos. A camada de embedding
mantém uma matriz com forma:

```text
[vocab_size, embedding_dim]
```

Para cada ID, ela busca uma linha dessa matriz. Assim, uma sequência com forma
`[batch_size, context_length]` passa a ter forma:

```text
[batch_size, context_length, embedding_dim]
```

Os valores começam como parâmetros ajustáveis. A informação semântica não vem
do número do ID; ela é aprendida quando os embeddings são atualizados durante o
treinamento.

## 5. Positional Embeddings

Um embedding de token sozinho não informa em qual posição o token aparece. Sem
posição, sequências com os mesmos tokens em ordens diferentes poderiam ter
representações muito parecidas.

Para cada posição do contexto, uma segunda tabela fornece um vetor com a mesma
dimensão dos embeddings dos tokens:

```text
token_embedding       [batch_size, context_length, embedding_dim]
position_embedding    [context_length, embedding_dim]
resultado              [batch_size, context_length, embedding_dim]
```

O resultado é obtido pela soma dos dois vetores. Nesta Sprint, será utilizada a
forma aprendida apresentada no capítulo, usando uma segunda camada
`nn.Embedding` para as posições.

## 6. Dataset, DataLoader e lotes

O `Dataset` encapsula as amostras de entrada e alvo e define como uma amostra
individual é recuperada. O `DataLoader` agrupa essas amostras em lotes, pode
embaralhá-las e controla o descarte do último lote incompleto.

O lote entregue ao modelo deve conter:

```text
x_batch: [batch_size, context_length]
y_batch: [batch_size, context_length]
```

Depois da camada de embedding e da informação posicional, `x_batch` será
transformado em uma entrada tridimensional para as camadas seguintes.

## 7. Relação entre as etapas

As etapas não são independentes:

```text
corpus
  ↓
tokenizador
  ↓
tokens e vocabulário
  ↓
Token IDs
  ↓
janelas de contexto + alvos deslocados
  ↓
Dataset/DataLoader
  ↓
embeddings de tokens + embeddings de posição
  ↓
entrada numérica do modelo
```

Um erro em uma etapa altera todas as dimensões e valores posteriores. Por
exemplo, um tokenizador que separa incorretamente um token especial muda o
vocabulário, os IDs, a quantidade de amostras e os vetores consultados.

## 8. Preparação para a próxima Sprint

O mecanismo de atenção receberá os vetores posicionais produzidos aqui. Ele
trabalhará sobre a dimensão de contexto e a dimensão dos embeddings para
calcular relações entre tokens. Portanto, esta Sprint deve entregar lotes
consistentes, com tamanhos fixos e alvos alinhados, antes da implementação da
atenção.
