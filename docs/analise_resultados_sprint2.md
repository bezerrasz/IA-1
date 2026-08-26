# Análise técnica dos resultados — Sprint 2

## Visão geral

O pipeline foi executado com o corpus `the-verdict.txt`, que possui 20.479
caracteres. O tokenizador simples produziu 4.690 tokens e um vocabulário de
1.132 itens, incluindo os dois tokens especiais reservados pelo projeto.

Os resultados confirmam que cada etapa altera a representação e as dimensões
dos dados. O texto deixa de ser uma sequência de caracteres, passa a ser uma
sequência de índices e, finalmente, uma sequência de vetores com informação de
posição.

## Por que um LLM não pode trabalhar diretamente com o texto bruto?

Texto bruto é uma estrutura simbólica formada por caracteres e regras de
codificação. As operações matriciais da rede neural precisam de valores
numéricos organizados em tensores com dimensões conhecidas.

A tokenização resolve a primeira parte do problema ao definir unidades de
processamento. O vocabulário e os Token IDs transformam essas unidades em
índices inteiros. Depois, a camada de embedding converte os índices em vetores
de ponto flutuante, que podem ser multiplicados e combinados pelas camadas
neurais.

Isso não significa que qualquer conversão numérica seja suficiente. Os números
precisam manter uma correspondência estável com o vocabulário e ser convertidos
em embeddings; caso contrário, a rede receberia apenas códigos sem uma
representação adequada para aprender relações.

## Qual é a função do vocabulário?

O vocabulário define o conjunto de tokens conhecidos e estabelece a tabela
`token ↔ Token ID`. No experimento, os 4.690 tokens do corpus foram reduzidos a
1.132 tokens distintos, mais os tokens especiais quando ainda não estavam
presentes.

Essa tabela é necessária para três operações:

1. codificar tokens textuais em IDs;
2. recuperar tokens a partir dos IDs;
3. definir `vocab_size`, que determina o número de linhas da matriz de
   embeddings.

Como consequência, o vocabulário precisa ser salvo ou reconstruído exatamente
com a mesma regra durante a inferência. Alterar a ordem dos IDs muda o
significado de todos os índices usados pelo modelo.

## Qual é a diferença entre um token e um Token ID?

Token é a unidade textual, como `"painting"`, `"."` ou `<|unk|>`. Token ID é o
inteiro atribuído a essa unidade pelo vocabulário.

O token é legível e pode ser usado para reconstruir texto. O ID é uma chave de
acesso. Por exemplo, se `"the"` estiver associado ao ID 10, o valor 10 só
representa `"the"` dentro daquele vocabulário específico.

## Por que os Token IDs não são utilizados diretamente como representação semântica?

IDs são valores discretos e arbitrários. Se um token recebe ID 10 e outro
recebe ID 11, isso não significa que os tokens sejam semanticamente próximos.
Uma operação numérica sobre os IDs trataria a diferença entre 10 e 11 como
significativa, embora a numeração tenha sido criada apenas para indexação.

A camada de embedding substitui cada ID por um vetor aprendível. As relações
úteis entre tokens são formadas pelos parâmetros dessa matriz durante o
treinamento, não pela distância entre seus IDs.

## Qual é a função dos embeddings?

Embeddings convertem IDs em vetores contínuos. Com `batch_size=4`, contexto 8 e
dimensão 64, o lote de IDs `[4, 8]` transforma-se em um lote vetorial `[4, 8,
64]`.

Os experimentos mostraram o custo dessa escolha. Mantendo batch e contexto
fixos, aumentar a dimensão de 16 para 128 elevou o tensor combinado de 512 para
4.096 elementos e a memória aproximada de 2 KiB para 16 KiB em `float32`.

Esses valores representam somente o tensor de saída, não todos os parâmetros
da matriz de embeddings, gradientes ou estados do otimizador. Mesmo assim,
demonstram que a dimensão influencia diretamente a memória e o custo das
operações posteriores.

## Por que é necessário representar a posição dos tokens?

O embedding de token informa qual unidade apareceu, mas não informa onde ela
apareceu. A ordem é essencial em linguagem: trocar a posição de tokens pode
mudar a interpretação da frase.

O `PositionalEmbedding` fornece um vetor para cada posição do contexto e o
`InputEmbedding` soma esse vetor ao embedding do token. O teste experimental
usou o mesmo ID repetido em quatro posições; os vetores produzidos para as duas
primeiras posições foram diferentes. Assim, o mesmo token passou a ter
representações distintas conforme sua posição.

Nesta Sprint foi usada a tabela posicional aprendível do capítulo. A dimensão
posicional é igual à dimensão do token embedding para que a soma seja possível.

## Qual é a relação entre tamanho do contexto e quantidade de amostras de treinamento?

Cada amostra precisa de `L` tokens de entrada e mais um token para o alvo
deslocado. Com `N` tokens, contexto `L` e stride `S`, a quantidade é:

```text
floor((N - (L + 1)) / S) + 1
```

No experimento, `N=4.690` e `S=4`:

| Contexto | Amostras |
| ---: | ---: |
| 4 | 1.172 |
| 8 | 1.171 |
| 16 | 1.169 |
| 32 | 1.165 |

Como o stride foi mantido fixo em 4, a redução foi moderada. Se o stride
acompanhar o contexto, as janelas terão menos sobreposição e o número de
amostras cairá mais rapidamente. Um contexto maior fornece mais informação a
cada previsão, mas reduz a quantidade de janelas possíveis e aumenta o custo
das representações e das camadas seguintes.

## Qual é o impacto da dimensão do embedding sobre as estruturas utilizadas pelo modelo?

A dimensão é a última dimensão do tensor de entrada. Para um lote com forma
`[B, T]`, a saída possui forma `[B, T, D]`, em que `D` é a dimensão do
embedding.

No experimento com `B=4` e `T=8`:

| Dimensão | Forma | Elementos | Memória aproximada |
| ---: | --- | ---: | ---: |
| 16 | `[4, 8, 16]` | 512 | 2 KiB |
| 32 | `[4, 8, 32]` | 1.024 | 4 KiB |
| 64 | `[4, 8, 64]` | 2.048 | 8 KiB |
| 128 | `[4, 8, 128]` | 4.096 | 16 KiB |

Mantendo `B` e `T` constantes, dobrar `D` dobra o número de elementos do
tensor. A dimensão maior pode oferecer mais capacidade de representação, mas
consome mais memória e aumenta o custo das camadas seguintes, inclusive da
atenção da próxima Sprint.

## Qual é a função do DataLoader no pipeline?

O `Dataset` cria e fornece uma amostra individual. O `DataLoader` transforma
essas amostras em lotes, controla a ordem de leitura e pode descartar o último
lote incompleto.

Com contexto 8 e stride 4, o conjunto possui 1.171 amostras. Os testes
registraram:

| Batch size | Lotes com último lote preservado |
| ---: | ---: |
| 2 | 586 |
| 4 | 293 |
| 8 | 147 |
| 16 | 74 |

O número de lotes diminui quando o batch aumenta, mas cada lote ocupa mais
memória. `drop_last=True` mantém lotes uniformes, enquanto `drop_last=False`
preserva as amostras restantes no último lote. Essa decisão deve ser registrada
porque altera quantos exemplos são usados em uma época.

## Quais informações produzidas nesta Sprint serão utilizadas pela atenção da próxima Sprint?

A próxima Sprint receberá principalmente:

- lotes de Token IDs com forma `[batch_size, context_length]`;
- alvos deslocados para cálculo da previsão do próximo token;
- embeddings de tokens;
- embeddings posicionais;
- embeddings combinados com forma `[batch_size, context_length, embedding_dim]`;
- `context_length`, necessário para limitar a sequência e construir a máscara
  causal;
- `embedding_dim`, que define a dimensão dos vetores consultados pela atenção.

A atenção calculará relações entre posições da sequência. Ela não deve receber
texto bruto nem IDs como se fossem vetores semânticos; deve receber a
representação vetorial com posição produzida nesta Sprint.

## Limitações observadas

O tokenizador atual é intencionalmente simples e baseado em palavras e
pontuação. Ele não implementa BPE, por isso textos fora do corpus podem gerar
`<|unk|>`. Nos exemplos testados, `"Hello, do you like tea?"` produziu 7 tokens
e 1 desconhecido, enquanto `"Jack Gisburn painted a portrait."` produziu 6
tokens e nenhum desconhecido.

Além disso, os embeddings ainda não foram treinados. Os vetores são parâmetros
inicializados pela camada e só adquirirão relações semânticas após a etapa de
treinamento. Os valores de memória apresentados são uma estimativa do tensor
de saída em `float32`, não do consumo total de um treinamento.

## Conclusão

Os experimentos confirmam o fluxo exigido pela Sprint 2. O texto é convertido
em tokens e IDs estáveis, organizado em pares autorregressivos, agrupado em
lotes e transformado em vetores que combinam conteúdo e posição. Essa saída é
a interface numérica necessária para iniciar a implementação do mecanismo de
atenção.
