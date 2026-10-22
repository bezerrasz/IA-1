# Análise dos resultados — Sprint 4

## Leitura dos experimentos de arquitetura

Os experimentos foram executados com o mesmo vocabulário de 1.132 tokens,
pesos inicializados com sementes controladas e `dropout=0.0`. O objetivo foi
comparar capacidade estrutural e custo do forward; como não houve treinamento,
as continuações geradas não representam qualidade linguística.

### Número de blocos

A configuração base possui 90.112 parâmetros e levou 0,865 ms no forward
medido. Ao passar de dois para quatro blocos, o modelo foi para 107.200
parâmetros e 1,598 ms. O aumento de aproximadamente 19% nos parâmetros e de
aproximadamente 85% no tempo observado confirma que cada bloco acrescenta uma
atenção, uma FFN e suas normalizações. A medição de tempo em modelos pequenos
é sensível ao sistema operacional, mas a tendência estrutural é esperada.

### Número de heads

Com `D=32` fixo, aumentar de quatro para oito heads manteve 90.112 parâmetros,
porque as projeções Q, K e V continuam tendo dimensão total `D`. Entretanto, o
tensor estimado de scores passou de 1.024 para 2.048 elementos (`B × H × T ×
T`). O número de heads muda a divisão dos subespaços e o paralelismo da
atenção, mas não necessariamente aumenta os parâmetros quando `D` permanece
constante.

### Dimensão do embedding

Ao dobrar `D` de 32 para 64 e ajustar a FFN de 64 para 128, o modelo passou a
212.992 parâmetros, contra 90.112 na base. O custo maior vem dos embeddings,
das projeções da atenção, da FFN e do LM Head, cuja saída continua tendo 1.132
logits por posição. A maior representação oferece mais capacidade, mas aumenta
memória e operações em quase todas as subcamadas.

### Dimensão da FeedForward Network

Mantendo `D=32` e aumentando `F` de 64 para 128, os parâmetros subiram para
98.432. O aumento é menor que o da dimensão do embedding porque afeta
principalmente as duas projeções da FFN; a atenção, os embeddings e o LM Head
permanecem inalterados. O resultado confirma que `F` controla a capacidade
não linear posição a posição.

### Tamanho do contexto

Ao passar de contexto 16 para 32, o tensor de scores estimado foi de 1.024
para 4.096 elementos, quadruplicando conforme `T²`. O forward observado passou
de 0,865 ms para 1,246 ms e os parâmetros foram de 90.112 para 90.624; essa
pequena diferença de parâmetros vem da tabela de embeddings posicionais, que
precisa armazenar mais posições. O resultado evidencia por que o contexto é
uma variável especialmente custosa na atenção.

## Leitura dos experimentos de geração

As três estratégias receberam o mesmo prompt `The court` e geraram quatro
tokens. O greedy produziu uma única continuação determinística. As duas
amostragens usaram `top_k=10` e temperaturas 0,7 e 1,3; neste caso específico,
com pesos não treinados e a semente registrada, produziram a mesma sequência.
Isso não significa que as temperaturas sejam equivalentes: a temperatura
altera a distribuição antes do sorteio, mas uma amostra isolada pode coincidir.

O `<|unk|>` na saída é coerente com o tokenizador simples do projeto. Como o
GPT ainda não recebeu treinamento, seus logits não expressam probabilidade
linguística aprendida; as diferenças de saída servem para verificar o fluxo de
seleção e a influência potencial dos parâmetros.

## Conclusão

Os cinco entregáveis foram integrados: o Transformer Block preserva a forma
`[B,T,D]`, o GPT empilha blocos e produz `[B,T,V]`, a geração usa a última
posição dos logits e os experimentos registram como `D`, `H`, quantidade de
blocos, `F` e `T` afetam parâmetros, scores e tempo. Para uma avaliação de
qualidade, a próxima etapa necessária seria treinar o modelo com os pares de
entrada e alvo da Sprint 2 e então comparar perda, perplexidade e amostras
geradas.

