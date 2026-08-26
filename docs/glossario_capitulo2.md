# Glossário Técnico — Capítulo 2

## Token

- **Original:** Token
- **Português:** Token ou unidade de texto
- **Definição:** Unidade menor em que o texto é dividido para ser processado.
Pode ser uma palavra, parte de palavra, número ou símbolo.
- **Função no modelo:** É a unidade básica que será convertida em um índice e
depois em um vetor.
- **Relações:** É produzido pela tokenização, recebe um Token ID no vocabulário
e origina um embedding.
- **Exemplo:** Em `"Hello, world!"`, um tokenizador simples pode produzir
`["Hello", ",", "world", "!"]`.

## Tokenization

- **Original:** Tokenization
- **Português:** Tokenização
- **Definição:** Processo de dividir um texto em tokens.
- **Função no modelo:** Converte a entrada textual em uma sequência que pode ser
codificada numericamente.
- **Relações:** É a primeira transformação do pipeline e determina quais itens
entrarão no vocabulário.
- **Exemplo:** A expressão regular do projeto separa palavras, pontuação e
travessões.

## Tokenizer

- **Original:** Tokenizer
- **Português:** Tokenizador
- **Definição:** Componente que implementa a tokenização e, normalmente, as
operações de codificação e decodificação.
- **Função no modelo:** Mantém a regra de conversão entre texto, tokens e IDs.
- **Relações:** Usa o vocabulário e pode tratar tokens especiais e desconhecidos.
- **Exemplo:** `SimpleTokenizerV2` implementa `encode()` e `decode()`.

## Vocabulary

- **Original:** Vocabulary
- **Português:** Vocabulário
- **Definição:** Conjunto de tokens conhecidos e tabela que associa cada token a
um identificador inteiro.
- **Função no modelo:** Define quais tokens podem ser representados diretamente
e qual é o tamanho da matriz de embeddings.
- **Relações:** É construído a partir dos tokens e fornece a relação
`token ↔ Token ID`.
- **Exemplo:** `{"the": 10, ".": 11}`.

## Token ID

- **Original:** Token ID
- **Português:** Identificador numérico do token
- **Definição:** Inteiro usado como índice de um token dentro do vocabulário.
- **Função no modelo:** Permite consultar o vetor correspondente na camada de
embedding.
- **Relações:** Não possui significado semântico próprio; depende do vocabulário
e é convertido em embedding.
- **Exemplo:** Se `"the"` está associado a `10`, seu Token ID é `10`.

## Encode e Decode

- **Original:** Encode / Decode
- **Português:** Codificar / Decodificar
- **Definição:** `encode` transforma texto em IDs; `decode` reconstrói tokens ou
texto a partir dos IDs.
- **Função no modelo:** Faz a ponte entre a interface textual e a representação
numérica usada durante o treinamento e a geração.
- **Relações:** Dependem do mesmo vocabulário para que a conversão seja
consistente.
- **Exemplo:** `encode("the") -> [10]` e `decode([10]) -> "the"`.

## Special Token

- **Original:** Special token
- **Português:** Token especial
- **Definição:** Token reservado para representar uma condição de controle ou
um caso que não é uma palavra comum.
- **Função no modelo:** Delimita documentos, marca desconhecidos ou representa
outros sinais necessários ao pipeline.
- **Relações:** Deve estar no vocabulário e ser preservado pelo tokenizador.
- **Exemplo:** `<|endoftext|>` separa textos e `<|unk|>` representa um token
ausente do vocabulário.

## Unknown Token (`<|unk|>`)

- **Original:** Unknown token / UNK
- **Português:** Token desconhecido
- **Definição:** Token especial usado quando uma unidade textual não existe no
vocabulário.
- **Função no modelo:** Evita que uma palavra fora do vocabulário interrompa a
codificação da sequência.
- **Relações:** É escolhido durante `encode()` e recebe seu próprio ID.
- **Exemplo:** Se `"computer"` não estiver no vocabulário,
`encode("computer")` pode gerar o ID de `<|unk|>`.

## End-of-Text Token (`<|endoftext|>`)

- **Original:** End-of-text token
- **Português:** Token de fim de texto
- **Definição:** Marcador reservado que indica o encerramento de um documento
ou a separação entre trechos.
- **Função no modelo:** Ajuda o modelo a distinguir limites entre documentos e
contextos diferentes.
- **Relações:** Precisa ser reconhecido antes da tokenização genérica e incluído
no vocabulário.
- **Exemplo:** `texto_a + <|endoftext|> + texto_b`.

## Byte Pair Encoding (BPE)

- **Original:** Byte Pair Encoding
- **Português:** Codificação por pares de bytes
- **Definição:** Algoritmo que combina unidades frequentes para formar tokens de
subpalavras.
- **Função no modelo:** Permite cobrir palavras novas com partes conhecidas e
reduzir a quantidade de tokens desconhecidos.
- **Relações:** É uma alternativa mais robusta ao tokenizador simples por
palavras usado nesta Sprint.
- **Exemplo:** Uma palavra rara pode ser dividida em subpalavras já presentes no
vocabulário. O projeto não implementa BPE nesta etapa.

## Context Length

- **Original:** Context length / Context size
- **Português:** Tamanho ou comprimento de contexto
- **Definição:** Número máximo de tokens presentes em cada sequência de entrada.
- **Função no modelo:** Define quanto texto o modelo observa em uma amostra e
o tamanho da tabela de embeddings posicionais.
- **Relações:** Afeta a quantidade de amostras, o custo computacional e a forma
dos tensores.
- **Exemplo:** Com `context_length = 4`, cada entrada possui quatro IDs.

## Stride

- **Original:** Stride
- **Português:** Passo da janela
- **Definição:** Quantidade de tokens que a janela de contexto avança ao criar a
próxima amostra.
- **Função no modelo:** Controla a sobreposição entre amostras.
- **Relações:** Junto com o tamanho do contexto, determina a quantidade de
sequências produzidas.
- **Exemplo:** Contexto 4 e stride 2 produzem janelas sobrepostas:
`[0,1,2,3]`, `[2,3,4,5]`.

## Sliding Window

- **Original:** Sliding window
- **Português:** Janela deslizante
- **Definição:** Estratégia de percorrer os Token IDs em blocos de tamanho fixo.
- **Função no modelo:** Cria diversas amostras de treinamento a partir de um
único corpus contínuo.
- **Relações:** Usa `context_length` e `stride` e gera pares de entrada e alvo.
- **Exemplo:** Uma janela de entrada `[t0,t1,t2]` possui alvo `[t1,t2,t3]`.

## Input Chunk

- **Original:** Input chunk
- **Português:** Bloco de entrada
- **Definição:** Subsequência de Token IDs fornecida ao modelo como contexto.
- **Função no modelo:** É a sequência sobre a qual o modelo fará previsões.
- **Relações:** É produzida pela janela deslizante e convertida em embeddings.
- **Exemplo:** `input_chunk = [12, 8, 31, 4]`.

## Target Chunk

- **Original:** Target chunk
- **Português:** Bloco alvo
- **Definição:** Sequência esperada pelo treinamento, deslocada uma posição em
relação ao bloco de entrada.
- **Função no modelo:** Fornece o próximo token correto para calcular a perda.
- **Relações:** Tem o mesmo tamanho do input chunk e representa o objetivo
autorregressivo.
- **Exemplo:** Para entrada `[12, 8, 31, 4]`, o alvo pode ser `[8, 31, 4, 19]`.

## Autoregressive Language Modeling

- **Original:** Autoregressive language modeling
- **Português:** Modelagem de linguagem autorregressiva
- **Definição:** Forma de modelagem em que cada token é previsto a partir dos
tokens anteriores.
- **Função no modelo:** Define por que os alvos são deslocados em uma posição.
- **Relações:** Usa sequências de entrada, alvos e, na próxima Sprint, máscara
causal na atenção.
- **Exemplo:** Dado `"I like"`, o alvo pode ser o token correspondente a
`"tea"`.

## Embedding

- **Original:** Token embedding
- **Português:** Incorporação ou representação vetorial do token
- **Definição:** Vetor contínuo associado a cada Token ID por uma matriz
aprendível.
- **Função no modelo:** Transforma índices discretos em valores que as camadas
neurais podem processar.
- **Relações:** A matriz tem forma `[vocab_size, embedding_dim]` e será ajustada
no treinamento.
- **Exemplo:** `nn.Embedding(vocab_size, embedding_dim)(ids)`.

## Embedding Dimension

- **Original:** Embedding dimension
- **Português:** Dimensão do embedding
- **Definição:** Quantidade de valores numéricos em cada vetor de token.
- **Função no modelo:** Determina a largura da representação e das camadas que
receberão esses vetores.
- **Relações:** Deve ser igual à dimensão dos embeddings posicionais e afeta
memória e custo computacional.
- **Exemplo:** Com dimensão 256, cada token é representado por 256 valores.

## Positional Embedding

- **Original:** Positional embedding
- **Português:** Embedding posicional
- **Definição:** Vetor associado à posição ocupada por um token dentro do
contexto.
- **Função no modelo:** Injeta informação de ordem na representação dos tokens.
- **Relações:** É somado ao token embedding e possui a mesma dimensão vetorial.
- **Exemplo:** Para contexto 4 e dimensão 256, a tabela posicional tem forma
`[4, 256]`.

## Batch

- **Original:** Batch
- **Português:** Lote
- **Definição:** Grupo de amostras processadas em conjunto.
- **Função no modelo:** Melhora o aproveitamento do hardware e permite calcular
gradientes sobre várias sequências de uma vez.
- **Relações:** Seu tamanho é definido por `batch_size` no DataLoader.
- **Exemplo:** O lote de IDs possui forma `[8, 4]` quando há 8 amostras com
contexto 4.

## Dataset

- **Original:** Dataset
- **Português:** Conjunto de dados
- **Definição:** Abstração que armazena ou fornece amostras e implementa o acesso
a uma amostra e o tamanho do conjunto.
- **Função no modelo:** Encapsula os pares de entrada e alvo.
- **Relações:** É consumido pelo DataLoader.
- **Exemplo:** `GPTDatasetV1[idx]` retorna `(input_ids, target_ids)`.

## DataLoader

- **Original:** DataLoader
- **Português:** Carregador de dados
- **Definição:** Utilitário do PyTorch que agrupa, embaralha e itera sobre um
Dataset em lotes.
- **Função no modelo:** Entrega lotes padronizados ao treinamento sem exigir que
todo o corpus seja carregado de uma vez na etapa de processamento.
- **Relações:** Usa `batch_size`, `shuffle` e `drop_last` e fornece os dados que
serão convertidos em embeddings.
- **Exemplo:** `DataLoader(dataset, batch_size=8, shuffle=True)`.

## Drop Last

- **Original:** `drop_last`
- **Português:** Descartar último lote incompleto
- **Definição:** Opção que remove o último lote quando ele possui menos amostras
que os demais.
- **Função no modelo:** Mantém dimensões de lote uniformes durante os
experimentos.
- **Relações:** É um parâmetro do DataLoader e pode reduzir o número total de
amostras efetivamente usadas em uma época.
- **Exemplo:** Com 10 amostras e `batch_size=4`, `drop_last=True` usa dois lotes
completos e descarta duas amostras.
