# Treinamento e uso dos pesos aprendidos

Esta etapa acrescenta treinamento ao GPT da Sprint 4, reutilizando o
tokenizador, o Dataset, os embeddings e os Transformer Blocks existentes.
Os módulos continuam nas pastas `src/`, `experiments/`, `tests/` e `docs/`.

## Primeiro treinamento

Na raiz do repositório, com o ambiente virtual preparado:

```powershell
.\.venv\Scripts\python.exe experiments\treinar_gpt.py
```

O comando usa `the-verdict.txt`, executa até 20 épocas e informa a perda de
treino e validação a cada época. Uma época é uma passagem por todas as janelas
do conjunto de treino. Uma thread de CPU é usada por padrão, adequada às
matrizes pequenas deste experimento. CUDA é selecionada automaticamente se
estiver disponível na instalação do PyTorch; `--device cpu` força a CPU.

Os resultados são gravados em:

- `experiments/gpt_treinado.pt`: melhor checkpoint observado na validação;
- `experiments/treinamento_metricas.json`: histórico completo da execução;
- `docs/resultados_treinamento.md`: configuração, métricas e geração antes/depois.

Pesos e JSON de execução ficam locais e são ignorados pelo Git. O relatório
Markdown permite registrar os resultados sem publicar arquivos binários.

## Gerar texto com o modelo treinado

```powershell
.\.venv\Scripts\python.exe experiments\gerar_texto.py --prompt "I HAD always"
```

Para digitar diferentes frases no próprio terminal:

```powershell
.\.venv\Scripts\python.exe experiments\gerar_texto.py --interactive
```

Digite `/sair` para encerrar. Esse modo gera continuações, não respostas de um
chatbot treinado com instruções. O corpus é em inglês; palavras fora do
vocabulário aparecem como `<|unk|>`.

Para amostrar com temperatura e top-k:

```powershell
.\.venv\Scripts\python.exe experiments\gerar_texto.py --prompt "I HAD always" --sample --temperature 0.8 --top-k 10 --max-new-tokens 30
```

O programa carrega arquitetura, pesos e vocabulário do checkpoint. Nenhum
download é necessário. O experimento original `sprint4_experimentos.py`
continua comparando modelos com pesos aleatórios; use `gerar_texto.py` para
testar o aprendizado salvo.

## Procedimento de treinamento

1. O texto é tokenizado e dividido em 80% de tokens para treino e 20% para
   validação, mantendo a ordem original.
2. O vocabulário é construído somente com os tokens de treino. Os desconhecidos
   na validação são contados no relatório.
3. As janelas são criadas separadamente em cada conjunto. Nenhuma janela de
   treino cruza a fronteira da validação.
4. O DataLoader embaralha apenas as janelas de treino. O último lote incompleto
   é preservado.
5. O GPT produz logits `[B,T,V]`, comparados com os alvos deslocados `[B,T]`
   pela cross-entropy. Não se aplica softmax antes dessa função.
6. `backward()` calcula os gradientes, que são limitados em norma a 1,0; o
   AdamW ajusta os parâmetros com taxa inicial 0,0003 e weight decay 0,01.
7. Ao fim de cada época, a avaliação é executada sem gradientes e com dropout
   desativado. A média é ponderada pela quantidade de tokens de cada lote.
8. O menor erro de validação determina o checkpoint. Após cinco épocas sem
   melhora, o treino termina antecipadamente. Os pesos iniciais também são
   candidatos, caso nenhuma época melhore a validação.

O checkpoint guarda configuração, vocabulário, tokens especiais, pesos, época,
perda de validação, parâmetros do treino, hash do corpus e estado do otimizador.
A gravação é feita em arquivo temporário e então substituída atomicamente.
O carregamento usa `weights_only=True` e reconstrói exatamente a configuração
salva. O comando atual inicia treinos novos; não oferece retomada do otimizador.

## Alterar a configuração

```powershell
.\.venv\Scripts\python.exe experiments\treinar_gpt.py --epochs 30 --learning-rate 0.001 --context-length 32 --embedding-dim 64 --feed-forward-dim 128 --checkpoint experiments\gpt_alternativo.pt
```

Essa é uma nova execução desde a inicialização aleatória. Reexecutar sem trocar
`--checkpoint` substitui os pesos daquele caminho, e o JSON e o relatório
padrão também são atualizados. Para preservar todas as métricas de uma variante,
informe também `--metrics` e `--report` com outros nomes nas pastas existentes.
Consulte `--help` para as opções completas.

## Como interpretar o aprendizado

Uma perda menor no treino indica melhor ajuste às amostras apresentadas. A
validação mede um trecho não utilizado para ajustar os pesos. Se o treino
melhora e a validação piora, há sinal de sobreajuste. O checkpoint selecionado
permite voltar ao melhor resultado observado nesse conjunto.

O conto é curto para aprender linguagem de forma ampla. Pode ocorrer memorização
de trechos e repetição de palavras. Como o vocabulário foi criado só no treino,
parte da validação avalia a previsão de `<|unk|>` em vez da palavra original;
essa proporção é registrada para contextualizar as métricas. A avaliação de
validação serve à seleção do modelo e não substitui um conjunto de teste
independente para medir sua generalização final.

## Verificações automatizadas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Os testes adicionais verificam queda da perda em uma sequência previsível,
separação dos dados e do vocabulário, média por token com lotes incompletos,
parada antecipada, escolha da melhor época e igualdade das previsões após
salvar/carregar os pesos e o vocabulário.
