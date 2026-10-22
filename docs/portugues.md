# Adaptação didática do GPT para português

## Usar o modelo já treinado

Na raiz do repositório:

```powershell
.\.venv\Scripts\python.exe experiments\gerar_texto.py --language pt --interactive
```

Exemplos de entrada:

```text
olá, tudo bem?
O que você faz?
o que é um token?
```

O comando carrega `experiments/gpt_portugues_v2.pt`. Use `/sair` para encerrar.
Por padrão, a seleção é greedy, que ajuda a reproduzir os exemplos. Para
amostrar tokens, acrescente `--sample --temperature 0.8 --top-k 10`.

As perguntas acima estão nos dados de treinamento e servem como demonstração
do ajuste aos exemplos. Responder a essas perguntas não comprova generalização
para perguntas novas. A resposta impressa é produzida pelo GPT; o programa não
consulta uma tabela de respostas nem um tradutor.

## Treinar novamente em português

```powershell
.\.venv\Scripts\python.exe experiments\treinar_gpt.py --language pt
```

O perfil usa `corpus_dialogos_pt_treino.txt` e a validação explícita de
`corpus_dialogos_pt_validacao.txt`. Os componentes continuam nas pastas
existentes. Os arquivos produzidos são:

- `experiments/gpt_portugues_v2.pt`: pesos, vocabulário, normalização e metadados;
- `experiments/treinamento_metricas_pt_v2.json`: histórico local das perdas;
- `docs/resultados_treinamento_portugues_v2.md`: relatório reproduzível.

O perfil revisado usa embedding 96, quatro heads, dois blocos,
FFN 256, contexto 64, lote 16, dropout 0,1 e taxa de aprendizado 0,001. O limite é de
100 épocas, com parada após 15 épocas sem melhora na validação. A semente é
123 e o treino utiliza uma thread de CPU, salvo configuração explícita.

Cada diálogo completo forma uma amostra. As perguntas fornecem contexto, mas
somente os tokens das respostas e do marcador de fim participam da perda.
O preenchimento à direita de lotes com tamanhos diferentes é ignorado no cálculo
da perda. Nesse perfil, `--stride` não controla as amostras: ele só se aplica
ao treinamento de continuação usado no perfil em inglês.

Cada execução inicia pesos novos. Para preservar uma variante anterior, use
`--checkpoint`, `--metrics` e `--report` com outros nomes. O arquivo do modelo
em inglês, `experiments/gpt_treinado.pt`, é independente do perfil português.
O checkpoint inicial em português, `experiments/gpt_portugues.pt`, também é
preservado para comparação. Os checkpoints são arquivos locais ignorados pelo Git; uma cópia nova do
repositório precisa executar o treino para produzi-los.

## Origem e organização do corpus

O corpus é **sintético e didático**, escrito para este repositório durante a
adaptação. Não é uma tradução do conto original, um conjunto coletado de
pessoas ou uma base representativa de conversas reais. O arquivo original
`corpus_portugues.txt` foi preservado. A revisão acrescenta variações de
perguntas, cumprimentos, pedidos diretos e exemplos supervisionados de somas.

Para reproduzir os arquivos de dados:

```powershell
.\.venv\Scripts\python.exe experiments\preparar_dialogos_portugues.py
```

O gerador escreve 1.440 diálogos de treino, 228 de validação e 228 de teste.
As respostas são rótulos de treinamento e ficam somente no preparo dos dados
e na avaliação; o programa de geração executa o GPT e não importa essas tabelas.

Cada documento possui esta estrutura:

```text
<|user|> olá, tudo bem?
<|assistant|> olá! tudo bem. ... <|endoftext|>
```

O tokenizador preserva cada marcador como uma unidade. A separação é explícita
em arquivos: não há perguntas iguais após normalização e tokenização entre as
divisões. Os assuntos, perguntas-base e respostas são compartilhados; várias
variações mudam apenas prefixos. Essa avaliação controlada permite estudar
consistência em paráfrases e prefixos, sem demonstrar generalização aberta.
O conjunto de teste não participa da criação do vocabulário, do cálculo de
gradientes ou da seleção do checkpoint.

O vocabulário é construído só no treino. Palavras exclusivas da validação
viram `<|unk|>` e sua quantidade é registrada no relatório. As métricas deste
corpus não devem ser comparadas diretamente com as do conto em inglês, pois
o vocabulário, os exemplos e a arquitetura são diferentes.

## O que mudou na implementação

- `src/config.py`: perfis `en` e `pt` com corpus, parâmetros e caminhos próprios;
- `src/tokenizer.py`: normalização opcional para minúsculas e Unicode NFC,
  preservando acentos, e separação opcional de operadores para reconhecer
  `1+1` e `1 + 1` da mesma forma. O perfil antigo mantém o comportamento anterior;
- `src/training.py` e `src/dataset.py`: divisão por documentos, amostras de
  diálogo e máscara de perda para respostas e preenchimento;
- `src/checkpoint.py`: armazenamento da normalização. Checkpoints anteriores,
  sem esse campo, continuam carregando com a regra original;
- `src/gpt.py`: parada opcional ao gerar o marcador de fim, inclusive em lotes;
- `src/generation.py`: formatação de usuário/assistente e retorno apenas dos
  tokens da resposta. Não há respostas fixas no código;
- comandos de treinamento e geração: seleção do idioma com `--language pt`.

Durante a inferência, o programa prepara
`<|user|> pergunta <|assistant|>`, prevê os tokens seguintes e encerra no
marcador de fim ou no limite configurado. O modo atual trata cada pergunta
separadamente; ele não mantém um histórico de conversa entre entradas.

## Limitações

Na primeira versão, com somente 95 diálogos de treino, a melhor época foi a 12. A perda de
validação das respostas caiu de 6,5647 para 5,2609. O teste interativo produziu:

| Entrada | Saída greedy observada |
|---|---|
| `olá, tudo bem?` | `sou um modelo de linguagem de linguagem.` |
| `O que você faz?` | `sou um modelo de linguagem de linguagem.` |
| `o que é um token?` | `um modelo de linguagem de linguagem.` |

Esses resultados da primeira versão demonstravam o carregamento e a geração no novo idioma, mas
também uma limitação concreta: o modelo produz respostas genéricas parecidas
para perguntas diferentes. Os testes de software não demonstram compreensão
das perguntas. O [relatório completo](resultados_treinamento_portugues.md)
registra a configuração e as perdas daquela execução. A versão revisada tem
seu próprio relatório de treinamento e uma avaliação de respostas livres.

Para reproduzir a avaliação após o treino:

```powershell
.\.venv\Scripts\python.exe experiments\avaliar_portugues.py
```

O comando gera `docs/avaliacao_portugues.md`, com respostas reais e a taxa de
correspondência exata nas 228 perguntas de teste controlado. Também verifica
as perguntas observadas no terminal, identificadas como vistas no treino.
Essa verificação de respostas é separada dos testes de software.

Na revisão, o checkpoint escolhido pela validação foi o da época 29. Ele
reproduziu corretamente os oito exemplos conhecidos usados na verificação,
incluindo as entradas enviadas no terminal. No teste controlado, acertou
87/228 respostas exatamente (38,16%): 78/100 nas somas e apenas 9/128 nas
demais perguntas. Isso mostra forte dependência da formulação dos exemplos.
A perda baixa de validação com a resposta correta fornecida posição a posição
não garante uma boa continuação livre para novas formulações.

As somas de 0 a 9 estão nos dados de treinamento. Acertos nessas contas podem
ser memorização e não comprovam um algoritmo de aritmética ou acertos em
números maiores. Nenhuma calculadora foi acrescentada à inferência.

O modelo continua pequeno, treinado com poucos exemplos. Pode repetir frases,
misturar explicações e errar mesmo em perguntas presentes no corpus. A
normalização resolve diferenças de caixa e de composição Unicode, mas não
torna duas palavras distintas equivalentes: `voce` e `você`, por exemplo,
precisam aparecer nos dados para serem conhecidas. Alguns cumprimentos sem
acento foram incluídos explicitamente.

O formato de diálogo ensina uma tarefa simples de pergunta e resposta. Isso
não equivale ao treinamento de um assistente de uso geral. Para medir qualidade
ampla em português seriam necessários dados mais variados, um conjunto de teste
independente e avaliação de respostas, além da perda de próximo token.

## Testes

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Os testes adicionais cobrem acentos, caixa, marcadores, divisão dos documentos,
compatibilidade com checkpoints antigos, persistência da normalização e parada
por fim de texto. O relatório registra as saídas reais do treinamento local.
