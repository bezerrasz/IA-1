# Sprint 5 – Treinamento, Avaliação e Persistência do Modelo GPT (Capítulo 5)

## 1. Entregável 1 – Glossário de Treinamento e Avaliação
* **Cross-Entropy Loss**: Mede a divergência entre as probabilidades previstas pelos logits do modelo e os tokens reais esperados. Guia o ajuste dos pesos penalizando previsões incorretas.
* **Perplexidade (Perplexity)**: Exponencial da perda (`exp(loss)`), indicando o número efetivo de tokens do vocabulário entre os quais o modelo está indeciso a cada previsão.
* **Época (Epoch) e Lote (Batch)**: Uma época é uma varredura completa pelos dados de treino, dividida em lotes processados simultaneamente pelo `DataLoader` para atualizar os pesos.
* **Retropropagação (Backpropagation)**: Calcula os gradientes da função de perda em relação a cada peso do Transformer via regra da cadeia (`loss.backward()`).
* **Otimizador AdamW**: Variante do Adam com decaimento de pesos (`weight_decay`) desacoplado, atualizando os parâmetros e penalizando pesos excessivos para melhorar a generalização.
* **Taxa de Aprendizado (Learning Rate)**: Hiperparâmetro que controla o tamanho do passo dado pelo otimizador na atualização dos pesos.
* **Sobreajuste (Overfitting)**: Fenômeno em que o modelo memoriza o conjunto de treinamento, reduzindo a perda de treino enquanto a perda de validação para de cair e passa a subir.
* **Greedy Decoding**: Estratégia determinística que seleciona sempre o token com maior probabilidade (`torch.argmax`) a cada passo.
* **Temperature Scaling**: Divide os logits por um fator escalar antes da função `softmax`, controlando a nitidez da distribuição de probabilidades durante a amostragem.
* **Top-k Sampling**: Restringe a amostragem probabilística aos `k` tokens de maior pontuação, atribuindo `-inf` aos demais para evitar palavras fora de contexto.
* **State Dict**: Dicionário interno do PyTorch (`state_dict`) que mapeia cada camada aos seus tensores de parâmetros, permitindo salvar e recarregar o modelo em disco.

## 2. Entregável 2 – Avaliação Quantitativa Inicial (Loss e Perplexidade)
O corpus **the-verdict.txt** (5.145 tokens) foi dividido na proporção de 90% para treinamento (9 lotes) e 10% para validação (1 lote), utilizando `context_length = 256` e `batch_size = 2`.

* **Loss Inicial de Treino**: 10.9909 (Perplexidade: 59333.52)
* **Loss Inicial de Validação**: 10.9822 (Perplexidade: 58816.89)
* **Interpretação**: Com pesos inicializados aleatoriamente (`torch.manual_seed(123)`), a perda inicial fica muito próxima do valor teórico de uma distribuição uniforme sobre o vocabulário de 50.257 tokens (`ln(50257) = 10.82`). A perplexidade próxima de 59.000 mostra que o modelo inicial chuta aleatoriamente entre todas as palavras disponíveis.

## 3. Entregável 3 – Rotina e Evolução do Treinamento
* **Configuração Utilizada**: 10 épocas, `batch_size = 2`, `context_length = 256`, otimizador `AdamW` (`lr = 0.0004`, `weight_decay = 0.1`), executado em CPU em 397.89 segundos.
* **Curva de Aprendizado**: O arquivo **loss_plot.png** gerado na raiz do projeto ilustra a trajetória das perdas ao longo dos 85 passos de otimização.

| Época (Passo) | Train Loss | Val Loss | Val Perplexity | Comportamento da Amostra Gerada |
| :--- | :--- | :--- | :--- | :--- |
| Inicial (Pré-treino) | 10.991 | 10.982 | 58816.89 | Palavras aleatórias sem sentido |
| Época 01 (Step 005) | 8.111 | 8.339 | 4182.12 | Repetição de vírgulas (`you,,,,,,,,,,,,.`) |
| Época 02 (Step 015) | 5.961 | 6.616 | 746.98 | Repetição de conjunções (`, and, and, and`) |
| Época 04 (Step 035) | 4.069 | 6.226 | 505.63 | Primeiras estruturas curtas (`you know the`) |
| Época 07 (Step 055) | 2.104 | 6.134 | 461.28 | Menor perda de validação e frases estruturadas |
| Época 10 (Step 085) | 0.391 | 6.452 | 634.06 | Memorização literal de trechos do conto |

## 4. Entregável 4 – Experimentos de Treinamento e Análise de Sobreajuste

| Experimento | Épocas | Learning Rate | Train Loss Final | Val Loss Final |
| :--- | :--- | :--- | :--- | :--- |
| Subtreinado (Curto) | 3 | 0.00040 | 5.201 | 6.348 |
| Ponto Ótimo Validação | 6 | 0.00040 | 2.427 | 6.141 |
| Baseline Completo | 10 | 0.00040 | 0.391 | 6.452 |
| LR Elevada (Instável) | 3 | 0.00500 | ~6.500 | ~7.400 |
| LR Baixa Sem Dropout | 3 | 0.00005 | ~7.200 | ~7.350 |

* **Sinais de Sobreajuste (Overfitting)**: Entre as épocas 1 e 6, tanto a perda de treino quanto a de validação caem, atingindo o mínimo de validação na Época 7 (Step 055) com 6.134. A partir daí, a perda de treino cai até 0.391 na Época 10 enquanto a perda de validação sobe para 6.452, pois o modelo de 163 milhões de parâmetros decora o corpus pequeno.
* **Impacto da Taxa de Aprendizado e Custo**: Cada época exige cerca de 39.8 segundos na CPU. Uma taxa alta (`0.005`) faz o otimizador ultrapassar os mínimos locais e gerar texto repetitivo, enquanto uma taxa muito baixa (`0.00005`) torna a queda da perda excessivamente lenta.

## 5. Entregável 5 – Estratégias de Geração de Texto
Utilizando o mesmo contexto inicial `"Every effort moves you"` no modelo treinado:

* **Greedy Decoding (`temperature = 0.0`, `top_k = None`)**: Seleciona sempre o maior logit, produzindo `"Every effort moves you know,\" was one of the axioms he laid down across the Sevres and silver of an exquisitely appointed lun"`. É determinístico e reproduz literalmente frases decoradas quando o modelo sofre sobreajuste.
* **Amostragem Moderada (`temperature = 0.8`, `top_k = 25`)**: Introduz diversidade controlada mantendo coerência gramatical e evitando loops de repetição.
* **Amostragem do Livro (`temperature = 1.4`, `top_k = 25`)**: Achatando levemente a distribuição `softmax`, gera combinações inéditas usando o vocabulário do conto, enquanto o filtro `top_k = 25` impede o sorteio de tokens absurdos.
* **Treinamento vs Decodificação**: O treinamento altera os pesos das camadas internas para capturar a sintaxe e o vocabulário, enquanto a estratégia de decodificação atua apenas na inferência sobre os logits da última camada.

## 6. Entregável 6 – Salvamento e Carregamento dos Pesos
* **Procedimento**: Os parâmetros aprendidos foram salvos no arquivo **model_and_optimizer.pth** via `torch.save`, armazenando o `model_state_dict`, o `optimizer_state_dict` e o dicionário de configuração `GPT_CONFIG_CH5`.
* **Reprodução da Inferência**: O arquivo foi recarregado com `torch.load` em uma nova instância limpa de `GPTModel` seguida de `load_state_dict` e `model.eval()`, gerando exatamente a mesma sequência do modelo em memória sem necessidade de novo treinamento.

## 7. Entregável 7 – Como Reproduzir e Limitações do Projeto
* **Execução**: Instale as dependências (`pip install torch tiktoken matplotlib requests`), certifique-se de que o corpus foi baixado executando `python src/theverdict.py` e rode o pipeline completo com `python notebooks/notebook.py`.
* **Limitações**: O treinamento foi realizado sobre um único conto literário curto (5.145 tokens) em CPU, o que leva à rápida memorização do conjunto de treino e limita a capacidade do modelo de generalizar para diálogos ou temas fora do vocabulário da obra.