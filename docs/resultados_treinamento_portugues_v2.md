# Resultado do treinamento do GPT em português

Registro gerado por `experiments/treinar_gpt.py`. Este é um experimento
adicional ao escopo da Sprint 4.

## Configuração e dados

- Corpus: `corpus_dialogos_pt_treino.txt`;
- Divisão: `explicit_validation`; validação: `corpus_dialogos_pt_validacao.txt`;
- Origem: sintético didático, escrito para este repositório;
- Tarefa: `dialogue`; normalização: `nfc_lower`;
- Escopo da perda: `assistant_tokens` (padding sempre ignorado);
- Dispositivo: `cpu`; Python 3.12.14; PyTorch 2.13.0+cpu; 1 thread(s);
- Tempo observado: 205.83 segundos;
- Parâmetros: 316544;
- Tokens: 26744 no treino e 5117 na validação, separados antes da criação das janelas;
- Vocabulário: 708 tokens, construído apenas no treino;
- Tokens desconhecidos na validação: 6 (0.12%);
- SHA-256 do corpus: `f4be30954a6d7709a1d91f57970d0887c8c12e27a7829e83084403b141531ed3`.

Configuração do modelo:

```json
{
  "vocab_size": 708,
  "context_length": 64,
  "embedding_dim": 96,
  "num_heads": 4,
  "num_layers": 2,
  "feed_forward_dim": 256,
  "dropout": 0.1,
  "bias": true
}
```

Configuração do treinamento:

```json
{
  "epochs": 100,
  "batch_size": 16,
  "learning_rate": 0.001,
  "weight_decay": 0.01,
  "train_fraction": 0.8,
  "stride": 16,
  "seed": 123,
  "patience": 15,
  "max_grad_norm": 1.0
}
```

## Perdas observadas

Cross-entropy média por token. As perdas são medidas com dropout desativado.
O checkpoint escolhido é o de menor perda de validação, incluindo a época 0.

| Época | Treino | Validação | Novo melhor checkpoint |
|---:|---:|---:|---|
| 0 | 6.7348 | 6.7250 | Inicial |
| 1 | 3.6203 | 3.9219 | Sim |
| 2 | 2.4901 | 2.7421 | Sim |
| 3 | 1.7397 | 1.8814 | Sim |
| 4 | 1.1716 | 1.2000 | Sim |
| 5 | 0.7573 | 0.7249 | Sim |
| 6 | 0.4889 | 0.4321 | Sim |
| 7 | 0.3088 | 0.2657 | Sim |
| 8 | 0.2138 | 0.1839 | Sim |
| 9 | 0.1536 | 0.1424 | Sim |
| 10 | 0.1142 | 0.1116 | Sim |
| 11 | 0.0905 | 0.0987 | Sim |
| 12 | 0.0714 | 0.0819 | Sim |
| 13 | 0.0521 | 0.0687 | Sim |
| 14 | 0.0387 | 0.0583 | Sim |
| 15 | 0.0281 | 0.0576 | Sim |
| 16 | 0.0195 | 0.0495 | Sim |
| 17 | 0.0138 | 0.0423 | Sim |
| 18 | 0.0152 | 0.0472 | Não |
| 19 | 0.0120 | 0.0442 | Não |
| 20 | 0.0081 | 0.0392 | Sim |
| 21 | 0.0085 | 0.0379 | Sim |
| 22 | 0.0067 | 0.0361 | Sim |
| 23 | 0.0059 | 0.0336 | Sim |
| 24 | 0.0041 | 0.0370 | Não |
| 25 | 0.0065 | 0.0361 | Não |
| 26 | 0.0032 | 0.0388 | Não |
| 27 | 0.0026 | 0.0357 | Não |
| 28 | 0.0025 | 0.0329 | Sim |
| 29 | 0.0028 | 0.0294 | Sim |
| 30 | 0.0036 | 0.0335 | Não |
| 31 | 0.0027 | 0.0314 | Não |
| 32 | 0.0021 | 0.0334 | Não |
| 33 | 0.0049 | 0.0332 | Não |
| 34 | 0.0032 | 0.0349 | Não |
| 35 | 0.0025 | 0.0363 | Não |
| 36 | 0.0021 | 0.0298 | Não |
| 37 | 0.0029 | 0.0331 | Não |
| 38 | 0.0029 | 0.0350 | Não |
| 39 | 0.0023 | 0.0327 | Não |
| 40 | 0.0033 | 0.0352 | Não |
| 41 | 0.0017 | 0.0374 | Não |
| 42 | 0.0019 | 0.0336 | Não |
| 43 | 0.0013 | 0.0349 | Não |
| 44 | 0.0013 | 0.0317 | Não |

Melhor época: **29**. Perda de validação: **0.0294**.

## Geração antes e depois

Mesma entrada e seleção greedy nas duas execuções. No modo de diálogo,
a entrada é formatada com os marcadores de usuário e assistente.

Prompt: `olá, tudo bem?`

Antes:

```text
carregue modo desativado ola checkpoint erros perceber diga recebe residual previsões transformar outro que alterar greedy novas organização guardar depois python preservar demonstrou nos muda checkpoints processamento contribuir treinados usamos feedforward conhecidas tarefas textos dependem operações história história valeu materiais
```

Depois (melhor checkpoint):

```text
olá! tudo bem. como posso ajudar?
```

## Interpretação e limitações

A queda da perda de treino demonstra ajuste aos exemplos apresentados.
A validação usa um trecho separado e orienta a escolha do checkpoint e
a parada antecipada. Se a perda de treino continuar caindo enquanto a
validação piora, há sinal de sobreajuste.

O corpus é curto e em português. O modelo pode aprender repetições e
memorizar trechos; este resultado não o torna um assistente de conversa.
Palavras exclusivas da validação viram `<|unk|>`, portanto parte da
perda avalia esse marcador, e não a previsão da palavra original.

O checkpoint binário contém os pesos, a configuração e o vocabulário.
Ele fica no computador e não é versionado no Git. As métricas completas
ficam no JSON local produzido pelo mesmo comando.
