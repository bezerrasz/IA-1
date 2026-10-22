# Resultado do treinamento inicial do GPT

Registro gerado por `experiments/treinar_gpt.py`. Este é um experimento
adicional ao escopo da Sprint 4.

## Configuração e dados

- Corpus: `the-verdict.txt`;
- Dispositivo: `cpu`; Python 3.12.14; PyTorch 2.13.0+cpu; 1 thread(s);
- Tempo observado: 19.25 segundos;
- Parâmetros: 83072;
- Tokens: 3752 no treino e 938 na validação, separados antes da criação das janelas;
- Vocabulário: 1022 tokens, construído apenas no treino;
- Tokens desconhecidos na validação: 119 (12.69%);
- SHA-256 do corpus: `b41e41a68f0398a3154ae69e2e4c0e2694e17fe0d66730536837f1b01935b31f`.

Configuração do modelo:

```json
{
  "vocab_size": 1022,
  "context_length": 16,
  "embedding_dim": 32,
  "num_heads": 4,
  "num_layers": 2,
  "feed_forward_dim": 64,
  "dropout": 0.1,
  "bias": true
}
```

Configuração do treinamento:

```json
{
  "epochs": 20,
  "batch_size": 8,
  "learning_rate": 0.0003,
  "weight_decay": 0.01,
  "train_fraction": 0.8,
  "stride": 8,
  "seed": 123,
  "patience": 5,
  "max_grad_norm": 1.0
}
```

## Perdas observadas

Cross-entropy média por token. As perdas são medidas com dropout desativado.
O checkpoint escolhido é o de menor perda de validação, incluindo a época 0.

| Época | Treino | Validação | Novo melhor checkpoint |
|---:|---:|---:|---|
| 0 | 7.1253 | 7.1092 | Inicial |
| 1 | 6.9170 | 6.9560 | Sim |
| 2 | 6.5519 | 6.6794 | Sim |
| 3 | 6.1617 | 6.3551 | Sim |
| 4 | 5.8983 | 6.1354 | Sim |
| 5 | 5.7443 | 6.0156 | Sim |
| 6 | 5.6665 | 5.9666 | Sim |
| 7 | 5.6262 | 5.9521 | Sim |
| 8 | 5.5980 | 5.9482 | Sim |
| 9 | 5.5693 | 5.9470 | Sim |
| 10 | 5.5327 | 5.9430 | Sim |
| 11 | 5.4862 | 5.9333 | Sim |
| 12 | 5.4284 | 5.9128 | Sim |
| 13 | 5.3667 | 5.8918 | Sim |
| 14 | 5.3011 | 5.8746 | Sim |
| 15 | 5.2347 | 5.8557 | Sim |
| 16 | 5.1660 | 5.8436 | Sim |
| 17 | 5.0988 | 5.8357 | Sim |
| 18 | 5.0293 | 5.8219 | Sim |
| 19 | 4.9600 | 5.8128 | Sim |
| 20 | 4.8928 | 5.8145 | Não |

Melhor época: **19**. Perda de validação: **5.8128**.

## Geração antes e depois

Mesma sequência inicial e seleção greedy nas duas execuções.

Prompt: `I HAD always`

Antes:

```text
I HAD always sweetly) exquisitely crossed panelling went glimpse work flash live back white-panelled <|endoftext|> what under most height) equanimity abruptly curious live abruptly which
```

Depois (melhor checkpoint):

```text
I HAD always, and of" I had," I was the mantel-piece the mantel-piece the mantel-piece the mantel-piece the mantel-piece the mantel-piece the mantel-piece
```

## Interpretação e limitações

A queda da perda de treino demonstra ajuste aos exemplos apresentados.
A validação usa um trecho separado e orienta a escolha do checkpoint e
a parada antecipada. Se a perda de treino continuar caindo enquanto a
validação piora, há sinal de sobreajuste.

O corpus é curto e em inglês. O modelo pode aprender repetições e
memorizar trechos; este resultado não o torna um assistente de conversa.
Palavras exclusivas da validação viram `<|unk|>`, portanto parte da
perda avalia esse marcador, e não a previsão da palavra original.

O checkpoint binário contém os pesos, a configuração e o vocabulário.
Ele fica no computador e não é versionado no Git. As métricas completas
ficam no JSON local produzido pelo mesmo comando.
