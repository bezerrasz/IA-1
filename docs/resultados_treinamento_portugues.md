# Resultado do treinamento do GPT em português

Registro gerado por `experiments/treinar_gpt.py`. Este é um experimento
adicional ao escopo da Sprint 4.

## Configuração e dados

- Corpus: `corpus_portugues.txt`;
- Origem: sintético didático, escrito para este repositório;
- Tarefa: `dialogue`; normalização: `nfc_lower`;
- Escopo da perda: `assistant_tokens` (padding sempre ignorado);
- Dispositivo: `cpu`; Python 3.12.14; PyTorch 2.13.0+cpu; 1 thread(s);
- Tempo observado: 11.30 segundos;
- Parâmetros: 149120;
- Tokens: 2626 no treino e 691 na validação, separados antes da criação das janelas;
- Vocabulário: 609 tokens, construído apenas no treino;
- Tokens desconhecidos na validação: 84 (12.16%);
- SHA-256 do corpus: `08f0ee5053624fdf4057bc13ef72fa2f7440999bcedb0b74fa6803b5abce7eae`.

Configuração do modelo:

```json
{
  "vocab_size": 609,
  "context_length": 64,
  "embedding_dim": 64,
  "num_heads": 4,
  "num_layers": 2,
  "feed_forward_dim": 128,
  "dropout": 0.1,
  "bias": true
}
```

Configuração do treinamento:

```json
{
  "epochs": 100,
  "batch_size": 8,
  "learning_rate": 0.001,
  "weight_decay": 0.01,
  "train_fraction": 0.8,
  "stride": 16,
  "seed": 123,
  "patience": 12,
  "max_grad_norm": 1.0
}
```

## Perdas observadas

Cross-entropy média por token. As perdas são medidas com dropout desativado.
O checkpoint escolhido é o de menor perda de validação, incluindo a época 0.

| Época | Treino | Validação | Novo melhor checkpoint |
|---:|---:|---:|---|
| 0 | 6.5251 | 6.5647 | Inicial |
| 1 | 6.0735 | 6.2555 | Sim |
| 2 | 5.5622 | 5.8744 | Sim |
| 3 | 5.2247 | 5.6039 | Sim |
| 4 | 5.0134 | 5.4972 | Sim |
| 5 | 4.8607 | 5.4607 | Sim |
| 6 | 4.7153 | 5.4308 | Sim |
| 7 | 4.5536 | 5.3898 | Sim |
| 8 | 4.3770 | 5.3569 | Sim |
| 9 | 4.1849 | 5.3241 | Sim |
| 10 | 3.9924 | 5.2982 | Sim |
| 11 | 3.7940 | 5.2715 | Sim |
| 12 | 3.5881 | 5.2609 | Sim |
| 13 | 3.3816 | 5.2780 | Não |
| 14 | 3.1820 | 5.2843 | Não |
| 15 | 2.9740 | 5.2898 | Não |
| 16 | 2.7726 | 5.3019 | Não |
| 17 | 2.5787 | 5.3494 | Não |
| 18 | 2.3899 | 5.3891 | Não |
| 19 | 2.2111 | 5.4200 | Não |
| 20 | 2.0452 | 5.4964 | Não |
| 21 | 1.8791 | 5.5046 | Não |
| 22 | 1.7289 | 5.5058 | Não |
| 23 | 1.5874 | 5.6087 | Não |
| 24 | 1.4587 | 5.6691 | Não |

Melhor época: **12**. Perda de validação: **5.2609**.

## Geração antes e depois

Mesma entrada e seleção greedy nas duas execuções. No modo de diálogo,
a entrada é formatada com os marcadores de usuário e assistente.

Prompt: `olá, tudo bem?`

Antes:

```text
associação seu acontece nos nesse critérios pesos melhor ativação criação seja construir sabe assim observar amostras limitado pesos escore prevê divida local dúvidas previsão das após maior perceber tchau taxa indicam aprendeu alvos oferece conjunto deste instruções números aprendido salvo
```

Depois (melhor checkpoint):

```text
sou um modelo de linguagem de linguagem.
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
