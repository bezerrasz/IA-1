# Entregável 4 — Inferência e geração de texto

## Caminho da previsão

O método `GPTModel.forward()` produz logits com forma `[B,T,V]`. Para obter o
próximo token, o procedimento usa somente a última posição:

```python
logits = model(context_ids)[:, -1, :]  # [B, V]
probabilities = torch.softmax(logits / temperature, dim=-1)
next_token = torch.multinomial(probabilities, num_samples=1)
```

Logits são escores reais não normalizados. O softmax transforma esses escores
em uma distribuição de probabilidades cuja soma é 1. A temperatura é aplicada
antes do softmax: valores menores concentram a distribuição nos maiores
logits, enquanto valores maiores deixam a amostragem mais variada.

## Estratégias implementadas

O método `GPTModel.generate()` oferece duas formas de seleção:

- **Greedy decoding (`do_sample=False`):** escolhe o token com maior logit por
  `argmax`. É determinístico para um modelo em modo de avaliação;
- **Amostragem (`do_sample=True`):** aplica temperatura e sorteia segundo a
  distribuição de probabilidades;
- **Top-k (`top_k=k`):** antes do softmax, substitui por `-inf` todos os logits
  que não estão entre os `k` maiores. Pode ser combinado com temperatura.

Se o contexto crescer além do limite do modelo, a geração usa somente os
`context_length` tokens mais recentes. Isso mantém as posições válidas para o
embedding posicional e para a máscara causal.

## Laço autorregressivo

Para cada token novo, o método executa:

1. recorta o contexto recente;
2. calcula os logits da sequência;
3. seleciona o próximo Token ID;
4. concatena o ID selecionado ao contexto;
5. repete até produzir `max_new_tokens` tokens.

```text
prompt [t0,t1]
      │
      ├─ forward → logits da posição t1 → seleciona t2
      └─ contexto [t0,t1,t2]
                    │
                    ├─ forward → seleciona t3
                    └─ ...
```

`src/generation.py` fornece `generate_text()`, que faz a ponte entre a
interface textual e o método numérico: chama `tokenizer.encode(prompt)`, cria
um tensor `[1,T]`, executa a geração e chama `tokenizer.decode()` no resultado.

## Observação sobre os resultados

O modelo desta Sprint é inicializado, mas não treinado. Portanto, a geração
serve para validar o fluxo e comparar estratégias, não para avaliar qualidade
linguística. A distribuição inicial depende dos pesos aleatórios; depois de um
treinamento, os mesmos mecanismos passarão a refletir padrões aprendidos do
corpus.

