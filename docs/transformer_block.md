# Entregável 2 — Transformer Block

## Estrutura implementada

O bloco está em `src/transformer.py` e usa a atenção causal de
`src/attention.py`. A configuração escolhida é **pre-norm**, na qual a
normalização ocorre antes de cada subcamada:

```text
x ──► LayerNorm ──► Multi-Head Attention ──► Dropout ──► + ──►
│                                                        │
└────────────────────────────────────────────────────────┘

   ──► LayerNorm ──► FeedForward(D→F→D) ──► Dropout ──► + ──► saída
    │                                                   │
    └───────────────────────────────────────────────────┘
```

Para uma entrada com forma `[B, T, D]`:

1. `LayerNorm` normaliza os `D` valores de cada posição e mantém `[B, T, D]`;
2. a atenção projeta consultas, chaves e valores e organiza os vetores como
   `[B, H, T, Dh]`, em que `Dh = D/H`;
3. a máscara triangular impede que a posição `t` leia posições maiores que
   `t`;
4. as cabeças são concatenadas e a projeção final retorna `[B, T, D]`;
5. a soma residual preserva a representação anterior;
6. a FFN transforma cada posição em `[B,T,F]`, aplica `GELU` e retorna a
   dimensão original `[B,T,D]`;
7. uma segunda soma residual produz a saída do bloco.

## Função de cada componente

- **Multi-Head Attention:** mistura informação entre posições do contexto e
  captura relações em `H` subespaços diferentes;
- **LayerNorm:** estabiliza a escala das ativações antes de cada subcamada;
- **FeedForward Network:** adiciona capacidade não linear sem misturar
  posições entre si;
- **GELU:** fornece a ativação suave da FFN;
- **Dropout:** regulariza pesos e ativações durante o treinamento;
- **Conexões residuais:** mantêm o caminho da entrada e facilitam o fluxo de
  gradientes.

## Decisões e validações

O módulo valida que `D` é divisível pelo número de heads, que a sequência não
excede `context_length` e que as dimensões de entrada são tridimensionais. Os
testes em `tests/test_transformer.py` verificam forma, configuração inválida,
preservação do contexto e o comportamento causal: alterar tokens futuros não
altera as saídas das posições anteriores.

Embora a branch publicada ainda não contenha os arquivos da Sprint 3, a
atenção foi isolada em um módulo próprio e documentada como camada de
integração. Assim, a Sprint 4 possui a dependência executável necessária sem
alterar a estrutura de pastas do projeto.

