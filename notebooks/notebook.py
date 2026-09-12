import sys
import os
import torch

# Resolve o caminho para achar a pasta src
caminho_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(caminho_src)

from tokeniz import raw_text, vocab, SimpleTokenizerV2, create_dataloader_v1
from attention import SelfAttention_v1, SelfAttention_v2, CausalAttention, MultiHeadAttention

# =====================================================================
# TESTES DO CAPÍTULO 2 (DATASET, TOKENIZER, EMBEDDINGS)
# =====================================================================
print("="*50 + "\nCAPÍTULO 2: PREPARAÇÃO DE DADOS\n" + "="*50)

tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

max_length, batch_size, stride, embedding_dim = 4, 8, 4, 256

dataloader = create_dataloader_v1(
    raw_text, tokenizer, batch_size=batch_size, max_length=max_length,
    stride=stride, shuffle=False, drop_last=True,
)
data_iter = iter(dataloader)
inputs_ch2, targets_ch2 = next(data_iter)

vocab_size = len(vocab)
token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
pos_embedding_layer = torch.nn.Embedding(max_length, embedding_dim)
input_embeddings = token_embedding_layer(inputs_ch2) + pos_embedding_layer(torch.arange(max_length))

print("Shape final dos Embeddings (Ch2):", input_embeddings.shape)


# =====================================================================
# EXPERIMENTOS DO CAPÍTULO 3 (MECANISMOS DE ATENÇÃO)
# =====================================================================
print("\n" + "="*50 + "\nCAPÍTULO 3: MECANISMOS DE ATENÇÃO\n" + "="*50)

# VETOR DE ENTRADAS SIMULADO (6 tokens, d_in=3)
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)
d_in = inputs.shape[1]
d_out = 2


print("\n--- 1. Implementação de Scaled Dot-Product Attention (Passo a Passo) ---")
attn_scores = inputs @ inputs.T
print("Matriz de Atenção (Scores):\n", attn_scores)

attn_weights = torch.softmax(attn_scores, dim=-1)
print("\nPesos de Atenção (Normalizados):\n", attn_weights)

all_context_vecs = attn_weights @ inputs
print("\nVetores de Contexto Finais:\n", all_context_vecs)
print(">>> ANÁLISE: A matriz de atenção revela as relações matemáticas entre todos os tokens da frase simultaneamente. A normalização com Softmax garante que os pesos (importância) somem 1.0 (100%).")


print("\n--- 2. Implementação de Self-Attention (v1 e v2) ---")
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print("Output SelfAttention_v1 (Matriz de Contexto):\n", sa_v1(inputs))

torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
print("\nOutput SelfAttention_v2 (Usando nn.Linear):\n", sa_v2(inputs))
print(">>> ANÁLISE: Ambas as classes geram vetores de contexto a partir dos pesos treináveis (W_q, W_k, W_v). A V2 substitui parâmetros manuais por 'nn.Linear', tornando o código do PyTorch otimizado e preparado para backpropagation.")


print("\n--- 3. Implementação de Causal Attention ---")
batch = torch.stack((inputs, inputs), dim=0) # Simulando um Batch com 2 sequências
context_length = batch.shape[1]

torch.manual_seed(123)
ca = CausalAttention(d_in, d_out, context_length, dropout=0.5)
context_vecs_ca = ca(batch)
print("Shape do output CausalAttention:", context_vecs_ca.shape)
print(context_vecs_ca)
print(">>> ANÁLISE: O uso da máscara causal (matriz triangular superior) força a IA a olhar apenas para as palavras anteriores. Valores futuros são substituídos por -infinito para evitar que o modelo 'roube' a resposta durante o treinamento.")


print("\n--- 4. Implementação de Multi-Head Attention ---")
torch.manual_seed(123)
mha = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0, num_heads=2)
context_vecs_mha = mha(batch)
print("Shape do output MultiHeadAttention:", context_vecs_mha.shape)
print(context_vecs_mha)
print(">>> ANÁLISE: A atenção multi-cabeças divide os cálculos em múltiplas representações menores (aqui, num_heads=2). Isso permite que a IA aprenda diferentes tipos de contextos e relações linguísticas simultaneamente (ex: sintaxe em uma cabeça, semântica na outra).")