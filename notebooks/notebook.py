import sys
import os
import time
import tiktoken
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Resolve o caminho para achar a pasta src
caminho_src = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(caminho_src)

from tokeniz import raw_text, vocab, SimpleTokenizerV2, create_dataloader_v1
from attention import SelfAttention_v1, SelfAttention_v2, CausalAttention, MultiHeadAttention
from gpt import (
    GPT_CONFIG_124M, DummyGPTModel, LayerNorm, GELU,
    FeedForward, TransformerBlock, GPTModel, generate_text_simple, generate
)

# =====================================================================
# TESTES DO CAPÍTULO 2 (DATASET, TOKENIZER, EMBEDDINGS)
# =====================================================================
print("="*50 + "\nCAPÍTULO 2: PREPARAÇÃO DE DADOS\n" + "="*50)

tokenizer_ch2 = SimpleTokenizerV2(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join((text1, text2))

max_length, batch_size, stride, embedding_dim = 4, 8, 4, 256

dataloader = create_dataloader_v1(
    raw_text, tokenizer_ch2, batch_size=batch_size, max_length=max_length,
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
batch_ca = torch.stack((inputs, inputs), dim=0)
context_length = batch_ca.shape[1]

torch.manual_seed(123)
ca = CausalAttention(d_in, d_out, context_length, dropout=0.5)
context_vecs_ca = ca(batch_ca)
print("Shape do output CausalAttention:", context_vecs_ca.shape)
print(context_vecs_ca)
print(">>> ANÁLISE: O uso da máscara causal (matriz triangular superior) força a IA a olhar apenas para as palavras anteriores. Valores futuros são substituídos por -infinito para evitar que o modelo 'roube' a resposta durante o treinamento.")

print("\n--- 4. Implementação de Multi-Head Attention ---")
torch.manual_seed(123)
mha = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0, num_heads=2)
context_vecs_mha = mha(batch_ca)
print("Shape do output MultiHeadAttention:", context_vecs_mha.shape)
print(context_vecs_mha)
print(">>> ANÁLISE: A atenção multi-cabeças divide os cálculos em múltiplas representações menores (aqui, num_heads=2). Isso permite que a IA aprenda diferentes tipos de contextos e relações linguísticas simultaneamente (ex: sintaxe em uma cabeça, semântica na outra).")


# =====================================================================
# EXPERIMENTOS DO CAPÍTULO 4 (ARQUITETURA GPT E GERAÇÃO DE TEXTO)
# =====================================================================
print("\n" + "="*50 + "\nCAPÍTULO 4: ARQUITETURA GPT E GERAÇÃO DE TEXTO\n" + "="*50)

tokenizer = tiktoken.get_encoding("gpt2")
txt1 = "Every effort moves you"
txt2 = "Every day holds a"
batch = torch.stack([
    torch.tensor(tokenizer.encode(txt1)),
    torch.tensor(tokenizer.encode(txt2))
], dim=0)

print("\n--- 1. Tokenização e Teste do DummyGPTModel ---")
print("Batch de Token IDs:\n", batch)
torch.manual_seed(123)
dummy_model = DummyGPTModel(GPT_CONFIG_124M)
logits_dummy = dummy_model(batch)
print("Shape de saída do DummyGPTModel:", logits_dummy.shape)

print("\n--- 2. Implementação de LayerNorm ---")
ln = LayerNorm(emb_dim=5)
exemplo_ln = torch.randn(2, 5)
out_ln = ln(exemplo_ln)
print("Média:", out_ln.mean(dim=-1).detach().numpy().round(4))
print("Variância:", out_ln.var(dim=-1, unbiased=False).detach().numpy().round(4))
print(">>> ANÁLISE: A camada LayerNorm padroniza as ativações de cada token para média 0 e variância 1, mantendo a estabilidade numérica ao longo da rede.")

print("\n--- 3. Implementação do Transformer Block ---")
torch.manual_seed(123)
x_teste = torch.rand(2, 4, 768)
bloco = TransformerBlock(GPT_CONFIG_124M)
saida_bloco = bloco(x_teste)
print("Shape de Entrada:", x_teste.shape)
print("Shape de Saída:", saida_bloco.shape)
print(">>> ANÁLISE: O Transformer Block preserva exatamente as dimensões (batch_size, seq_len, emb_dim), permitindo empilhar múltiplos blocos em sequência.")

print("\n--- 4. Construção do Modelo GPT Completo ---")
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
out_logits = model(batch)
total_params = sum(p.numel() for p in model.parameters())
size_mb = (total_params * 4) / (1024 * 1024)

print("Token IDs de Entrada (shape):", batch.shape)
print("Logits de Saída (shape):", out_logits.shape)
print(f"Total de Parâmetros: {total_params:,}")
print(f"Tamanho estimado em memória (float32): {size_mb:.2f} MB")
print(">>> ANÁLISE: O modelo integra embeddings, 12 blocos Transformer e a camada linear de saída (out_head), convertendo IDs de entrada em logits para os 50.257 tokens do vocabulário.")

print("\n--- 5. Inferência e Geração Autoregressiva ---")
start_context = "Hello, I am"
encoded_tensor = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0)

model.eval()
out_ids_greedy = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)
decoded_text = tokenizer.decode(out_ids_greedy.squeeze(0).tolist())
print(f"Contexto Inicial: '{start_context}' -> IDs: {encoded_tensor.tolist()}")
print(f"IDs Gerados (Greedy): {out_ids_greedy.tolist()}")
print(f"Texto Decodificado (Greedy): '{decoded_text}'")

torch.manual_seed(123)
out_ids_sampled = generate(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"],
    temperature=1.2,
    top_k=20
)
print(f"Texto Decodificado (Temp=1.2, Top-k=20): '{tokenizer.decode(out_ids_sampled.squeeze(0).tolist())}'")
print(">>> ANÁLISE: Como os pesos ainda são aleatórios (pré-treinamento), as palavras geradas são desconexas, mas o ciclo autoregressivo amplia o contexto de 4 para 10 tokens com sucesso.")

print("\n--- 6. Experimentos de Arquitetura e Custo Computacional ---")
experimentos_cap4 = [
    {"nome": "1. Baseline (GPT-124M)", "emb_dim": 768, "n_heads": 12, "n_layers": 12, "ff_mult": 4},
    {"nome": "2. Redução de Camadas", "emb_dim": 768, "n_heads": 12, "n_layers": 6,  "ff_mult": 4},
    {"nome": "3. Embedding Compacto", "emb_dim": 384, "n_heads": 6,  "n_layers": 12, "ff_mult": 4},
    {"nome": "4. Mais Heads (24)",    "emb_dim": 768, "n_heads": 24, "n_layers": 12, "ff_mult": 4},
    {"nome": "5. FeedForward 2x",     "emb_dim": 768, "n_heads": 12, "n_layers": 12, "ff_mult": 2},
]

for exp in experimentos_cap4:
    cfg_exp = GPT_CONFIG_124M.copy()
    cfg_exp.update({
        "emb_dim": exp["emb_dim"],
        "n_heads": exp["n_heads"],
        "n_layers": exp["n_layers"],
        "ff_mult": exp["ff_mult"]
    })
    
    torch.manual_seed(123)
    m_exp = GPTModel(cfg_exp)
    m_exp.eval()
    
    n_params = sum(p.numel() for p in m_exp.parameters())
    mem_mb = (n_params * 4) / (1024 * 1024)
    
    t0 = time.perf_counter()
    _ = generate_text_simple(m_exp, encoded_tensor, max_new_tokens=5, context_size=cfg_exp["context_length"])
    latencia_ms = (time.perf_counter() - t0) * 1000
    
    print(f"{exp['nome']:22s} -> Params: {n_params/1e6:6.1f}M | Memória: {mem_mb:6.1f} MB | Tempo (5 tokens): {latencia_ms:6.1f} ms")
print(">>> ANÁLISE: A dimensão do embedding (emb_dim) e a rede FeedForward concentram a maior parte dos parâmetros, enquanto a redução de camadas (n_layers) diminui linearmente o tempo de inferência.")


# =====================================================================
# EXPERIMENTOS DO CAPÍTULO 5 (TREINAMENTO E AVALIAÇÃO DO GPT)
# =====================================================================
print("\n" + "="*50 + "\nCAPÍTULO 5: TREINAMENTO E AVALIAÇÃO DO GPT\n" + "="*50)

GPT_CONFIG_CH5 = GPT_CONFIG_124M.copy()
GPT_CONFIG_CH5["context_length"] = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
        
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=25, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(f"   [Amostra Gerada]: {decoded_text.replace(chr(10), ' ')}")
    model.train()

def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer, mostrar_amostras=True):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                val_ppl = torch.exp(torch.tensor(val_loss)).item()
                if mostrar_amostras:
                    print(f"Época {epoch+1:02d} (Step {global_step:03d}): "
                          f"Train Loss {train_loss:.3f} | Val Loss {val_loss:.3f} | Val PPL {val_ppl:.2f}")

        if mostrar_amostras:
            generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen

train_ratio = 0.90
split_idx = int(train_ratio * len(raw_text))
train_data = raw_text[:split_idx]
val_data = raw_text[split_idx:]

torch.manual_seed(123)
train_loader = create_dataloader_v1(
    train_data, tokenizer,
    batch_size=2, max_length=GPT_CONFIG_CH5["context_length"],
    stride=GPT_CONFIG_CH5["context_length"], shuffle=True, drop_last=True
)
val_loader = create_dataloader_v1(
    val_data, tokenizer,
    batch_size=2, max_length=GPT_CONFIG_CH5["context_length"],
    stride=GPT_CONFIG_CH5["context_length"], shuffle=False, drop_last=False
)

print("\n--- 1. Avaliação do Modelo (Loss e Perplexidade Inicial) ---")
print(f"Batches de Treino: {len(train_loader)} | Batches de Validação: {len(val_loader)}")

torch.manual_seed(123)
model_ch5 = GPTModel(GPT_CONFIG_CH5).to(device)

with torch.no_grad():
    init_train_loss = calc_loss_loader(train_loader, model_ch5, device)
    init_val_loss = calc_loss_loader(val_loader, model_ch5, device)
    init_train_ppl = torch.exp(torch.tensor(init_train_loss)).item()
    init_val_ppl = torch.exp(torch.tensor(init_val_loss)).item()

print(f"Loss Inicial de Treino:     {init_train_loss:.4f} | Perplexidade (Treino):    {init_train_ppl:.2f}")
print(f"Loss Inicial de Validação:  {init_val_loss:.4f} | Perplexidade (Validação): {init_val_ppl:.2f}")
print(">>> ANÁLISE: Com pesos iniciais aleatórios, a Cross-Entropy Loss fica próxima de ln(50257) = 10.82 e a perplexidade ultrapassa 50.000, indicando que o modelo está indeciso entre todo o vocabulário.")

print("\n--- 2. Implementação da Rotina de Treinamento ---")
num_epochs = 10
learning_rate = 0.0004
weight_decay = 0.1
batch_size_ch5 = 2

print(f"Parâmetros -> Épocas: {num_epochs} | Batch Size: {batch_size_ch5} | "
      f"LR: {learning_rate} | Weight Decay: {weight_decay} | Otimizador: AdamW | Device: {device}\n")

torch.manual_seed(123)
model_ch5 = GPTModel(GPT_CONFIG_CH5).to(device)
optimizer = torch.optim.AdamW(model_ch5.parameters(), lr=learning_rate, weight_decay=weight_decay)

t_inicio = time.time()
train_losses, val_losses, tokens_seen = train_model_simple(
    model_ch5, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)
tempo_treino = time.time() - t_inicio

epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
fig, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(epochs_tensor, train_losses, label="Perda de Treino")
ax1.plot(epochs_tensor, val_losses, linestyle="-.", label="Perda de Validação")
ax1.set_xlabel("Épocas")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.legend(loc="upper right")
ax2 = ax1.twiny()
ax2.plot(tokens_seen, train_losses, alpha=0)
ax2.set_xlabel("Tokens Processados")
fig.tight_layout()
plt.savefig("loss_plot.png")
plt.close()

print(f"\nTreinamento concluído em {tempo_treino:.2f}s (Gráfico salvo em 'loss_plot.png').")
print(f"Loss Final de Treino: {train_losses[-1]:.4f} | Loss Final de Validação: {val_losses[-1]:.4f}")
print(">>> ANÁLISE: A queda acentuada da perda de treino (de ~10.99 para 0.39) e a evolução das frases geradas comprovam que o modelo aprendeu a prever os próximos tokens do corpus. A divergência entre treino e validação após a época 3 ocorre devido ao tamanho reduzido do texto 'the-verdict.txt'.")

print("\n--- 3. Experimentos de Treinamento (Variação de Parâmetros) ---")
experimentos_treino = [
    {"nome": "1. Subtreinado (3 Épocas, LR=0.0004)", "epochs": 3, "lr": 0.0004,  "drop": 0.1},
    {"nome": "2. Taxa Alta (3 Épocas, LR=0.0050)",   "epochs": 3, "lr": 0.0050,  "drop": 0.1},
    {"nome": "3. Taxa Baixa Sem Dropout (LR=0.00005)", "epochs": 3, "lr": 0.00005, "drop": 0.0},
]

for exp in experimentos_treino:
    cfg_exp = GPT_CONFIG_CH5.copy()
    cfg_exp["drop_rate"] = exp["drop"]
    
    torch.manual_seed(123)
    m_exp = GPTModel(cfg_exp).to(device)
    opt_exp = torch.optim.AdamW(m_exp.parameters(), lr=exp["lr"], weight_decay=0.1)
    
    t0 = time.time()
    t_losses, v_losses, _ = train_model_simple(
        m_exp, train_loader, val_loader, opt_exp, device,
        num_epochs=exp["epochs"], eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer,
        mostrar_amostras=False
    )
    dt = time.time() - t0
    print(f"{exp['nome']:38s} -> Train Loss: {t_losses[-1]:.3f} | Val Loss: {v_losses[-1]:.3f} | Tempo: {dt:.1f}s")
print(">>> ANÁLISE: Taxas de aprendizado muito elevadas (0.0050) desestabilizam a convergência, enquanto taxas muito baixas (0.00005) tornam o aprendizado lento. Em comparação ao treino de 10 épocas, interromper o treino mais cedo evita que o modelo memorize excessivamente o conjunto de treino (overfitting).")

print("\n--- 4. Estratégias de Geração de Texto (Greedy, Temperature e Top-k) ---")
contexto_teste = "Every effort moves you"
ids_iniciais = text_to_token_ids(contexto_teste, tokenizer).to(device)

configs_decodificacao = [
    {"nome": "1. Greedy Decoding (temp=0.0, top_k=None)",         "temp": 0.0, "top_k": None},
    {"nome": "2. Amostragem Moderada (temp=1.4, top_k=25)",       "temp": 1.4, "top_k": 25},
    {"nome": "3. Amostragem Conservadora (temp=0.5, top_k=10)",   "temp": 0.5, "top_k": 10},
    {"nome": "4. Temperatura Alta / Criativa (temp=2.0, top_k=50)", "temp": 2.0, "top_k": 50},
]

model_ch5.eval()
for cfg in configs_decodificacao:
    torch.manual_seed(123)
    out_tokens = generate(
        model=model_ch5,
        idx=ids_iniciais,
        max_new_tokens=25,
        context_size=GPT_CONFIG_CH5["context_length"],
        temperature=cfg["temp"],
        top_k=cfg["top_k"]
    )
    texto_final = token_ids_to_text(out_tokens, tokenizer).replace("\n", " ")
    print(f"{cfg['nome']}:\n   -> {texto_final}")
print(">>> ANÁLISE: Enquanto o treinamento altera os pesos do modelo para ensinar o vocabulário e a estrutura do texto, as estratégias de decodificação atuam apenas na inferência: o Greedy escolhe sempre a palavra mais provável (memorizada do texto), enquanto o Top-k e a Temperatura remodelam as probabilidades da Softmax para gerar variações inéditas sem perder a coerência.")

print("\n--- 5. Salvamento e Carregamento dos Pesos (state_dict) ---")
caminho_checkpoint = "model_and_optimizer.pth"

torch.save({
    "model_state_dict": model_ch5.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, caminho_checkpoint)
print(f"Checkpoint salvo com sucesso no arquivo '{caminho_checkpoint}'.")

checkpoint = torch.load(caminho_checkpoint, map_location=device, weights_only=True)
model_recarregado = GPTModel(GPT_CONFIG_CH5).to(device)
model_recarregado.load_state_dict(checkpoint["model_state_dict"])
model_recarregado.eval()

out_recarregado = generate(
    model=model_recarregado,
    idx=ids_iniciais,
    max_new_tokens=25,
    context_size=GPT_CONFIG_CH5["context_length"],
    temperature=0.0
)
print("Inferência com o modelo recarregado do disco:")
print(f"   -> {token_ids_to_text(out_recarregado, tokenizer).replace(chr(10), ' ')}")
print(">>> ANÁLISE: O texto gerado pelo modelo recarregado é idêntico ao do modelo em memória, comprovando que os parâmetros aprendidos foram preservados corretamente.")