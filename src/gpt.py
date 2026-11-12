import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention

GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Tamanho do vocabulário (tiktoken gpt2)
    "context_length": 1024, # Tamanho máximo do contexto (Cap. 4)
    "emb_dim": 768,         # Dimensão dos embeddings
    "n_heads": 12,          # Número de cabeças de atenção
    "n_layers": 12,         # Número de Transformer Blocks
    "drop_rate": 0.1,       # Taxa de Dropout
    "qkv_bias": False,      # Bias nas matrizes Query, Key e Value
    "ff_mult": 4            # Multiplicador da camada oculta FeedForward (padrão 4x)
}

# =====================================================================
# SEÇÃO 4.1: ESQUELETO INICIAL (DUMMY)
# =====================================================================
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
    def forward(self, x):
        return x

# =====================================================================
# SEÇÕES 4.2 E 4.3: LAYERNORM, GELU E FEEDFORWARD NETWORK
# =====================================================================
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # x shape: (batch_size, seq_len, emb_dim)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ff_mult = cfg.get("ff_mult", 4)
        hidden_dim = ff_mult * cfg["emb_dim"]
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], hidden_dim), # Expande: (B, T, emb_dim) -> (B, T, 4*emb_dim)
            GELU(),
            nn.Linear(hidden_dim, cfg["emb_dim"]), # Comprime: (B, T, 4*emb_dim) -> (B, T, emb_dim)
        )

    def forward(self, x):
        return self.layers(x)

# =====================================================================
# SEÇÃO 4.5 (ENTREGÁVEL 2): TRANSFORMER BLOCK
# =====================================================================
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Bloco 1: Pre-LayerNorm -> Multi-Head Attention -> Dropout -> Conexão Residual
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Bloco 2: Pre-LayerNorm -> FeedForward -> Dropout -> Conexão Residual
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x

# =====================================================================
# SEÇÃO 4.6 (ENTREGÁVEL 3): ARQUITETURA GPT COMPLETA
# =====================================================================
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # 1. Entrada in_idx: (batch_size, seq_len)
        batch_size, seq_len = in_idx.shape
        
        # 2. Embeddings de token e posição: (batch_size, seq_len, emb_dim)
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        
        # 3. Passagem pelos N Transformer Blocks: mantém (batch_size, seq_len, emb_dim)
        x = self.trf_blocks(x)
        
        # 4. Normalização final e Language Model Head: (batch_size, seq_len, vocab_size)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# =====================================================================
# SEÇÃO 4.7 (ENTREGÁVEL 4): INFERÊNCIA E GERAÇÃO AUTOREGRESSIVA
# =====================================================================
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        # Corta o contexto se exceder context_size suportado pelo pos_emb
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
        
        # Foca apenas no último passo de tempo: (B, T, vocab_size) -> (B, vocab_size)
        logits = logits[:, -1, :]
        
        # Converte logits em probabilidades via Softmax
        probas = torch.softmax(logits, dim=-1)
        
        # Seleção Greedy (token com maior probabilidade): (B, 1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        
        # Concatena o token previsto ao contexto para a próxima iteração
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx

# =====================================================================
# CAPÍTULO 5 (ENTREGÁVEL 2): AVALIAÇÃO QUANTITATIVA (LOSS)
# =====================================================================
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    # Achata os tensores para (batch_size * seq_len, vocab_size) e (batch_size * seq_len)
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