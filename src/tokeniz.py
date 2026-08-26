import torch
from torch.utils.data import DataLoader

try:
    from .config import CORPUS_PATH, SPECIAL_TOKENS
    from .dataset import GPTDatasetV1
    from .tokenizer import SimpleTokenizerV2, build_vocab, tokenize_text
except ImportError:  # Permite executar o arquivo diretamente a partir de src/.
    from config import CORPUS_PATH, SPECIAL_TOKENS
    from dataset import GPTDatasetV1
    from tokenizer import SimpleTokenizerV2, build_vocab, tokenize_text

# 1. Ler o texto e criar o vocabulário.
with CORPUS_PATH.open("r", encoding="utf-8") as f:
    raw_text = f.read()

# Separa palavras, pontuação e tokens especiais para extrair os tokens únicos.
preprocessed_text = tokenize_text(raw_text, SPECIAL_TOKENS)

# Cria uma lista ordenada sem palavras repetidas e gera o dicionário
vocab = build_vocab(preprocessed_text, SPECIAL_TOKENS)
all_tokens = list(vocab)

# CRIAÇÃO DA FUNÇÃO CREATE_DATALOADER_V1
def create_dataloader_v1(txt, tokenizer, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last
    )
    return dataloader
