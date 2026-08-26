import torch
from torch.utils.data import Dataset, DataLoader

try:
    from .config import CORPUS_PATH, SPECIAL_TOKENS
    from .tokenizer import SimpleTokenizerV2, build_vocab, tokenize_text
except ImportError:  # Permite executar o arquivo diretamente a partir de src/.
    from config import CORPUS_PATH, SPECIAL_TOKENS
    from tokenizer import SimpleTokenizerV2, build_vocab, tokenize_text

# 1. Ler o texto e criar o vocabulário.
with CORPUS_PATH.open("r", encoding="utf-8") as f:
    raw_text = f.read()

# Separa palavras, pontuação e tokens especiais para extrair os tokens únicos.
preprocessed_text = tokenize_text(raw_text, SPECIAL_TOKENS)

# Cria uma lista ordenada sem palavras repetidas e gera o dicionário
vocab = build_vocab(preprocessed_text, SPECIAL_TOKENS)
all_tokens = list(vocab)

# CRIAÇÃO DA CLASSE GPTDATASETV1 
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        token_ids = tokenizer.encode(txt)
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

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
