import re
import os
import torch
from torch.utils.data import Dataset, DataLoader

# 1. LER O TEXTO E CRIAR O VOCABULÁRIO (O passo que estava faltando)
diretorio_atual = os.path.dirname(__file__)
caminho_texto = os.path.join(diretorio_atual, "..", "the-verdict.txt")

with open(caminho_texto, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Separa todas as palavras e pontuações para extrair os tokens únicos
preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed_text = [item.strip() for item in preprocessed_text if item.strip()]

# Cria uma lista ordenada sem palavras repetidas e gera o dicionário
all_tokens = sorted(list(set(preprocessed_text)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}

# CRIAÇÃO DA CLASSE SIMPLETOKENIZERV2 
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

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