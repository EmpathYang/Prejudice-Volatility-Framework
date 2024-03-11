from torch.utils.data import Dataset
import os
import pandas as pd
import torch

from utils import TEMPLATE_DIR, DATA_DIR

class PCFDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.T, self.T_dist = self.load_csv(os.path.join(TEMPLATE_DIR, "TemplatesFor"+args.T+".csv"), "T")
        self.X, self.X_dist = self.load_csv(os.path.join(DATA_DIR, args.X+".csv"), "X")
        self.Y = [y.strip().split(",") for y in open(os.path.join(DATA_DIR, args.Y+".txt"), "r").readlines()]

    def __len__(self):
        return len(self.T) * len(self.X)

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__ function.")
    
    def data_collator(self, features):
        raise NotImplementedError("Subclasses must implement data_collator function.")
    
    def get_P_matrix_shape(self):
        return (len(self.X), len(self.T), len(self.Y))
    
    def get_T_dist(self):
        return self.T_dist
    
    def get_X_dist(self):
        return self.X_dist

    def load_csv(self, csv_filepath, column_key):
        df = pd.read_csv(csv_filepath)
        return df[column_key].to_list(), df["distribution"].to_numpy()
    
class BertDataset(PCFDataset):
    def __init__(self, args, tokenizer):
        super().__init__(args)
        self.mask_token_num = 5
        self.tokenizer = tokenizer
        self.class_indices = self.get_class_indices()

    def __getitem__(self, idx):
        T_idx = idx // len(self.X)
        X_idx = idx % len(self.X)
        return self.T[T_idx].replace("[X]", self.X[X_idx]).replace("[Y]", "[MASK]"*self.mask_token_num), (X_idx, T_idx)

    def get_class_indices(self):
        class_indices = []
        self.tokenizer.padding_side = "right"
        for y in self.Y:
            tokenized_features = self.tokenizer(
                y,
                truncation=True,
                padding=True,
                max_length=self.mask_token_num,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]
            if tokenized_features.shape[-1] < self.mask_token_num:
                tokenized_features = torch.cat([tokenized_features, self.tokenizer.pad_token_id*torch.ones((tokenized_features.shape[0], self.mask_token_num-tokenized_features.shape[-1]), dtype=tokenized_features.dtype)], dim=-1)
            class_indices.append(tokenized_features)
        return class_indices
    
    def data_collator(self, features):
        texts = [f[0] for f in features]
        X_idx = [f[1][0] for f in features]
        T_idx = [f[1][1] for f in features]
        self.tokenizer.padding_side = "left"
        tokenized_features = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return tokenized_features, X_idx, T_idx
    
    def compute_probability(self, logits):
        # logits (batch_size, seq_num, vocab_size)
        # return (batch_size, Y_class_num)
        masked_logits = logits[:, -(1+self.mask_token_num):-1, :]
        masked_probs = torch.softmax(masked_logits, dim=-1)
        masked_prob_logs = torch.log(masked_probs)
        
        class_probs = []
        for masked_prob_log in masked_prob_logs:
            class_probs.append([])
            for class_indice in self.class_indices:
                log_buffer = masked_prob_log.unsqueeze(0)
                log_buffer = log_buffer.repeat(class_indice.shape[0], 1, 1)
                class_prob_logs = torch.gather(log_buffer, -1, class_indice.unsqueeze(-1))
                class_prob_logs[torch.where(class_indice==self.tokenizer.pad_token_id)] = 0
                class_prob_logs = torch.sum(class_prob_logs.squeeze(-1), dim=-1)
                class_prob_logs /= torch.count_nonzero(class_indice, dim=-1)
                class_prob = torch.exp(torch.sum(class_prob_logs)).detach().item()
                class_probs[-1].append(class_prob)
        
        return class_probs

class GptDataset(PCFDataset):
    def __init__(self, args, tokenizer):
        super().__init__(args)
        self.mask_token_num = 5
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.class_indices = self.get_class_indices()

    def __len__(self):
        return len(self.T) * len(self.X)

    def __getitem__(self, idx):
        T_idx = idx // len(self.X)
        X_idx = idx % len(self.X)
        return self.T[T_idx].replace("[X]", self.X[X_idx]).replace("[Y]", " ".join(["..."]*self.mask_token_num)), (X_idx, T_idx)

    def get_class_indices(self):
        class_indices = []
        self.tokenizer.padding_side = "right"
        for y in self.Y:
            tokenized_features = self.tokenizer(
                y,
                truncation=True,
                padding=True,
                max_length=self.mask_token_num,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]
            if tokenized_features.shape[-1] < self.mask_token_num:
                tokenized_features = torch.cat([tokenized_features, self.tokenizer.pad_token_id*torch.ones((tokenized_features.shape[0], self.mask_token_num-tokenized_features.shape[-1]), dtype=tokenized_features.dtype)], dim=-1)
            class_indices.append(tokenized_features)
        return class_indices
    
    def data_collator(self, features):
        texts = [f[0] for f in features]
        X_idx = [f[1][0] for f in features]
        T_idx = [f[1][1] for f in features]
        self.tokenizer.padding_side = "left"
        tokenized_features = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return tokenized_features, X_idx, T_idx
    
    def compute_probability(self, logits):
        # logits (batch_size, seq_num, vocab_size)
        # return (batch_size, Y_class_num)
        masked_logits = logits[:, -(1+self.mask_token_num):-1, :]
        masked_probs = torch.softmax(masked_logits, dim=-1)
        masked_prob_logs = torch.log(masked_probs)
        
        class_probs = []
        for masked_prob_log in masked_prob_logs:
            class_probs.append([])
            for class_indice in self.class_indices:
                log_buffer = masked_prob_log.unsqueeze(0)
                log_buffer = log_buffer.repeat(class_indice.shape[0], 1, 1)
                class_prob_logs = torch.gather(log_buffer, -1, class_indice.unsqueeze(-1))
                class_prob_logs[torch.where(class_indice==self.tokenizer.pad_token_id)] = 0
                class_prob_logs = torch.sum(class_prob_logs.squeeze(-1), dim=-1)
                class_prob_logs /= torch.count_nonzero(class_indice, dim=-1)
                class_prob = torch.exp(torch.sum(class_prob_logs)).detach().item()
                class_probs[-1].append(class_prob)
        
        return class_probs

  
if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    args = argparse.Namespace(T="Gender", X="occupation", Y="gender", batch_size=4)
    dataset = BertDataset(args=args, tokenizer=tokenizer)

    print(dataset.get_class_indices())

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, collate_fn=dataset.data_collator)
    for batch in dataloader:
        print(batch)

    logits = torch.rand(4, 12, len(tokenizer))
    print(dataset.compute_probability(logits=logits))

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    args = argparse.Namespace(T="Gender", X="occupation", Y="gender", batch_size=4)
    dataset = GptDataset(args=args, tokenizer=tokenizer)

    print(dataset.get_class_indices())

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, collate_fn=dataset.data_collator)
    for batch in dataloader:
        print(batch)

    logits = torch.rand(4, 12, len(tokenizer))
    print(dataset.compute_probability(logits=logits))