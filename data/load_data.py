import torch
from datasets import load_dataset
from typing import Tuple

class DataLoader:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_data, self.eval_data = self._load_data()

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ds = load_dataset("SSahas/llm_pretrain_dataset")
        all_data = torch.tensor(ds['train']['input_ids'], dtype=torch.long, device=self.device)
        
        # Calculate split index (90% for training)
        split_idx = int(0.9 * len(all_data))
        
        # Split the data
        train_data = all_data[:split_idx]
        eval_data = all_data[split_idx:]
        
        return train_data, eval_data

    def get_batch(self, split: str = 'train') -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == 'train' else self.eval_data
        ix = torch.randint(len(data) - self.config['model']['block_size'], (self.config['training']['batch_size'],))
        x = torch.stack([data[i:i+self.config['model']['block_size']] for i in ix])
        y = torch.stack([data[i+1:i+self.config['model']['block_size']+1] for i in ix])
        return x.to(self.device), y.to(self.device)
