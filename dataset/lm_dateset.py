from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length#输入给GPU的最大长度
        #使用HuggingFace datasets 惰性加载，避免一次性读入大文件
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    #我们拿到的是jsonl里的每一行
    #需要输出input_ids, labels
    def __getitem__(self, index):
        sample = self.samples[index]
        #使用tokenizer把文本转化成input_id
        tokens = self.tokenizer(str(sample['text']),
                             add_special_tokens=False, #不需要自动添加特殊token
                             max_length=self.max_length - 2, #留出位置给BOS和EOS
                             truncation=True).input_ids#如果长度超过max_length,自动剪切
        #加上BOS和EOS
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        #加上pad填充
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        #转化成tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        #自行编写labels，防止PAD参与loss计算
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        return input_ids, labels