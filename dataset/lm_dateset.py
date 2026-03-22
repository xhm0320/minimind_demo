from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    ## 定义候选系统提示列表
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    #对话数据非空 + 第一条不是system角色（避免重复加system提示）
    if conversations and conversations[0].get('role') != 'system':
        #按20%的概率随机添加
        if random.random() < add_system_ratio:
            ## 随机选一个系统提示，插到对话最前面
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    # 不满足条件则返回原对话
    return conversations

#对话后处理，清理模块渲染后多余的空<think>块
def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

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
    
class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length#输入序列最大长度
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')# 加载JSONL格式的对话数据集
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids#预计算assistant开头的token id
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids  # 预计算结束符的token id

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = conversations.copy()# 复制对话数据（避免修改原数据）
        # 处理工具调用（可选）：如果第一条是system且包含functions，提取工具定义
        tools = conversations[0]["functions"] if (
            conversations 
            and conversations[0]["role"] == "system" 
            and conversations[0].get("functions")
            ) else None
        #  应用对话模板
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,#  先返回文本，不直接转token id
            add_generation_prompt=False,# 不添加“生成提示”
            tools=tools# 工具调用配置
        )

    def generate_labels(self, input_ids):
         # 初始化labels：全为-100（PyTorch中-100表示该位置不计算损失）
        labels = [-100] * len(input_ids)
        i = 0
        #滑动窗口，寻找回答部分，并回复本来的label
        while i < len(input_ids):
             # 找到assistant开头的位置（匹配self.bos_id）
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        # 1. 加载单条样本
        sample = self.samples[index]
        # 2. 预处理：随机添加系统提示
        conversations = pre_processing_chat(sample['conversations'])
        # 3. 格式化对话为模型输入文本
        prompt = self.create_chat_prompt(conversations)
        # 4. 后处理：清理特殊占位符
        prompt = post_processing_chat(prompt)
        # 5. 文本转token id，截断到max_length
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 6. 补齐到max_length（不足部分用pad_token_id填充）
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
         # 7. 生成训练标签（仅标注assistant回答）
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        chosen = sample['chosen']  # 是一个 list，里面包含若干 {role, content}
        rejected = sample['rejected']  # 同上

        #将 chosen和rejected 对话历史转换为模型可读的格式化字符串。
        #tokenize=False：只转换为字符串，不转换为 token ID。
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False#
        )
        chosen_prompt = post_processing_chat(chosen_prompt)
        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        
        #将 chosen_prompt和rejected_encoding 转换为 token ID，并统一长度。
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        #保存 chosen 对应的 token ID 序列。
        chosen_input_ids = chosen_encoding['input_ids']
        #生成损失掩码，标记哪些位置是 “assistant 回复”（只计算这部分损失）。
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
        #保存 rejected 对应的 token ID 序列。
        rejected_input_ids = rejected_encoding['input_ids']
        #生成 rejected 对应的损失掩码。
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        #去掉序列的最后一个 token并将列表转换为 PyTorch 长整型张量
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        #构造损失掩码，对应目标输出的位置。
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        #构造 rejected 对应的模型输入。
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        #构造 rejected 对应的损失掩码。
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask
    
class RLAIFDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        messages = []
        answer = ''
        for i, turn in enumerate(conversations):
            # 根据轮次索引奇偶性分配角色：偶数轮是 user，奇数轮是 assistant
            role = 'user' if i % 2 == 0 else 'assistant'
            # 将当前轮次整理为 Hugging Face 标准对话格式 {"role": "...", "content": "..."}，并添加到 messages
            messages.append({"role": role, "content": turn['content']})
             # 不断更新 answer，最终保存最后一轮对话的内容（即 assistant 的目标回复）
            answer = turn['content']
         # 使用分词器的 apply_chat_template 方法，将除最后一轮外的对话历史转换为模型可读的字符串
        prompt = self.tokenizer.apply_chat_template(
            messages[:-1],
            tokenize=False,
            add_generation_prompt=True  # 这里需要True
        )
        prompt = post_processing_chat(prompt)
        return prompt, answer

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt, answer = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': answer
        }

if __name__ == "__main__":
    pass


