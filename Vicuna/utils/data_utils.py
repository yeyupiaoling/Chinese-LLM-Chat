import copy
from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import transformers
from fastchat.conversation import get_conv_template
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother

# 用于数据填充的ID
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# 输入模型指令模板
def text_template(instruction, input_=None):
    user_input = f"{instruction}\n\n### Input:\n{input_}" if input_ != '' and input_ is not None \
        else f"{instruction}"
    return user_input


# 自定义数据读取
class CustomDataset(Dataset):
    def __init__(self, list_data: List[dict], tokenizer: transformers.LlamaTokenizer, input_pattern: str):
        super(CustomDataset, self).__init__()
        self.tokenizer = tokenizer
        if input_pattern == "prompt":
            self.conv = get_conv_template('dolly_v2')
        elif input_pattern == "chat":
            self.conv = get_conv_template('vicuna_v1.1')
        else:
            raise ValueError("input_pattern must be either 'prompt' or 'chat'")
        self.cached_data_dict = {}

        self.sources = [text_template(example['instruction'], example.get('input', None)) for example in list_data]
        # 获取输出
        self.targets = [example['output'] for example in list_data]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]
        source = self.sources[i]
        target = self.targets[i]
        conv = self.conv.copy()
        conv.append_message(conv.roles[0], source)
        conv.append_message(conv.roles[1], target)
        text = conv.get_prompt()
        sep = conv.sep + conv.roles[1] + ": "
        # 获取模型输入和输出的全部tokens
        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   padding="max_length",
                                   max_length=self.tokenizer.model_max_length,
                                   truncation=True).input_ids[0]
        parts = text.split(sep)
        parts[0] += sep
        # 只获取模型的输入tokens
        source_ids = self.tokenizer(parts[0],
                                    return_tensors="pt",
                                    padding="max_length",
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True).input_ids[0]
        # 复制一份作为label
        label = copy.deepcopy(input_ids)
        # 判断不等于pad_token_id的总和，即为输入的长度
        input_ids_lens = source_ids.ne(self.tokenizer.pad_token_id).sum().item() - 1
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        # 将label的前面prompt部分填充为IGNORE_TOKEN_ID
        label[:input_ids_lens] = IGNORE_TOKEN_ID
        ret = dict(input_ids=input_ids, labels=label, attention_mask=attention_mask)
        self.cached_data_dict[i] = ret
        return ret


@dataclass
class DataCollator(object):
    tokenizer: transformers.LlamaTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
