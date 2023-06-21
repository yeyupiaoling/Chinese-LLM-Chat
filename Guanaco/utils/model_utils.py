import copy
from dataclasses import dataclass
from typing import Dict, Sequence

import bitsandbytes as bnb
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from transformers import LogitsProcessor, LogitsProcessorList
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
DEFAULT_PAD_TOKEN = "[PAD]"


def find_all_linear_names(bits, model):
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    max_source_length: int
    max_target_length: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources = self.tokenizer(sources,
                                           max_length=self.max_source_length,
                                           truncation=True,
                                           add_special_tokens=False)
        tokenized_targets = self.tokenizer(targets,
                                           max_length=self.max_target_length,
                                           truncation=True,
                                           add_special_tokens=False)
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(tokenized_sources['input_ids'], tokenized_targets['input_ids']):
            input_id = torch.tensor(tokenized_source + tokenized_target)
            label = torch.tensor([IGNORE_TOKEN_ID for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
            input_ids.append(input_id)
            labels.append(label)
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_TOKEN_ID)
        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        return data_dict


class InvalidScoreLogitsProcessor(LogitsProcessor):

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 0] = 1.0
        return scores


def get_logits_processor() -> LogitsProcessorList:
    logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    return logits_processor


def load_from_checkpoint(resume_from_checkpoint, model=None):
    pass

