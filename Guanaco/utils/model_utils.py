import copy
import os
import shutil
from dataclasses import dataclass
from typing import Dict, Sequence

import bitsandbytes as bnb
import peft
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_pt_utils import LabelSmoother
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

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


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: peft.PeftModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


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


# 保存模型回调，用于修改模型名称
class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if args.local_rank == 0 or args.local_rank == -1:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

            peft_model_dir = os.path.join(checkpoint_folder, "adapter_model")
            kwargs["model"].save_pretrained(peft_model_dir)

            peft_model_path = os.path.join(checkpoint_folder, "adapter_model/adapter_model.bin")
            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)
            shutil.copy(peft_model_path, pytorch_model_path)
        return control
