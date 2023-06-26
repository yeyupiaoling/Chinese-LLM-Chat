from typing import Dict

import bitsandbytes as bnb
import torch
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


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


def load_from_checkpoint(resume_from_checkpoint, model=None):
    pass


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    num_layers = 28
    layers_per_gpu = 30 / num_gpus
    device_map = {"transformer.word_embeddings": 0,
                  "transformer.final_layernorm": 0,
                  "transformer.prefix_encoder": 0,
                  "lm_head": 0}
    added_layers = 2
    target_gpu = 0

    for i in range(num_layers):
        if added_layers >= layers_per_gpu:
            target_gpu += 1
            added_layers = 0
        assert target_gpu < num_gpus
        device_map[f"transformer.layers.{i}"] = target_gpu
        added_layers += 1

    return device_map
