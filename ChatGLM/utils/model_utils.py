import os
import shutil

import bitsandbytes as bnb
import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_pt_utils import LabelSmoother
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

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
