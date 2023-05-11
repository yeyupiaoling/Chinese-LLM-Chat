import os
import shutil

import torch
from huggingface_hub import snapshot_download
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


def print_arguments(args):
    print("----------------- 配置参数 ----------------------")
    for arg, value in vars(args).items():
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def download_data(save_path):
    p = snapshot_download(repo_id='Chinese-Vicuna/guanaco_belle_merge_v1.0', repo_type='dataset')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    shutil.copyfile(os.path.join(p, 'merge.json'), save_path)


def get_masks_and_position_ids(seq_len, context_length, device, gmask=False, position_encoding_2d=True):
    mask_position = (seq_len - 2)  # is equal to `seq.index(mask_token)` or `seq.index(130001)`
    attention_mask = torch.ones((1, context_length, context_length), device=device)
    attention_mask.tril_()
    attention_mask[..., : mask_position - 1] = 1
    attention_mask = (attention_mask < 0.5).bool()

    if position_encoding_2d:
        seq_length = seq_len - 1  # is equal to `seq_length = seq.index(130004)`
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[seq_length:] = mask_position
        block_position_ids = torch.cat((torch.zeros(seq_length, dtype=torch.long, device=device),
                                        torch.arange(context_length - seq_length, dtype=torch.long,
                                                     device=device) + 1,))
        position_ids = torch.stack((position_ids, block_position_ids), dim=0)
    else:
        position_ids = torch.arange(context_length, dtype=torch.long, device=device)
        if not gmask:
            position_ids[context_length - 1:] = mask_position
    return attention_mask, position_ids


# 保存模型回调，用于修改模型名称
class SavePeftModelCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_dir = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_dir)

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model/adapter_model.bin")
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        shutil.copy(peft_model_path, pytorch_model_path)
        return control
