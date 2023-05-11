import os
import shutil

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
