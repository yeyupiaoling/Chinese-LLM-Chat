import os
import shutil

import transformers
from huggingface_hub import snapshot_download


def download_data(save_path):
    p = snapshot_download(repo_id='Chinese-Vicuna/guanaco_belle_merge_v1.0', repo_type='dataset')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    shutil.copyfile(os.path.join(p, 'merge.json'), save_path)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
