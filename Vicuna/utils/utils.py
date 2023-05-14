import os
import shutil

import transformers
from huggingface_hub import snapshot_download


def print_arguments(args):
    print("----------------- 配置参数 ----------------------")
    for arg, value in vars(args).items():
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def strtobool(val):
    val = val.lower()
    if val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (val,))


def add_arguments(argname, type, default, help, argparser, **kwargs):
    type = strtobool if type == bool else type
    argparser.add_argument("--" + argname,
                           default=default,
                           type=type,
                           help=help + ' Default: %(default)s.',
                           **kwargs)


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
