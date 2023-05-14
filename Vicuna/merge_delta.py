import argparse
import functools
import os

import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser()
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("base_model", type=str, default="decapoda-research/llama-7b-hf", help="基本模型")
add_arg("delta_path", type=str, default="output/checkpoint-final/", help="微调保存的模型路径", choices=['lmsys/vicuna-7b-delta-v1.1', 'lmsys/vicuna-13b-delta-v1.1'])
parser.add_argument('--output_dir', type=str, default='models/', help="合并模型的保存目录")
add_arg("cache_dir", type=str, default="cache/")
args = parser.parse_args()
print_arguments(args)


print(f"加载基础模型：{args.base_model}")
base = LlamaForCausalLM.from_pretrained(args.base_model,
                                        torch_dtype=torch.float16,
                                        device_map={"": "cpu"},
                                        cache_dir=args.cache_dir)

print(f"Loading the delta from {args.delta_path}")
delta = LlamaForCausalLM.from_pretrained(args.delta_path,
                                         torch_dtype=torch.float16,
                                         device_map={"": "cpu"},
                                         cache_dir=args.cache_dir)
base_tokenizer = LlamaTokenizer.from_pretrained(args.delta_path, use_fast=False, cache_dir=args.cache_dir)

print("Applying the delta")
for name, param in tqdm(base.state_dict().items(), desc="Applying delta"):
    assert name in delta.state_dict()
    param.data += delta.state_dict()[name]

# 保存的文件夹路径
save_directory = os.path.join(args.output_dir, f'{os.path.basename(args.base_model)}-vicuna')
base.save_pretrained(save_directory)
base_tokenizer.save_pretrained(save_directory)
print(f'合并模型保持在：{save_directory}')
