import argparse
import functools
import os

import torch
from peft import PeftModel, PeftConfig
from transformers import LlamaForCausalLM, LlamaTokenizer

from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser()
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("lora_model", type=str, default="output/checkpoint-final/", help="微调保存的模型路径",
                    choices=['Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco'])
add_arg("output_dir", type=str, default="models/", help="合并模型的保存目录")
add_arg("cache_dir",  type=str, default="cache/",  help="模型缓存目录")
args = parser.parse_args()
print_arguments(args)


# 获取Lora配置参数
peft_config = PeftConfig.from_pretrained(args.lora_model)
# 获取ChatGLM的基本模型
base_model = LlamaForCausalLM.from_pretrained(peft_config.base_model_name_or_path,
                                              torch_dtype=torch.float16,
                                              device_map={"": "cpu"},
                                              cache_dir=args.cache_dir)
# 获取token器
base_tokenizer = LlamaTokenizer.from_pretrained(peft_config.base_model_name_or_path,
                                                add_eos_token=True,
                                                cache_dir=args.cache_dir)
# 与Lora模型合并
model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=torch.float16)

# 保存的文件夹路径
save_directory = os.path.join(args.output_dir, f'{os.path.basename(peft_config.base_model_name_or_path)}-finetune')
os.makedirs(save_directory, exist_ok=True)

# 合并参数
model = model.merge_and_unload()
model.train(False)

# 保存模型到指定目录中
model.save_pretrained(save_directory)
base_tokenizer.save_pretrained(save_directory)
print(f'合并模型保持在：{save_directory}')
