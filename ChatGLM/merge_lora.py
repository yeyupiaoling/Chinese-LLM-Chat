import argparse
import functools
import os

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModel, AutoTokenizer

from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser()
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("lora_model", type=str, default="output/checkpoint-final/", help="微调保存的模型路径")
add_arg("output_dir", type=str, default="models/",  help="合并模型的保存目录")
add_arg("cache_dir",  type=str, default="cache/",   help="模型缓存目录")
args = parser.parse_args()
print_arguments(args)


# 检查模型文件是否存在
assert os.path.exists(args.lora_model), f"模型文件{args.lora_model}不存在"
# 获取Lora配置参数
peft_config = PeftConfig.from_pretrained(args.lora_model)
# 获取ChatGLM的基本模型
base_model = AutoModel.from_pretrained(peft_config.base_model_name_or_path,
                                       torch_dtype=torch.float16,
                                       device_map={"": "cpu"},
                                       trust_remote_code=True,
                                       cache_dir=args.cache_dir)
# 获取token器
base_tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path,
                                               trust_remote_code=True,
                                               cache_dir=args.cache_dir)
# 与Lora模型合并
model = PeftModel.from_pretrained(base_model, args.lora_model)

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
