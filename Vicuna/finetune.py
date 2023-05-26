import argparse
import functools
import json
import os
import random

from transformers import Trainer, LlamaTokenizer, LlamaForCausalLM, TrainingArguments

from utils.data_utils import CustomDataset, DataCollator
from utils.utils import safe_save_model_for_hf_trainer, download_data, print_arguments, add_arguments

parser = argparse.ArgumentParser()
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("base_model",  type=str, default="decapoda-research/llama-7b-hf", help="微调的基础模型")
add_arg("data_path",   type=str, default="dataset/merge.json", help="数据集的路径")
add_arg("output_path", type=str, default="output/", help="模型保存路径")
add_arg("cache_dir",   type=str, default="cache/",  help="模型缓存目录")
add_arg("eval_steps",  type=int, default=200,       help="多少步数评估一次")
add_arg("save_steps",  type=int, default=200,       help="多少步数保存模型一次")
add_arg("test_size",   type=int, default=2000,      help="分割测试集的大小")
add_arg("num_workers", type=int, default=4,         help="读取数据集的线程数量")
add_arg("per_device_train_batch_size", type=int, default=4,  help="训练的batch size")
add_arg("gradient_accumulation_steps", type=int, default=32, help="记录累积的次数")
add_arg("resume_from_checkpoint",      type=str, default=None, help="恢复检查点的路径")
add_arg("local_files_only", type=bool,  default=False, help="是否只在本地加载模型，不尝试下载")
add_arg("logging_steps",    type=int,   default=50,   help="多少步输出一次日志")
add_arg("model_max_length", type=int,   default=512,  help="模型最大输入长度")
add_arg("num_train_epochs", type=int,   default=3,    help="总的迭代次数")
add_arg("learning_rate",    type=float, default=5e-4, help="学习率大小")
add_arg("use_8bit",         type=bool,  default=False, help="是否将模型量化为8位")
add_arg("input_pattern",    type=str,   default="prompt", help="使用输入的模板类型")
args = parser.parse_args()
print_arguments(args)


# 多卡训练时的GPU处理
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

# 下载数据集
if 'merge.json' in args.data_path and not os.path.exists(args.data_path):
    download_data(args.data_path)
# 读取JSON数据集文件
with open(args.data_path, "r", encoding="utf-8") as f:
    list_data_dict = json.load(f)
random.shuffle(list_data_dict)
data_len = len(list_data_dict)

# 获取模型
model = LlamaForCausalLM.from_pretrained(args.base_model, device_map=device_map, cache_dir=args.cache_dir,
                                         local_files_only=args.local_files_only, load_in_8bit=args.use_8bit)
# 获取token器
tokenizer = LlamaTokenizer.from_pretrained(args.base_model,
                                           cache_dir=args.cache_dir,
                                           local_files_only=args.local_files_only,
                                           model_max_length=args.model_max_length,
                                           padding_side="right",
                                           use_fast=False)
tokenizer.pad_token = tokenizer.unk_token

# 获取自定义数据集
train_dataset = CustomDataset(tokenizer=tokenizer,
                              list_data=list_data_dict[:data_len - args.test_size - 1],
                              input_pattern=args.input_pattern)
eval_dataset = CustomDataset(tokenizer=tokenizer,
                             list_data=list_data_dict[data_len - args.test_size:],
                             input_pattern=args.input_pattern)
data_collator = DataCollator(tokenizer=tokenizer)

# 定义训练参数
train_args = TrainingArguments(per_device_train_batch_size=args.per_device_train_batch_size,
                               gradient_accumulation_steps=args.gradient_accumulation_steps,
                               warmup_steps=100,
                               num_train_epochs=args.num_train_epochs,
                               learning_rate=args.learning_rate,
                               logging_steps=args.logging_steps,
                               evaluation_strategy="steps",
                               save_strategy="steps",
                               fp16=True,
                               eval_steps=args.eval_steps,
                               save_steps=args.save_steps,
                               output_dir=args.output_path,
                               optim='adamw_torch',
                               save_total_limit=5,
                               load_best_model_at_end=True,
                               ddp_find_unused_parameters=False if ddp else None,
                               dataloader_num_workers=args.num_workers,
                               report_to=['tensorboard'])

# 定义训练器
trainer = Trainer(model=model,
                  tokenizer=tokenizer,
                  args=train_args,
                  train_dataset=train_dataset,
                  eval_dataset=eval_dataset,
                  data_collator=data_collator)

# 开始训练
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# 保存模型
trainer.save_state()
if train_args.local_rank == 0 or train_args.local_rank == -1:
    print('保存最终模型')
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=os.path.join(train_args.output_dir, "checkpoint-final"))
