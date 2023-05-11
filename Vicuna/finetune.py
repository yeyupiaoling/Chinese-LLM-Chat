import argparse
import json
import os
import random

from transformers import Trainer, LlamaTokenizer, LlamaForCausalLM, TrainingArguments

from utils.data_utils import CustomDataset, DataCollator
from utils.utils import safe_save_model_for_hf_trainer, download_data, print_arguments

parser = argparse.ArgumentParser()
parser.add_argument("--base_model",  type=str, default="decapoda-research/llama-7b-hf", help="微调的基础模型")
parser.add_argument("--data_path",   type=str, default="dataset/merge.json", help="数据集的路径")
parser.add_argument("--output_path", type=str, default="output/", help="模型保存路径")
parser.add_argument("--cache_dir",   type=str, default="cache/",  help="模型缓存目录")
parser.add_argument("--eval_steps",  type=int, default=200,       help="多少步数评估一次")
parser.add_argument("--save_steps",  type=int, default=200,       help="多少步数保存模型一次")
parser.add_argument("--test_size",   type=int, default=2000,      help="分割测试集的大小")
parser.add_argument("--num_workers", type=int, default=4,         help="读取数据集的线程数量")
parser.add_argument("--per_device_train_batch_size", type=int, default=4,  help="训练的batch size")
parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="记录累积的次数")
parser.add_argument("--resume_from_checkpoint",      type=str, default=None, help="恢复检查点的路径")
parser.add_argument("--logging_steps",    type=int,   default=50,   help="多少步输出一次日志")
parser.add_argument("--model_max_length", type=int,   default=512,  help="模型最大输入长度")
parser.add_argument("--num_train_epochs", type=int,   default=3,    help="总的迭代次数")
parser.add_argument("--learning_rate",    type=float, default=5e-4, help="学习率大小")
parser.add_argument("--input_pattern",    type=str,   default="prompt", help="使用输入的模板类型")
args = parser.parse_args()
print_arguments(args)


# 下载数据集
if 'merge.json' in args.data_path and not os.path.exists(args.data_path):
    download_data(args.data_path)
# 读取JSON数据集文件
with open(args.data_path, "r", encoding="utf-8") as f:
    list_data_dict = json.load(f)
random.shuffle(list_data_dict)
data_len = len(list_data_dict)

# 获取模型
model = LlamaForCausalLM.from_pretrained(args.base_model, cache_dir=args.cache_dir)
# 获取token器
tokenizer = LlamaTokenizer.from_pretrained(args.base_model,
                                           cache_dir=args.cache_dir,
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
                               eval_steps=args.eval_steps,
                               save_steps=args.save_steps,
                               output_dir=args.output_path,
                               optim='adamw_torch',
                               save_total_limit=5,
                               load_best_model_at_end=True,
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
