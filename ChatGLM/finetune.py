import argparse
import functools
import os

import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig, Seq2SeqTrainingArguments, Seq2SeqTrainer

from utils.model_utils import find_all_linear_names, IGNORE_TOKEN_ID, load_from_checkpoint
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("base_model",  type=str, default="THUDM/chatglm2-6b",   help="微调的基础模型")
add_arg("data_path",   type=str, default="dataset/merge.json",  help="数据集的路径")
add_arg("output_path", type=str, default="output/",             help="模型保存路径")
add_arg("cache_dir",   type=str, default="cache/",              help="模型缓存目录")
add_arg("save_steps",  type=int, default=200,                   help="多少步数保存模型一次")
add_arg("per_device_train_batch_size", type=int, default=1,     help="训练的batch size")
add_arg("gradient_accumulation_steps", type=int, default=16,    help="记录累积的次数")
add_arg("resume_from_checkpoint",      type=str, default=None,  help="恢复检查点的路径")
add_arg("local_files_only",  type=bool, default=False,  help="是否只在本地加载模型，不尝试下载")
add_arg("max_source_length", type=int, default=512,     help="最大的模型输入长度")
add_arg("max_target_length", type=int, default=512,     help="最大的模型输出长度")
add_arg("num_train_epochs",  type=int, default=3,       help="总的迭代次数")
add_arg("num_workers",       type=int, default=4,       help="读取数据集的线程数量")
add_arg("logging_steps",     type=int, default=50,      help="多少步输出一次日志")
add_arg("learning_rate",     type=float, default=5e-4,  help="学习率大小")
add_arg("ignore_data_skip",  type=bool,  default=False, help="忽略数据不进行处理")
add_arg("ignore_pad_token_for_loss", type=bool, default=True, help="在计算损失的时候是否忽略pad_token")
# 显存优化
add_arg("fp16", type=bool, default=True,         help="是否使用fp16")
add_arg("bf16", type=bool, default=False,        help="是否使用bf16")
add_arg("max_memory_MB", type=int, default=8000, help="执行最大使用的显存")
add_arg("gradient_checkpointing", type=bool, default=True, help="使用梯度检查点机制节省内存，但会以较慢的反向传递为代价")
# Lora参数
add_arg("bits", type=int, default=4,             help="量化的位数，只能是4或者8")
add_arg("lora_r", type=int, default=64,          help="Lora参数r")
add_arg("lora_alpha", type=int, default=16,      help="Lora参数lora_alpha")
add_arg("lora_dropout", type=float, default=0.1, help="Lora参数lora_dropout")
add_arg("quant_type", type=str, default="nf4",   help="BitsAndBytesConfig量化参数quant_type")
add_arg("double_quant", type=bool, default=True, help="BitsAndBytesConfig量化参数double_quant")
args = parser.parse_args()
print_arguments(args)


# 多卡时GPU处理
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
# 获取数据集
dataset = load_dataset("json", data_files={'train': args.data_path})
# 获取token器
tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, cache_dir=args.cache_dir,
                                          local_files_only=args.local_files_only)

# 获取模型
print(f'加载模型：{args.base_model}...')
compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
torch_dtype = (torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
quantization_config = BitsAndBytesConfig(load_in_4bit=args.bits == 4,
                                         load_in_8bit=args.bits == 8,
                                         llm_int8_threshold=6.0,
                                         llm_int8_has_fp16_weight=False,
                                         bnb_4bit_compute_dtype=compute_dtype,
                                         bnb_4bit_use_double_quant=args.double_quant,
                                         bnb_4bit_quant_type=args.quant_type)
model = AutoModel.from_pretrained(args.base_model,
                                  trust_remote_code=True,
                                  cache_dir=args.cache_dir,
                                  local_files_only=args.local_files_only,
                                  load_in_4bit=args.bits == 4,
                                  load_in_8bit=args.bits == 8,
                                  device_map=device_map,
                                  quantization_config=quantization_config if args.bits in [4, 8] else None,
                                  torch_dtype=torch_dtype)

# 量化
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

print('加载LoRA模块...')
if args.resume_from_checkpoint:
    # 恢复训练时加载Lora参数
    print("Loading adapters from checkpoint.")
    model = PeftModel.from_pretrained(model, args.resume_from_checkpoint, is_trainable=True)
else:
    print(f'adding LoRA modules...')
    target_modules = find_all_linear_names(args.bits, model)
    print(target_modules)
    config = LoraConfig(r=args.lora_r,
                        lora_alpha=args.lora_alpha,
                        target_modules=target_modules,
                        lora_dropout=args.lora_dropout,
                        bias="none",
                        task_type="CAUSAL_LM")
    model = get_peft_model(model, config)


def data_collator(features: list) -> dict:
    len_ids = [feature["seq_len"] for feature in features]
    max_seq_length = max(len_ids) + 1
    input_ids_list, labels_list = [], []
    for feature in features:
        input_ids = feature["input_ids"]
        pad_len = max_seq_length - feature["seq_len"]

        context_length = input_ids.index(tokenizer.bos_token_id)
        mask_position = context_length - 1
        labels = [IGNORE_TOKEN_ID] * context_length + input_ids[mask_position + 1:]

        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        labels = labels + [tokenizer.pad_token_id] * pad_len
        if args.ignore_pad_token_for_loss:
            labels = [(l if l != tokenizer.pad_token_id else IGNORE_TOKEN_ID) for l in labels]

        input_ids_list.append(torch.LongTensor(input_ids))
        labels_list.append(torch.LongTensor(labels))
    input_ids = torch.stack(input_ids_list)
    labels = torch.stack(labels_list)
    return {"input_ids": input_ids, "labels": labels}


# 数据预处理
def preprocess(example):
    history = []
    instruction, input_, target = example["instruction"], example['input'], example["output"]
    if input_ is not None:
        instruction = instruction + input_
    if 'history' in example.keys() and example["history"] is not None:
        history = example["history"]
    prompt = ""
    for turn_idx, (old_query, response) in enumerate(history):
        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
    prompt += "[Round {}]\n问：{}\n答：".format(len(history), instruction)
    # 将文本转换为token_id
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    # 截取最大长度
    if len(prompt_ids) > args.max_source_length - 1:
        prompt_ids = prompt_ids[max(len(prompt_ids) - args.max_source_length, 0):]
    if len(target_ids) > args.max_target_length - 2:
        target_ids = target_ids[:args.max_target_length]
    input_ids = tokenizer.build_inputs_with_special_tokens(prompt_ids, target_ids)
    return {"input_ids": input_ids, "seq_len": len(input_ids)}


dataset = dataset.map(preprocess, num_proc=10)

# 训练参数
output_dir = os.path.join(args.output_path, os.path.basename(args.base_model))
train_args = Seq2SeqTrainingArguments(output_dir=output_dir,
                                      per_device_train_batch_size=args.per_device_train_batch_size,
                                      gradient_accumulation_steps=args.gradient_accumulation_steps,
                                      warmup_steps=100,
                                      learning_rate=args.learning_rate,
                                      logging_steps=args.logging_steps,
                                      save_steps=args.save_steps,
                                      gradient_checkpointing=args.gradient_checkpointing,
                                      fp16=args.fp16,
                                      save_strategy="steps",
                                      optim='paged_adamw_32bit',
                                      num_train_epochs=args.num_train_epochs,
                                      dataloader_num_workers=args.num_workers,
                                      save_total_limit=5,
                                      ddp_find_unused_parameters=False if ddp else None,
                                      report_to=['tensorboard'],
                                      remove_unused_columns=False)
model.config.use_cache = False

if train_args.local_rank == 0 or train_args.local_rank == -1:
    print(f"训练数据：{dataset['train'].num_rows}")
    print('=' * 90)
    model.print_trainable_parameters()
    print('=' * 90)

# 定义训练器
trainer = Seq2SeqTrainer(model=model,
                         tokenizer=tokenizer,
                         args=train_args,
                         train_dataset=dataset["train"],
                         data_collator=data_collator)
trainer._load_from_checkpoint = load_from_checkpoint

# 开始训练
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

trainer.save_state()
if train_args.local_rank == 0 or train_args.local_rank == -1:
    model.save_pretrained(os.path.join(args.output_path, "checkpoint-final"))
