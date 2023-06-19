import argparse
import functools
import os
from typing import Any

import torch
import transformers
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, set_peft_model_state_dict
from peft.tuners.lora import LoraLayer
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, Seq2SeqTrainer, LlamaTokenizer, \
    BitsAndBytesConfig, Seq2SeqTrainingArguments

from utils.model_utils import DataCollatorForCausalLM, smart_tokenizer_and_embedding_resize, find_all_linear_names, \
    DEFAULT_PAD_TOKEN, SavePeftModelCallback
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser()
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("base_model",  type=str, default="huggyllama/llama-7b", help="微调的基础模型")
add_arg("data_path",   type=str, default="dataset/merge.json",  help="数据集的路径")
add_arg("output_path", type=str, default="output/",             help="模型保存路径")
add_arg("cache_dir",   type=str, default="cache/",              help="模型缓存目录")
add_arg("save_steps",  type=int, default=200,                   help="多少步数保存模型一次")
add_arg("per_device_train_batch_size", type=int, default=1, help="训练的batch size")
add_arg("gradient_accumulation_steps", type=int, default=16, help="记录累积的次数")
add_arg("resume_from_checkpoint",      type=str, default=None, help="恢复检查点的路径")
add_arg("local_files_only",  type=bool, default=False,  help="是否只在本地加载模型，不尝试下载")
add_arg("max_source_length", type=int, default=512,     help="最大的模型输入长度")
add_arg("max_target_length", type=int, default=512,     help="最大的模型输出长度")
add_arg("num_train_epochs",  type=int, default=3,       help="总的迭代次数")
add_arg("num_workers",       type=int, default=4,       help="读取数据集的线程数量")
add_arg("logging_steps",     type=int, default=50,      help="多少步输出一次日志")
add_arg("warmup_ratio",      type=float, default=0.03,  help="预测步数比例")
add_arg("learning_rate",     type=float, default=2e-4,  help="学习率大小")
add_arg("lr_scheduler_type", type=str, default="constant",    help="学习率衰减方式")
add_arg("ignore_data_skip",  type=bool,  default=False,       help="忽略数据不进行处理")
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
assert args.bits in [4, 8]


def get_model(args):
    # 多卡时GPU处理
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

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
    model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                 cache_dir=args.cache_dir,
                                                 load_in_4bit=args.bits == 4,
                                                 load_in_8bit=args.bits == 8,
                                                 local_files_only=args.local_files_only,
                                                 device_map=device_map,
                                                 quantization_config=quantization_config,
                                                 torch_dtype=torch_dtype)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    # 量化
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    print('加载LoRA模块...')
    target_modules = find_all_linear_names(args.bits, model)
    print(target_modules)
    config = LoraConfig(r=args.lora_r,
                        lora_alpha=args.lora_alpha,
                        target_modules=target_modules,
                        lora_dropout=args.lora_dropout,
                        bias="none",
                        task_type="CAUSAL_LM")
    model = get_peft_model(model, config)
    # 恢复训练时加载Lora参数
    if args.resume_from_checkpoint:
        adapters_dict = torch.load(f'{args.resume_from_checkpoint}/pytorch_model.bin')
        set_peft_model_state_dict(model=model, peft_model_state_dict=adapters_dict)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    model.config.use_cache = False
    print('=' * 90)
    model.print_trainable_parameters()
    print('=' * 90)
    return model


def get_tokenizer(args, model):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model,
                                              cache_dir=args.cache_dir,
                                              local_files_only=args.local_files_only,
                                              padding_side="right",
                                              use_fast=False,
                                              tokenizer_type='llama')
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                                             tokenizer=tokenizer,
                                             model=model)
    if isinstance(tokenizer, LlamaTokenizer):
        tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
            "unk_token": tokenizer.convert_ids_to_tokens(
                model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
            ),
        })
    return tokenizer


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> (Any, Any):
    def preprocess(example):
        instruction, input_, output = example['instruction'], example['input'], example['output']
        prompt = f"Instruction: {instruction}\n"
        if input_ != '' and input_ is not None:
            prompt += f"{input_}\n"
        prompt += "Answer: "
        target = output
        return {"input": prompt, "output": target, "length": len(prompt) + len(target)}

    dataset = load_dataset("json", data_files={'train': args.data_path})
    train_dataset = dataset.map(preprocess, num_proc=10)['train']

    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer,
                                            max_source_length=args.max_source_length,
                                            max_target_length=args.max_target_length)

    return train_dataset, data_collator


def main():
    model = get_model(args)
    # Tokenizer
    tokenizer = get_tokenizer(args, model=model)
    set_seed(1234)

    train_dataset, data_collator = make_data_module(tokenizer=tokenizer, args=args)

    # 训练参数
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    output_dir = os.path.join(args.output_path, os.path.basename(args.base_model))
    training_args = Seq2SeqTrainingArguments(output_dir=output_dir,
                                             per_device_train_batch_size=args.per_device_train_batch_size,
                                             gradient_accumulation_steps=args.gradient_accumulation_steps,
                                             warmup_ratio=args.warmup_ratio,
                                             learning_rate=args.learning_rate,
                                             logging_steps=args.logging_steps,
                                             save_steps=args.save_steps,
                                             gradient_checkpointing=args.gradient_checkpointing,
                                             fp16=args.fp16,
                                             save_strategy="steps",
                                             optim='paged_adamw_32bit',
                                             lr_scheduler_type=args.lr_scheduler_type,
                                             num_train_epochs=args.num_train_epochs,
                                             dataloader_num_workers=args.num_workers,
                                             ddp_find_unused_parameters=False if ddp else None,
                                             save_total_limit=5,
                                             report_to=['tensorboard'],
                                             remove_unused_columns=False)
    trainer = Seq2SeqTrainer(model=model,
                             tokenizer=tokenizer,
                             args=training_args,
                             train_dataset=train_dataset,
                             data_collator=data_collator,
                             callbacks=[SavePeftModelCallback])

    # Training
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    if training_args.local_rank == 0 or training_args.local_rank == -1:
        model.save_pretrained(os.path.join(args.output_path, "checkpoint-final"))


if __name__ == '__main__':
    main()
