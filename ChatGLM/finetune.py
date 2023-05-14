import argparse
import functools
import os

import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_int8_training, AdaLoraConfig
from transformers import AutoTokenizer, AutoModel
from transformers import Trainer, TrainingArguments
from transformers.trainer_pt_utils import LabelSmoother

from utils.utils import download_data, get_masks_and_position_ids, SavePeftModelCallback, print_arguments, add_arguments

parser = argparse.ArgumentParser()
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("base_model",  type=str, default="THUDM/chatglm-6b",    help="微调的基础模型")
add_arg("data_path",   type=str, default="dataset/merge.json",  help="数据集的路径")
add_arg("output_path", type=str, default="output/",             help="模型保存路径")
add_arg("cache_dir",   type=str, default="cache/",              help="模型缓存目录")
add_arg("eval_steps",  type=int, default=200,                   help="多少步数评估一次")
add_arg("save_steps",  type=int, default=200,                   help="多少步数保存模型一次")
add_arg("test_size",   type=int, default=2000,                  help="分割测试集的大小")
add_arg("use_adalora", type=bool, default=True,                 help="是否使用AdaLora而不是Lora")
add_arg("per_device_train_batch_size", type=int, default=2, help="训练的batch size")
add_arg("per_device_eval_batch_size",  type=int, default=2, help="评估的batch size")
add_arg("gradient_accumulation_steps", type=int, default=8, help="记录累积的次数")
add_arg("resume_from_checkpoint",      type=str, default=None, help="恢复检查点的路径")
add_arg("local_files_only",  type=bool, default=False,  help="是否只在本地加载模型，不尝试下载")
add_arg("max_source_length", type=int, default=512,     help="最大的模型输入长度")
add_arg("max_target_length", type=int, default=512,     help="最大的模型输出长度")
add_arg("num_train_epochs",  type=int, default=3,       help="总的迭代次数")
add_arg("num_workers",       type=int, default=4,       help="读取数据集的线程数量")
add_arg("logging_steps",     type=int, default=50,      help="多少步输出一次日志")
add_arg("learning_rate",     type=float, default=5e-4,  help="学习率大小")
add_arg("use_8bit",          type=bool, default=True,   help="是否将模型量化为8位")
add_arg("ignore_data_skip",  type=bool,  default=False, help="忽略数据不进行处理")
args = parser.parse_args()
print_arguments(args)


# 用于填充的ID
IGNORE_TOKEN_ID = LabelSmoother.ignore_index
# 多卡训练处理
device_map = "auto"
world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    args.per_device_train_batch_size = args.per_device_train_batch_size * world_size

# 下载数据集
if 'merge.json' in args.data_path and not os.path.exists(args.data_path):
    download_data(args.data_path)
# 加载并分割数据集
dataset = load_dataset("json", data_files={'train': args.data_path})
dataset = dataset["train"].train_test_split(test_size=0.2, shuffle=True, seed=10000)
# 获取token器
tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, cache_dir=args.cache_dir,
                                          local_files_only=args.local_files_only)

# 设置Lora参数
if args.use_adalora:
    config = AdaLoraConfig(init_r=12, target_r=4, beta1=0.85, beta2=0.85, tinit=200, tfinal=1000, deltaT=10,
                           lora_alpha=32, lora_dropout=0.1, orth_reg_weight=0.5, task_type=TaskType.CAUSAL_LM,
                           target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"])
else:
    config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none",
                        task_type=TaskType.CAUSAL_LM)

if not args.use_8bit:
    model = AutoModel.from_pretrained(args.base_model, trust_remote_code=True, device_map=device_map,
                                      cache_dir=args.cache_dir, local_files_only=args.local_files_only).half()
else:
    # 将模型转量化8位
    model = AutoModel.from_pretrained(args.base_model, trust_remote_code=True, load_in_8bit=True, device_map=device_map,
                                      cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    model = prepare_model_for_int8_training(model)
model = get_peft_model(model, config)


# 数据补充填充
def data_collator(features: list) -> dict:
    # 获取一个batch的最长大小
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids, labels_list, attention_mask_list, position_ids_list = [], [], [], []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        ids = feature["input_ids"]
        seq_len = feature["seq_len"]
        mask_position = seq_len - 1
        pad_len = longest - ids_l
        # 根据prompt的长度创建label，前面prompt的长度用于填充IGNORE_TOKEN_ID，中间是模型输出，不够长度填充pad
        labels = [IGNORE_TOKEN_ID] * seq_len + ids[mask_position + 1:] + [tokenizer.pad_token_id] * pad_len
        labels_list.append(torch.LongTensor(labels))
        # 输入长度填充到对齐最长的数据
        ids = ids + [tokenizer.pad_token_id] * pad_len
        _ids = torch.LongTensor(ids)
        input_ids.append(_ids)
        # 获取掩码和
        attention_mask, position_ids = get_masks_and_position_ids(seq_len, longest, _ids.device)
        attention_mask_list.append(attention_mask)
        position_ids_list.append(position_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    attention_mask = torch.stack(attention_mask_list)
    position_ids = torch.stack(position_ids_list)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask, "position_ids": position_ids}


# 数据预处理
def preprocess(example):
    input_ids, seq_len = [], []
    for instruction, input_, output in zip(example['instruction'], example['input'], example['output']):
        prompt = f"Instruction: {instruction}\n"
        if input_ != '' and input_ is not None:
            prompt += f"{input_}\n"
        prompt += "Answer: "
        target = output
        # 将文本转换为token_id
        prompt_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(text=target, add_special_tokens=False)
        # 截取最大长度
        if len(prompt_ids) > args.max_source_length - 1:
            prompt_ids = prompt_ids[: args.max_source_length - 1]
        if len(target_ids) > args.max_target_length - 2:
            target_ids = target_ids[: args.max_target_length - 2]
        # 拼接两个token_id，并在中间和最后添加固定字符
        input_id = tokenizer.build_inputs_with_special_tokens(prompt_ids, target_ids)
        input_ids.append(input_id)
        # 模型输入长度
        seq_len.append(input_id.index(tokenizer.bos_token_id))
    return {"input_ids": input_ids, "seq_len": seq_len}


# 将预处理函数添加的数据集中
dataset.set_transform(preprocess)

# 训练参数
train_args = TrainingArguments(output_dir=args.output_path,
                               per_device_train_batch_size=args.per_device_train_batch_size,
                               per_device_eval_batch_size=args.per_device_eval_batch_size,
                               gradient_accumulation_steps=args.gradient_accumulation_steps,
                               num_train_epochs=args.num_train_epochs,
                               warmup_steps=100,
                               learning_rate=args.learning_rate,
                               logging_steps=args.logging_steps,
                               save_steps=args.save_steps,
                               eval_steps=args.eval_steps,
                               fp16=True,
                               save_strategy="steps",
                               evaluation_strategy="steps",
                               dataloader_num_workers=args.num_workers,
                               save_total_limit=5,
                               load_best_model_at_end=True,
                               ddp_find_unused_parameters=False if ddp else None,
                               report_to=['tensorboard'],
                               remove_unused_columns=False)
model.config.use_cache = False

if train_args.local_rank == 0 or train_args.local_rank == -1:
    print(f"训练数据：{dataset['train'].num_rows}，测试数据：{dataset['test'].num_rows}")
    print('=' * 90)
    model.print_trainable_parameters()
    print('=' * 90)

# 获取训练器
trainer = Trainer(model=model,
                  tokenizer=tokenizer,
                  args=train_args,
                  train_dataset=dataset["train"],
                  eval_dataset=dataset["test"],
                  callbacks=[SavePeftModelCallback],
                  data_collator=data_collator)

# 开始训练
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

trainer.save_state()
if train_args.local_rank == 0 or train_args.local_rank == -1:
    model.save_pretrained(os.path.join(args.output_path, "checkpoint-final"))
