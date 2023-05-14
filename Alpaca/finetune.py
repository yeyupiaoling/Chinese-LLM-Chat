import argparse
import functools
import os

from datasets import load_dataset
from peft import prepare_model_for_int8_training, LoraConfig, get_peft_model, TaskType, AdaLoraConfig
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers.trainer_pt_utils import LabelSmoother

from utils.utils import download_data, SavePeftModelCallback, print_arguments, add_arguments

parser = argparse.ArgumentParser()
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("base_model",  type=str, default="decapoda-research/llama-7b-hf",    help="微调的基础模型")
add_arg("data_path",   type=str, default="dataset/merge.json",  help="数据集的路径")
add_arg("output_path", type=str, default="output/",             help="模型保存路径")
add_arg("cache_dir",   type=str, default="cache/",              help="模型缓存目录")
add_arg("eval_steps",  type=int, default=200,                   help="多少步数评估一次")
add_arg("save_steps",  type=int, default=200,                   help="多少步数保存模型一次")
add_arg("test_size",   type=int, default=2000,                  help="分割测试集的大小")
add_arg("use_adalora", type=bool, default=True,                 help="是否使用AdaLora而不是Lora")
add_arg("per_device_train_batch_size", type=int, default=4, help="训练的batch size")
add_arg("per_device_eval_batch_size",  type=int, default=4, help="评估的batch size")
add_arg("gradient_accumulation_steps", type=int, default=32, help="记录累积的次数")
add_arg("resume_from_checkpoint",      type=str, default=None, help="恢复检查点的路径")
add_arg("local_files_only",  type=bool, default=False,  help="是否只在本地加载模型，不尝试下载")
add_arg("model_max_length",  type=int, default=256,     help="模型最大输入长度")
add_arg("num_train_epochs",  type=int, default=3,       help="总的迭代次数")
add_arg("num_workers",       type=int, default=4,       help="读取数据集的线程数量")
add_arg("logging_steps",     type=int, default=50,      help="多少步输出一次日志")
add_arg("learning_rate",     type=float, default=5e-4,  help="学习率大小")
add_arg("use_8bit",          type=bool,  default=True,  help="是否将模型量化为8位")
add_arg("ignore_data_skip",  type=bool,  default=False, help="忽略数据不进行处理")
args = parser.parse_args()
print_arguments(args)


# 用于填充的ID
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


# 输入模型指令模板
def text_template(instruction, input_=None):
    if input_ is None and input_ != '':
        return "Below is an instruction that describes a task, paired with an input that provides further context. " \
               "Write a response that appropriately completes the request.\n\n" \
               f"### Instruction:\n{instruction}\n{input_}\n\n### Response:"
    else:
        return "Below is an instruction that describes a task. " \
               "Write a response that appropriately completes the request.\n\n" \
               f"### Instruction:\n{instruction}\n\n### Response:"


# 多卡训练时的GPU处理
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
dataset = load_dataset("json", data_files=args.data_path)
dataset = dataset["train"].train_test_split(test_size=args.test_size, shuffle=True, seed=10000)
print(f"训练数据：{dataset['train'].num_rows}，测试数据：{dataset['test'].num_rows}")
# 获取token器
tokenizer = LlamaTokenizer.from_pretrained(args.base_model, add_eos_token=True, cache_dir=args.cache_dir,
                                           local_files_only=args.local_files_only)
tokenizer.pad_token_id = 0

# 设置Lora参数
if args.use_adalora:
    config = AdaLoraConfig(init_r=12, target_r=4, beta1=0.85, beta2=0.85, tinit=200, tfinal=1000, deltaT=10,
                           lora_alpha=32, lora_dropout=0.1, orth_reg_weight=0.5, task_type=TaskType.CAUSAL_LM,
                           target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"])
else:
    config = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none",
                        task_type=TaskType.CAUSAL_LM)

# 获取模型
if not args.use_8bit:
    model = LlamaForCausalLM.from_pretrained(args.base_model, device_map=device_map, cache_dir=args.cache_dir,
                                             local_files_only=args.local_files_only)
else:
    # 量化8位模型
    model = LlamaForCausalLM.from_pretrained(args.base_model, load_in_8bit=True, device_map=device_map,
                                             cache_dir=args.cache_dir, local_files_only=args.local_files_only)
    model = prepare_model_for_int8_training(model)
model = get_peft_model(model, config)


# 数据预处理
def preprocess(example):
    full_tokens, labels, attention_masks = [], [], []
    for instruction, input_, output in zip(example['instruction'], example['input'], example['output']):
        user_prompt = text_template(instruction, input_)
        # 获取输入prompt长度
        len_user_prompt_tokens = len(tokenizer(user_prompt,
                                               truncation=True,
                                               max_length=args.model_max_length + 1, ).input_ids) - 1  # no eos token
        # 拼接prompt和输出
        full_token = tokenizer(user_prompt + output,
                               truncation=True,
                               max_length=args.model_max_length + 1,
                               padding="max_length", ).input_ids[:-1]
        full_tokens.append(full_token)
        # 标签为prompt和输出一样，但会把前面的prompt替换为IGNORE_TOKEN_ID
        labels.append([IGNORE_TOKEN_ID] * len_user_prompt_tokens + full_token[len_user_prompt_tokens:])
        attention_masks.append([1] * (len(full_token)))
    return {"input_ids": full_tokens, "labels": labels, "attention_mask": attention_masks}


# 将预处理函数添加的数据集中
dataset.set_transform(preprocess)

# 训练参数
train_args = TrainingArguments(per_device_train_batch_size=args.per_device_train_batch_size,
                               per_device_eval_batch_size=args.per_device_eval_batch_size,
                               gradient_accumulation_steps=args.gradient_accumulation_steps,
                               warmup_steps=100,
                               num_train_epochs=args.num_train_epochs,
                               learning_rate=args.learning_rate,
                               fp16=True,
                               logging_steps=args.logging_steps,
                               evaluation_strategy="steps",
                               save_strategy="steps",
                               eval_steps=args.eval_steps,
                               save_steps=args.save_steps,
                               output_dir=args.output_path,
                               save_total_limit=5,
                               dataloader_num_workers=args.num_workers,
                               load_best_model_at_end=True,
                               optim='adamw_torch',
                               ddp_find_unused_parameters=False if ddp else None,
                               report_to=['tensorboard'],
                               remove_unused_columns=False,
                               ignore_data_skip=args.ignore_data_skip)
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
                  data_collator=DataCollatorForSeq2Seq(tokenizer))

# 开始训练
trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

# 保存模型
trainer.save_state()
if train_args.local_rank == 0 or train_args.local_rank == -1:
    model.save_pretrained(os.path.join(args.output_path, "checkpoint-final"))
