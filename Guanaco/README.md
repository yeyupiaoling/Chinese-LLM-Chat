# 项目介绍

当前文件夹主要使用QLora对LLama模型的微调，然后合并模型。在使用方面包含了两种使用方式，分别是直接在终端上面运行，该方式是非流式的，也就是直接显示结果，没有打字过程。另外一种是使用Gradio技术在网页中使用，这个过程是流式的，可以实时显示打字效果。


# 准备数据集

在开启微调模型之前，需要准备好数据，数据的格式如下面所示，是一个JSON的数据列表，在[HuggingFace](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0)
也有一个开源的数据集。如果想自定义数据集的话，请按照以下格式生成。

```json
[
  {
    "instruction": "用一句话描述地球为什么是独一无二的。\n ",
    "input": "",
    "output": "地球上有适宜生命存在的条件和多样化的生命形式。"
  }
]
```

# 微调模型

`finetune.py`就是微调LLama模型的，训练最重要的两个参数分别是：
 - `--base_model`指定微调的基础模型，这个参数值需要在[HuggingFace](https://huggingface.co/decapoda-research)存在的，默认的是`decapoda-research/llama-7b-hf`，这个不需要提前下载，启动训练时可以自动下载，当然也可以提前下载，那么`--base_model`指定就是路径，同时`--local_files_only`设置为True。
 - `--data_path`指定的是数据集路径，如果自己的数据，可以在这里[HuggingFace](https://huggingface.co/datasets/Chinese-Vicuna/guanaco_belle_merge_v1.0)下载下载数据来使用。
 - `--per_device_train_batch_size`指定的训练的batch大小，如果修改了这个参数，同时也要修改`--gradient_accumulation_steps`，使得它们的乘积一样。
 - `--output_path`指定训练时保存的检查点路径。
 - `--use_8bit`指定是否使用量化模型训练，如果想显存足够的话，最好将设置为False，这样训练速度快很多。
 - 其他更多的参数请查看这个程序。

### 单卡训练

单卡训练命令如下，Windows系统可以不添加`CUDA_VISIBLE_DEVICES`参数。

```shell
CUDA_VISIBLE_DEVICES=0 python finetune.py --base_model=decapoda-research/llama-7b-hf --output_dir=output/
```

### 多卡训练

多卡训练有两种方法，分别是torchrun和accelerate，开发者可以根据自己的习惯使用对应的方式。

1. 使用torchrun启动多卡训练，命令如下，通过`--nproc_per_node`指定使用的显卡数量。

```shell
torchrun --nproc_per_node=2 finetune.py --base_model=decapoda-research/llama-7b-hf --output_dir=output/
```

2. 使用accelerate启动多卡训练，如果是第一次使用accelerate，要配置训练参数，方式如下。

首先配置训练参数，过程是让开发者回答几个问题，基本都是默认就可以，但有几个参数需要看实际情况设置。

```shell
accelerate config
```

大概过程就是这样：

```
----------------------------------In which compute environment are you running?
This machine
----------------------------------Which type of machine are you using? 
multi-GPU
How many different machines will you use (use more than 1 for multi-node training)? [1]:
Do you wish to optimize your script with torch dynamo?[yes/NO]:
Do you want to use DeepSpeed? [yes/NO]:
Do you want to use FullyShardedDataParallel? [yes/NO]:
Do you want to use Megatron-LM ? [yes/NO]: 
How many GPU(s) should be used for distributed training? [1]:2
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list? [all]:
----------------------------------Do you wish to use FP16 or BF16 (mixed precision)?
fp16 
```

配置完成之后，可以使用以下命令查看配置。

```shell
accelerate env
```

开始训练命令如下。

```shell
accelerate launch finetune.py --base_model=decapoda-research/llama-7b-hf --output_dir=output/
```

输出日志如下：

```shell
{'loss': 0.9098, 'learning_rate': 0.000999046843662503, 'epoch': 0.01}                                                     
{'loss': 0.5898, 'learning_rate': 0.0009970611012927184, 'epoch': 0.01}                                                    
{'loss': 0.5583, 'learning_rate': 0.0009950753589229333, 'epoch': 0.02}                                                  
{'loss': 0.5469, 'learning_rate': 0.0009930896165531485, 'epoch': 0.02}                                          
{'loss': 0.5959, 'learning_rate': 0.0009911038741833634, 'epoch': 0.03}
```

# 合并模型

微调完成之后会有两个模型，第一个是基础模型，第二个是Lora模型，需要把这两个模型合并之后才能之后的操作。这个程序只需要传递两个参数，`--lora_model`指定的是训练结束后保存的Lora模型路径，注意如何不是最后的`checkpoint-final`后面还有`adapter_model`文件夹，也支持直接使用第三方提供的模型，目前找到的是`Facico/Chinese-Vicuna-lora-7b-3epoch-belle-and-guanaco`，有其他的也可以使用。第二个`--output_dir`是合并后模型的保存目录。

```shell
python merge_lora.py --lora_model=output/checkpoint-final --output_dir=models/
```

# 预测

本项目提供两种预测方式，在推理的时候要注意几个参数：
 - `--model_path`指定的是微调后合并的模型。
 - `--num_gpus`可以指定使用多少个显卡进行推理，如果一张显存不够的话，使用多张显卡可以分配显存。
 - `--input_pattern`指定模型输入的Prompt，微调的目前只是用了`prompt`。
 - `--load_8bit`指定是否加载使用量化模型。

`cli_demo.py`是在终端直接使用，为了简便，这里直接使用的是最终输出，不是流式输出，没有打字效果。

```shell
python cli_demo.py --model_path=./models/llama-7b-hf-finetune --num_gpus=2 --input_pattern=prompt --load_8bit=False
```

`gradio_ui.py`使用Gradio搭建了一个网页，部署到服务器，在网页中使用聊天，为流式输出，有打字效果。

```shell
python gradio_ui.py --model_path=./models/llama-7b-hf-finetune --num_gpus=2 --input_pattern=prompt --load_8bit=False
```

# 参考资料

1. https://github.com/yuanzhoulvpi2017/zero_nlp
2. https://github.com/tatsu-lab/stanford_alpaca
