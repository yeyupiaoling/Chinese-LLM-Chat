# 项目介绍

本项目是主要是使用LangChain技术，然后结合大语言模型来实现本地知识库聊天。在聊天的时候会加载本地的知识库，然后从本地知识库里面筛选出所要回答的问题并回复。

# 使用

本项目提供两种预测方式，在推理的时候要注意几个参数：
 - `--model_path`指定的是微调后合并的模型。
 - `--num_gpus`可以指定使用多少个显卡进行推理，如果一张显存不够的话，使用多张显卡可以分配显存。
 - `--input_pattern`指定模型输入的Prompt，微调的目前只是用了`prompt`，但如果是用原生需要指定为`chat`。
 - `--load_8bit`指定是否加载使用量化模型。

`cli_demo.py`是在终端直接使用，命令如下。运行的时候第一步会提升`请输入本地知识文件路径：`，这时候需要输入本地知识文件的路径，可以是某个文件也可以是文件夹，如果是已经有了可以直接回车跳过。后面就是正常的聊天了。

```shell
python cli_demo.py --model_path=THUDM/chatglm-6b --num_gpus=2 --input_pattern=chat --load_8bit=False
```

# 参考资料

1. https://github.com/imClumsyPanda/langchain-ChatGLM
