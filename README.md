# Chinese-LLM-Chat

大语言模型微调的项目，为了能够在消费显卡上面使用项目，全部使用的都是QLora技术。

# 目录结构

 - `Guanaco`：使用QLora微调LLama模型。
 - `ChatGLM`：使用QLora微调ChatGLM-6B模型。
 - `LangChina`：使用langchain实现本地知识库聊天。

## 安装环境

要使用这些项目之前要先按照以下方式安装环境。

- 首先安装的是Pytorch的GPU版本，如果已经安装过了，请跳过。

```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
```

- 安装所需的依赖库。

```shell
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# 参考资料

1. https://github.com/Facico/Chinese-Vicuna
2. https://github.com/yuanzhoulvpi2017/zero_nlp
3. https://github.com/imClumsyPanda/langchain-ChatGLM
4. https://github.com/tatsu-lab/stanford_alpaca