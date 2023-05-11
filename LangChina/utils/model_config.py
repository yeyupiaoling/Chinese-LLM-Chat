# 词向量模型名称
embedding_model_dict = {
    "ernie-xbase": "nghuyong/ernie-3.0-xbase-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "ernie-medium": "nghuyong/ernie-3.0-medium-zh",
    "ernie-mini": "nghuyong/ernie-3.0-mini-zh",
    "ernie-micro": "nghuyong/ernie-3.0-micro-zh",
    "ernie-nano": "nghuyong/ernie-3.0-nano-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
}


# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，
不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

# 文本分句长度
SENTENCE_SIZE = 200
