import argparse
import functools
import os

from utils.local_doc_qa import LocalDocQA
from utils.predictor import ChatGLMPredictor
from utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser()
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("model_path", type=str, default="THUDM/chatglm-6b",  help="合并后的模型路径或者原模型名称")
add_arg("cache_dir",  type=str,  default="cache/",    help="模型缓存目录")
add_arg("bits",       type=int,  default=4,           help="使用量化多少位")
add_arg("fp16",       type=bool, default=False,       help="是否半精度推理")
add_arg("local_files_only", type=bool, default=False, help="是否只在本地加载模型，不尝试下载")
args = parser.parse_args()
print_arguments(args)

# 获取模型推理器
llm_model = ChatGLMPredictor(model_path=args.model_path, fp16=args.fp16, bits=args.bits, cache_dir=args.cache_dir,
                             local_files_only=args.local_files_only)
local_doc_qa = LocalDocQA(llm_model, embedding_model='text2vec', cache_dir=None)


filepath = input("请输入本地知识文件路径：")
if filepath:
    local_doc_qa.init_knowledge_vector_store(filepath)

session_id = None
while True:
    query = input("请输入问题：")
    last_print_len = 0
    for resp, session_id in local_doc_qa.get_knowledge_based_answer(query=query,
                                                                    session_id=session_id,
                                                                    streaming=True):
        print(resp["result"][last_print_len:], end="", flush=True)
        last_print_len = len(resp["result"])
    # 出处
    print("\n\n")
    source_text = [f"出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}"
                   for inum, doc in enumerate(resp["source_documents"])]
    print("\n".join(source_text))
