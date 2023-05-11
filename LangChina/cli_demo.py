import argparse
import os

from utils.local_doc_qa import LocalDocQA
from utils.chatglm_predictor import ChatGLMPredictor

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="THUDM/chatglm-6b",  help="合并后的模型路径或者原模型名称")
parser.add_argument("--cache_dir",  type=str, default="cache/",               help="模型缓存目录")
parser.add_argument("--device",     type=str, choices=["cpu", "cuda", "mps"], default="cuda", help="使用哪个设备推理")
parser.add_argument("--num_gpus",   type=int, default=2,  help="使用多少个GPU推理")
parser.add_argument("--input_pattern", type=str, default="prompt", help="使用输入的模板类型")
parser.add_argument("--load_8bit",  action="store_true",  help="是否量化模型推理")
args = parser.parse_args()
print_arguments(args)


llm_model = ChatGLMPredictor(args.model_path, args.device, num_gpus=args.num_gpus, cache_dir=args.cache_dir,
                             load_8bit=args.load_8bit, input_pattern=args.input_pattern)
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
