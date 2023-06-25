import argparse
import functools
import os
import platform
import sys

from utils.guanaco_predictor import Predictor
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser()
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("model_path", type=str,  default="./models/llama-7b-finetune",   help="合并后的模型路径")
add_arg("lora_model", type=str,  default=None,        help="不使用合并模型，直接使用Lora模型")
add_arg("cache_dir",  type=str,  default="cache/",    help="模型缓存目录")
add_arg("bits",       type=int,  default=4,           help="使用量化多少位")
add_arg("fp16",       type=bool, default=False,       help="是否半精度推理")
add_arg("local_files_only", type=bool, default=False, help="是否只在本地加载模型，不尝试下载")
args = parser.parse_args()
print_arguments(args)


predictor = Predictor(model_path=args.model_path, lora_model=args.lora_model, fp16=args.fp16, bits=args.bits,
                      cache_dir=args.cache_dir, local_files_only=args.local_files_only)

os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
session_id = None


def main():
    global session_id
    print("输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        try:
            query = input("\n用户：")
        except UnicodeDecodeError:
            print("Detected decoding error at the inputs, please set the terminal encoding to utf-8.", file=sys.stderr)
            continue
        except Exception:
            raise
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            session_id = None
            os.system(clear_command)
            print("输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        param = {"prompt": query, "session_id": session_id}
        # 流式输出
        last_len = 0
        print("助手：", end="", flush=True)
        generator = predictor.generate_stream(**param)
        for output in generator:
            session_id = output['session_id']
            result = output['response']
            print(result[last_len:], end="", flush=True)
            last_len = len(result)
        print()


if __name__ == "__main__":
    main()
