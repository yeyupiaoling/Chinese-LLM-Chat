import argparse
import functools
import os
import platform
import sys

from utils.utils import print_arguments, add_arguments
from utils.chatglm_predictor import ChatGLMPredictor

parser = argparse.ArgumentParser()
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("model_path", type=str, default="./models/chatglm-6b-finetune",  help="合并后的模型路径或者原模型名称")
add_arg("cache_dir",  type=str, default="cache/",               help="模型缓存目录")
add_arg("device",     type=str, choices=["cpu", "cuda", "mps"], default="cuda", help="使用哪个设备推理")
add_arg("num_gpus",   type=int, default=2,  help="使用多少个GPU推理")
add_arg("input_pattern", type=str, default="prompt", help="使用输入的模板类型")
add_arg("load_8bit",  type=bool, default=False, help="是否量化模型推理")
args = parser.parse_args()
print_arguments(args)


# 获取模型推理器
predictor = ChatGLMPredictor(args.model_path, args.device, num_gpus=args.num_gpus, cache_dir=args.cache_dir,
                             load_8bit=args.load_8bit, input_pattern=args.input_pattern)

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
        # 流式输出
        generator = predictor.generate_stream(prompt=query, session_id=session_id)
        print("助手：", end="", flush=True)
        for output in generator:
            session_id = output['session_id']
            result = output['response']
            print(result, end="", flush=True)
        print()


if __name__ == "__main__":
    main()
