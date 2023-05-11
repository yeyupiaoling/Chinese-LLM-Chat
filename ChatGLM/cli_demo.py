import argparse
import os
import platform

from utils.utils import print_arguments
from utils.chatglm_predictor import ChatGLMPredictor

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./models/chatglm-6b-finetune",  help="合并后的模型路径或者原模型名称")
parser.add_argument("--cache_dir",  type=str, default="cache/",               help="模型缓存目录")
parser.add_argument("--device",     type=str, choices=["cpu", "cuda", "mps"], default="cuda", help="使用哪个设备推理")
parser.add_argument("--num_gpus",   type=int, default=2,  help="使用多少个GPU推理")
parser.add_argument("--input_pattern", type=str, default="prompt", help="使用输入的模板类型")
parser.add_argument("--load_8bit",  action="store_true",  help="是否量化模型推理")
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
        query = input("\n用户：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            session_id = None
            os.system(clear_command)
            print("输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        # 非流式输出
        result = predictor.generate_gate(prompt=query, session_id=session_id)
        response = result['response']
        session_id = result['session_id']
        print(f"助手：{response}")


if __name__ == "__main__":
    main()
