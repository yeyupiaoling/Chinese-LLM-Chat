import argparse
import functools
import os
import platform

from utils.utils import print_arguments, add_arguments
from utils.vicuna_predictor import VicunaPredictor

parser = argparse.ArgumentParser()
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("model_path", type=str, default="./models/llama-7b-hf-finetune",   help="合并后的模型路径或者原模型名称")
add_arg("device",     type=str, choices=["cpu", "cuda", "mps"], default="cuda", help="使用哪个设备推理")
add_arg("num_gpus",   type=int, default=2, help="使用多少个GPU推理")
add_arg("stream_interval", type=int, default=2,        help="流式识别的分割大小")
add_arg("input_pattern",   type=str, default="prompt", help="使用输入的模板类型")
add_arg("load_8bit",  action="store_true",  help="是否量化模型推理")
args = parser.parse_args()
print_arguments(args)


predictor = VicunaPredictor(args.model_path, args.device, num_gpus=args.num_gpus,
                            load_8bit=args.load_8bit, stream_interval=args.stream_interval, input_pattern=args.input_pattern)

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
        param = {"prompt": query, "session_id": session_id}
        # 非流式输出
        result = predictor.generate_gate(**param)
        response = result['response']
        session_id = result['session_id']
        print(f"助手：{response}")


if __name__ == "__main__":
    main()
