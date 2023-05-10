import json
import uuid
from typing import Dict

import torch

from utils.chatglm_parallel import load_model


class Predictor:
    def __init__(self, model_path, device, num_gpus, load_8bit=False, cache_dir=None, input_pattern='prompt'):
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        self.device = device
        self.input_pattern = input_pattern
        # 加载模型
        self.tokenizer, self.model = load_model(model_path=model_path, device=device, num_gpus=num_gpus,
                                                load_8bit=load_8bit, cache_dir=cache_dir)
        self.histories: Dict[str, list] = {}

    # 开始流式识别
    def generate_stream_gate(self, prompt, max_length=2048, top_p=0.7, temperature=0.95, session_id=None):
        if self.input_pattern == 'prompt':
            prompt = f"Instruction: {prompt}\nAnswer: "
        # 如果session_id存在之前的历史，则获取history
        if session_id and session_id in self.histories.keys():
            history = self.histories[session_id]
        else:
            # 否则创建新的session_id
            session_id = str(uuid.uuid4()).replace('-', '')
            history = None
        try:
            # 开始流式识别
            for response, history_out in self.model.stream_chat(self.tokenizer, prompt, history, max_length=max_length,
                                                                top_p=top_p, temperature=temperature):
                ret = {"response": response, "code": 0, "session_id": session_id}
                # 更新history
                self.histories[session_id] = history_out
                # 返回json格式的字节
                yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.OutOfMemoryError:
            ret = {"response": "显存不足", "code": 1, "session_id": session_id}
            yield json.dumps(ret).encode() + b"\0"

    # 非流式识别
    def generate_gate(self, prompt, max_length=2048, top_p=0.7, temperature=0.95, session_id=None):
        # 如果session_id存在于histories中，则获取history
        if session_id and session_id in self.histories.keys():
            history = self.histories[session_id]

        else:
            # 否则创建一个session_id
            session_id = str(uuid.uuid4()).replace('-', '')
            history = None
        try:
            # 调用model.chat函数
            response, history_out = self.model.chat(self.tokenizer, prompt, history, max_length=max_length,
                                                    top_p=top_p, temperature=temperature)
            ret = {"response": response, "code": 0, "session_id": session_id}
            # 更新history
            self.histories[session_id] = history_out
            return ret
        except torch.cuda.OutOfMemoryError:
            ret = {"response": "显存不足", "code": 1, "session_id": session_id}
            return ret
