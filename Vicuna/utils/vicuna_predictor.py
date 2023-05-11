import json
import uuid
from typing import Dict

import torch
from fastchat.conversation import get_conv_template, Conversation
from fastchat.serve.inference import load_model, generate_stream


class VicunaPredictor:
    def __init__(self, model_path, device, num_gpus, load_8bit=False, stream_interval=2, input_pattern="prompt"):
        self.device = device
        self.stream_interval = stream_interval
        # 加载模型
        self.model, self.tokenizer = load_model(model_path, device=device, num_gpus=num_gpus, load_8bit=load_8bit)
        self.model.eval()

        if hasattr(self.model.config, "max_sequence_length"):
            self.context_len = self.model.config.max_sequence_length
        elif hasattr(self.model.config, "max_position_embeddings"):
            self.context_len = self.model.config.max_position_embeddings
        else:
            self.context_len = 2048
        self.histories: Dict[str, Conversation] = {}
        if input_pattern == "prompt":
            self.conv = get_conv_template('dolly_v2')
        elif input_pattern == "chat":
            self.conv = get_conv_template('vicuna_v1.1')
        else:
            raise ValueError("input_pattern must be either 'prompt' or 'chat'")

    def generate_stream_gate(self, prompt, max_length=512, top_p=1.0, temperature=0.95, session_id=None):
        # 如果session_id存在，则将conv赋值给conv
        if session_id and session_id in self.histories.keys():
            conv = self.histories[session_id]
        else:
            # 否则，创建一个新的session_id，并创建一个新的Conversation
            session_id = str(uuid.uuid4()).replace('-', '')
            conv = self.conv.copy()
        # 将prompt添加到conv
        conv.append_message(conv.roles[0], prompt)
        # 将空字符添加到conv，用于占位模型回复
        conv.append_message(conv.roles[1], "")
        # 获取prompt
        prompt = conv.get_prompt()
        # 模型参数
        gen_params = {
            "prompt": prompt,
            "top_p": top_p,
            "temperature": temperature,
            "max_new_tokens": max_length,
            "stop": conv.stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }
        try:
            # 流式识别
            for output in generate_stream(self.model, self.tokenizer, gen_params, self.device, self.context_len,
                                          self.stream_interval):
                ret = {"response": output, "code": 0, "session_id": session_id}
                # 在模型回复添加到历史里面
                conv.messages[-1][-1] = output.strip()
                self.histories[session_id] = conv
                # 返回json格式的字节
                yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.OutOfMemoryError:
            ret = {"response": "显存不足", "code": 1, "session_id": session_id}
            yield json.dumps(ret).encode() + b"\0"

    # 非流式识别
    def generate_gate(self, prompt, max_length=512, top_p=1.0, temperature=0.95, session_id=None):
        generator = self.generate_stream_gate(prompt=prompt, max_length=max_length, top_p=top_p,
                                              temperature=temperature, session_id=session_id)
        result = ""
        session_id = None
        error_code = 0
        for output in generator:
            output = json.loads(output[:-1].decode("utf-8"))
            session_id = output['session_id']
            code = output['code']
            if code != 0:
                error_code = code
                break
            result = output['response']
        return {"response": result, "code": error_code, "session_id": session_id}
