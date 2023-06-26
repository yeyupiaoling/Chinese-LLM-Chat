import uuid
from typing import Dict

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from ChatGLM.utils.model_utils import auto_configure_device_map


class ChatGLMPredictor:
    def __init__(self, model_path, fp16=True, bits=8, cache_dir='cache', local_files_only=False):
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        # 加载模型
        self.model, self.tokenizer = self.load_model(model_path=model_path, fp16=fp16, bits=bits, cache_dir=cache_dir,
                                                     local_files_only=local_files_only)
        self.histories: Dict[str, list] = {}

    @staticmethod
    def load_model(model_path=None, fp16=True, bf16=False, bits=8, cache_dir='cache', local_files_only=False):
        config_kwargs = {"trust_remote_code": True, "cache_dir": cache_dir, "local_files_only": local_files_only}
        # 获取tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, **config_kwargs)

        config = AutoConfig.from_pretrained(model_path, **config_kwargs)
        # 加载模型
        torch_dtype = (torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32))
        model = AutoModel.from_pretrained(model_path, torch_dtype=torch_dtype, config=config, **config_kwargs)
        # 量化模型
        model.requires_grad_(False)
        model = model.half()
        model.quantize(bits)
        # 多卡处理
        if torch.cuda.device_count() > 1:
            from accelerate import dispatch_model
            device_map = auto_configure_device_map(torch.cuda.device_count())
            model = dispatch_model(model, device_map)
        else:
            model = model.cuda()

        model.eval()
        return model, tokenizer

    # 开始流式识别
    def generate_stream(self, prompt, max_length=2048, top_p=0.7, temperature=0.95, session_id=None):
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
                yield ret
        except torch.cuda.OutOfMemoryError:
            ret = {"response": "显存不足", "code": 1, "session_id": session_id}
            yield ret

    # 非流式识别
    def generate(self, prompt, max_length=2048, top_p=0.7, temperature=0.95, session_id=None):
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
