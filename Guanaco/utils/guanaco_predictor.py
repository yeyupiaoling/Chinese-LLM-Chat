import uuid
from threading import Thread
from typing import Dict, List

import torch
from transformers import TextIteratorStreamer, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, AutoConfig

from utils.model_utils import get_logits_processor
from utils.template import Template


class Predictor:
    def __init__(self, model_path, fp16=True, bits=8, cache_dir='cache', local_files_only=False):
        self.prompt_template = Template(name="default")
        self.histories: Dict[str, List] = {}
        # 加载模型
        self.model, self.tokenizer = self.load_model(model_path, fp16=fp16, bits=bits, cache_dir=cache_dir,
                                                     local_files_only=local_files_only)
        self.model.eval()
        # 生成参数
        self.gen_kwargs = {"do_sample": True, "top_k": 50, "num_beams": 1, "repetition_penalty": 1.0}

    @staticmethod
    def load_model(model_path, fp16=True, bf16=False, bits=8, double_quant=True, quant_type="nf4", cache_dir='cache',
                   local_files_only=False):
        config_kwargs = {"trust_remote_code": True, "cache_dir": cache_dir, "local_files_only": local_files_only}

        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", use_fast=False, **config_kwargs)
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id

        config = AutoConfig.from_pretrained(model_path, **config_kwargs)
        config_kwargs["device_map"] = "auto"
        compute_dtype = (torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float32))
        if bits == 8:
            config_kwargs["load_in_8bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        elif bits == 4:
            config_kwargs["load_in_4bit"] = True
            config_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=double_quant,
                bnb_4bit_quant_type=quant_type)

        # 加载模型
        torch_dtype = (torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32))
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     config=config,
                                                     torch_dtype=torch_dtype,
                                                     low_cpu_mem_usage=True,
                                                     **config_kwargs)
        model.requires_grad_(False)
        if bits not in [4, 8]:
            model = model.half()

        return model, tokenizer

    def generate_stream(self, prompt, max_new_tokens=512, top_p=0.7, temperature=0.95, session_id=None):
        # 如果session_id存在
        if session_id and session_id in self.histories.keys():
            history = self.histories[session_id]
            input_prompt = self.prompt_template.get_prompt(prompt, history=history)
        else:
            # 否则，创建一个新的session_id
            session_id = str(uuid.uuid4()).replace('-', '')
            self.histories[session_id] = []
            input_prompt = self.prompt_template.get_prompt(prompt, history=None)
        input_ids = self.tokenizer([input_prompt], return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(self.model.device)
        self.histories[session_id].append([prompt, None])

        streamer = TextIteratorStreamer(self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        # 生成参数
        self.gen_kwargs["temperature"] = temperature
        self.gen_kwargs["top_p"] = top_p
        self.gen_kwargs["max_new_tokens"] = max_new_tokens
        self.gen_kwargs["input_ids"] = input_ids
        self.gen_kwargs["logits_processor"] = get_logits_processor()
        self.gen_kwargs["streamer"] = streamer

        thread = Thread(target=self.model.generate, kwargs=self.gen_kwargs)
        thread.start()

        response = ""
        for new_text in streamer:
            response += new_text
            ret = {"response": response, "code": 0, "session_id": session_id}
            # 在模型回复添加到历史里面
            self.histories[session_id][-1][1] = response
            yield ret

    # 非流式识别
    def generate(self, prompt, max_new_tokens=512, top_p=0.7, temperature=0.95, session_id=None):
        # 如果session_id存在
        if session_id and session_id in self.histories.keys():
            history = self.histories[session_id]
            input_prompt = self.prompt_template.get_prompt(prompt, history=history)
        else:
            # 否则，创建一个新的session_id
            session_id = str(uuid.uuid4()).replace('-', '')
            self.histories[session_id] = []
            input_prompt = self.prompt_template.get_prompt(prompt, history=None)
        input_ids = self.tokenizer([input_prompt], return_tensors="pt")["input_ids"]
        input_ids = input_ids.to(self.model.device)
        self.histories[session_id].append([prompt, None])
        # 生成参数
        self.gen_kwargs["temperature"] = temperature
        self.gen_kwargs["top_p"] = top_p
        self.gen_kwargs["max_new_tokens"] = max_new_tokens
        self.gen_kwargs["input_ids"] = input_ids
        self.gen_kwargs["logits_processor"] = get_logits_processor()
        with torch.no_grad():
            generation_output = self.model.generate(**self.gen_kwargs)
        outputs = generation_output.tolist()[0][len(input_ids[0]):]
        response = self.tokenizer.decode(outputs, skip_special_tokens=True)
        # 在模型回复添加到历史里面
        self.histories[session_id][-1][1] = response
        return {"response": response, "code": 0, "session_id": session_id}
