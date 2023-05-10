from typing import Dict

from accelerate import load_checkpoint_and_dispatch
from torch.nn import Module
from transformers import AutoModel, AutoTokenizer


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    # transformer.word_embeddings 占用1层
    # transformer.final_layernorm 和 lm_head 占用1层
    # transformer.layers 占用 28 层
    # 总共30层分配到num_gpus张卡上
    num_trans_layers = 28
    per_gpu_layers = 30 / num_gpus

    # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
    # windows下 model.device 会被设置成 transformer.word_embeddings.device
    # linux下 model.device 会被设置成 lm_head.device
    # 在调用chat或者stream_chat时,input_ids会被放到model.device上
    # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
    # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
    device_map = {'transformer.word_embeddings': 0,
                  'transformer.final_layernorm': 0, 'lm_head': 0}

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.layers.{i}'] = gpu_target
        used += 1

    return device_map


def load_model(model_path, device, num_gpus, load_8bit=False, debug=False, cache_dir=None) -> (Module, Module):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)

    if num_gpus > 1 and device == 'cuda':
        device_map = auto_configure_device_map(num_gpus)
        try:
            model = load_checkpoint_and_dispatch(
                model, model_path, device_map=device_map, offload_folder="offload", offload_state_dict=True).half()
        except ValueError:
            # index.json not found
            multi_gpu_model_cache_dir = "./temp_model_dir"
            print(f"index.json not found, auto fixing and saving model to {multi_gpu_model_cache_dir} ...")

            assert multi_gpu_model_cache_dir is not None, "using auto fix, cache_dir must not be None"
            model.save_pretrained(multi_gpu_model_cache_dir, max_shard_size='2GB')
            model = load_checkpoint_and_dispatch(model, multi_gpu_model_cache_dir, device_map=device_map,
                                                 offload_folder="offload", offload_state_dict=True).half()

            print(f"loading model successfully, you should use checkpoint_path={multi_gpu_model_cache_dir} next time")

    if load_8bit:
        model = model.quantize(8).half()

    if (device == "cuda" and num_gpus == 1) or device == "mps":
        model.half().to(device)

    if debug:
        print(model)
    model.eval()
    return tokenizer, model
