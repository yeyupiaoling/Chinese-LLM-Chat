import argparse
import functools

import gradio as gr
import mdtex2html

from utils.chatglm_predictor import ChatGLMPredictor
from utils.utils import print_arguments, add_arguments

parser = argparse.ArgumentParser()
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg("model_path", type=str,  default="./models/chatglm2-6b-finetune",  help="合并后的模型路径或者原模型名称")
add_arg("cache_dir",  type=str,  default="cache/",    help="模型缓存目录")
add_arg("bits",       type=int,  default=4,           help="使用量化多少位")
add_arg("fp16",       type=bool, default=True,        help="是否半精度推理")
add_arg("local_files_only", type=bool, default=False, help="是否只在本地加载模型，不尝试下载")
add_arg("share",      type=bool, default=False,  help="是否共享链路")
args = parser.parse_args()
print_arguments(args)


# 获取模型推理器
predictor = ChatGLMPredictor(model_path=args.model_path, fp16=args.fp16, bits=args.bits, cache_dir=args.cache_dir,
                             local_files_only=args.local_files_only)


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, session_id):
    chatbot.append((parse_text(input), ""))
    generater = predictor.generate_stream(prompt=input, session_id=session_id, max_length=max_length, top_p=top_p,
                                          temperature=temperature)
    for output in generater:
        session_id = output['session_id']
        code = output['code']
        if code != 0:
            break
        result = output['response']
        chatbot[-1] = (parse_text(input), parse_text(result))
        yield chatbot, session_id


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">ChatGLM</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)

    session_id = gr.State('')

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, session_id], [chatbot, session_id],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, session_id], show_progress=True)

demo.queue().launch(server_name="0.0.0.0", share=args.share)
