#!/home/t3nzor/venv/bin/python
#
# Prerequisites:
# pip install gradio transformers accelerate torch
# 

import queue
import random, sys
from threading import Thread
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_ID = "Qwen/Qwen3.5-9B"

SYSTEM_PROMPT = (
    "You are a helpful, accurate, and concise AI assistant. "
    "Explain things clearly and ask clarifying questions when needed."
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

rng = random.SystemRandom()
def reseed(seed):
    torch.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)

class TokenStreamer:
    """Collects generated token IDs and makes full decoded text available via an iterator."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.token_ids: list[int] = []
        self.queue: queue.Queue[str | None] = queue.Queue()

    def put(self, value):
        if len(value.shape) > 1:
            value = value[0]
        self.token_ids.append(value[-1].item())
        text = self.tokenizer.decode(self.token_ids, skip_special_tokens=True)
        self.queue.put(text)

    def end(self):
        self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.queue.get()
        if value is None:
            raise StopIteration
        return value


def chat(user_input, messages):
    if messages is None:
        messages = []

    # Build messages with system prompt
    qwen_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    qwen_messages.extend(messages)
    qwen_messages.append({"role": "user", "content": user_input})

    # Apply Qwen chat template
    prompt = tokenizer.apply_chat_template(
        qwen_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    new_seed = rng.getrandbits(24)
    # TODO: update page element with value and RGB color
    print(new_seed, file=sys.stderr)
    reseed(new_seed)

    streamer = TokenStreamer(tokenizer)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=3000,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        streamer=streamer,
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": ""})

    bot_reply = ""
    for text in streamer:
        bot_reply = text
        messages[-1] = {"role": "assistant", "content": bot_reply}
        yield messages, messages, ""

    thread.join()

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Stochastic Sage")

    chatbot = gr.Chatbot()
    state = gr.State([])

    user_input = gr.Textbox(
        placeholder="Ask me anything...",
        show_label=False
    )

    user_input.submit(
        chat,
        inputs=[user_input, state],
        outputs=[chatbot, state, user_input]
    )

demo.queue(default_concurrency_limit=1).launch(share=True)
