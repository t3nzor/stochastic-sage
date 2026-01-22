#!/usr/bin/env python
#
# Prerequisites:
# pip install gradio transformers accelerate torch
# 

import random, sys
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

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

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)

    new_seed = rng.getrandbits(24)
    # TODO: update page element with value and RGB color
    print(new_seed, file=sys.stderr)
    reseed(new_seed)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    bot_reply = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": bot_reply})

    return messages

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
        outputs=[chatbot]
    ).then(
        lambda: "", None, user_input
    )

demo.launch()
