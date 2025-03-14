import sys

import fire
import torch
from peft import PeftModel
import transformers
import gradio as gr
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def main(
    load_8bit: bool = False,
    base_model: str = "output/merged/",
    #lora_model: str = "output/adapter",
):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    print('Loading weights')
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        #model = PeftModel.from_pretrained(
        #    model, lora_model, torch_dtype=torch.float16
        #)
    else:
        assert False

    print('Loading weights [done]')
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    #if not load_8bit:
    #    model.half()  # seems to fix bugs for some users.

    model.eval()
    #if torch.__version__ >= "2" and sys.platform != "win32":
    #    model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.6,
        top_p=0.9,
        top_k=40,
        num_beams=4,
        max_new_tokens=512,
        **kwargs,
    ):
        prompt = generate_prompt(instruction, input)
        print('prompt:', prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        print('output:', output)
        return output

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(
                lines=2, label="Instruction", placeholder="Tell me about alpacas."
            ),
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(minimum=0, maximum=1, value=0.1, label="Temperature"),
            gr.components.Slider(minimum=0, maximum=1, value=0.75, label="Top p"),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(minimum=1, maximum=4, step=1, value=4, label="Beams"),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="Llama-X",
        description="Improve LLaMA model to follow instructions.",
    ).launch(share=True)


def generate_prompt(instruction, input=None):
    if input:
        return f"""hello! {instruction} {input}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request."""


if __name__ == "__main__":
    fire.Fire(main)
