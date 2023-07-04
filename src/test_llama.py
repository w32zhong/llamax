# run with either:
# (1) python test_llama.py ~/llama-models/7B-hgf/ --direct_inference
# or
# (2) deepspeed --num_gpus 2 test_llama.py --world_size 2 ~/llama-models/7B-hgf/

# Also note that CUDA_VISIBLE_DEVICES can’t be used with DeepSpeed to control which devices should be used. To specify devices:
# Replace --num_gpus ... with --include=localhost:4,5,6,7

import os
import fire
import torch
import gradio as gr
import deepspeed
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig


def main(model_path, local_rank=0, world_size=1, direct_inference=False):
    print('RANK:', local_rank, '/', world_size)

    print('Loading weights')
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path,

        torch_dtype=torch.half,
        # RuntimeError: "addmm_impl_cpu_" not implemented for 'Half'

        #device_map="auto", # buggy!!! And we should not use both
        # device_map="auto" and deepspeed.init_inference() together.
        # The former is supported by Accelerate:
        # https://github.com/microsoft/DeepSpeed/issues/3028
    )

    model.eval()
    model.to(f'cuda:{local_rank}')
    device = model.device
    print('DEV:', device)

    model = deepspeed.init_inference(model, mp_size=world_size, dtype=torch.half)

    def inference(prompt='My name is Mariama, my favorite'):
        print('prompt:', prompt)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=128
            )
        output = tokenizer.decode(generation_output[0])
        print(f'rank#{local_rank} output:', output)
        return output

    if local_rank == 0:
        if direct_inference:
            inference()
        else:
            iface = gr.Interface(fn=inference, inputs="text", outputs="text")
            iface.launch(share=True)


if __name__ == '__main__':
    fire.Fire(main)
