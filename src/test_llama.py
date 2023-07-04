# run with either:
# (1) python test_llama.py ~/llama-models/7B-hgf/ --direct_inference
# or
# (2) deepspeed --num_gpus 2 test_llama.py --world_size 2 ~/llama-models/7B-hgf/

# Also note that CUDA_VISIBLE_DEVICES can’t be used with DeepSpeed to control which devices should be used. To specify devices:
# Replace --num_gpus ... with --include=localhost:4,5,6,7

import os
import gc
import fire
import torch
import gradio as gr
import deepspeed
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def main(model_path, local_rank=0, world_size=1,
    model_dtype=torch.half, direct_inference=False):
    print('RANK:', local_rank, '/', world_size)

    print('Loading weights')
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    torch.cuda.empty_cache()
    gc.collect()

    #with deepspeed.OnDevice(dtype=model_dtype, device=f'cuda:{local_rank}'):
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype,

        device_map="auto", # buggy!!! And we should not use both
        # device_map="auto" and deepspeed.init_inference() together.
        # The former is supported by Accelerate:
        # https://github.com/microsoft/DeepSpeed/issues/3028
    )

    deepspeed.init_distributed(rank=local_rank, world_size=world_size)

    model.eval()
    device = model.device
    curdev = torch.cuda.current_device()
    print('DEVICE:', device)
    print('CURDEV:', curdev)

    #ds_engine = deepspeed.init_inference(model,
    #    mp_size=world_size,
    #    dtype=model_dtype,
    #    replace_with_kernel_inject=False # buggy otherwise
    #)
    #model = ds_engine.module

    def inference(prompt='My name is Mariama, my favorite '):
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
            iface = gr.Interface(fn=inference,
                inputs="text", outputs="text")
            # Enabling the queue for inference times > 60 seconds:
            iface.queue().launch(debug=True, share=True, inline=False)

    torch.distributed.barrier()


if __name__ == '__main__':
    fire.Fire(main)
