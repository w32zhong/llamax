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
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from accelerate import infer_auto_device_map, dispatch_model, init_empty_weights, load_checkpoint_in_model


def main(model_path, local_rank=0, world_size=1,
    dtype='fp16', direct_inference=False):
    print('RANK:', local_rank, '/', world_size)

    load_in_8bit = False
    dtype_dict = {
        'fp32': torch.float,
        'fp16': torch.half,
        'bp16': torch.bfloat16,
    }
    if dtype == 'fp8':
        dtype = 'bp16'
        load_in_8bit = True
    model_dtype = dtype_dict[dtype]
    print('DTYPE:', model_dtype)

    print('Loading weights')
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    torch.cuda.empty_cache()
    gc.collect()

    max_memory = {
        0: "5GiB",
        1: "5GiB",
        2: "5GiB",
        3: "5GiB",
        4: "5GiB",
        5: "5GiB",
        6: "5GiB",
        7: "5GiB",
    }

    with init_empty_weights():
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            offload_folder="offload",
            torch_dtype=model_dtype,
            load_in_8bit=load_in_8bit,
            #max_memory=max_memory
        )
    if local_rank == 0:
        print(model.hf_device_map)

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
