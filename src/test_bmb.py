import os
import sys
import json
import torch
import datetime
import bmtrain as bmt
from functools import partial
from collections import OrderedDict

from model_center.model import Llama, LlamaConfig
from model_center.tokenizer import LlamaTokenizer
from model_center.generation.llama import LlamaRandomSampling


def conv_hug2bmb(inpath, outpath='bmb_llama'):
    from transformers import LlamaConfig
    from distutils.file_util import copy_file
    hf_config = LlamaConfig.from_pretrained(inpath)
    config = {
        'dim_model': hf_config.hidden_size,
        'dim_ff': hf_config.intermediate_size,
        'num_layers': hf_config.num_hidden_layers,
        'num_heads': hf_config.num_attention_heads,
        'dim_head': hf_config.hidden_size // hf_config.num_attention_heads,
    }

    with open(os.path.join(inpath, "pytorch_model.bin.index.json"), 'r') as f:
        index = json.load(f)
    shards = set([v for k, v in index["weight_map"].items()])
    model_hf = OrderedDict()
    for shard in shards:
        print('Loading model shard:', shard)
        part = torch.load(
            os.path.join(inpath, shard)
        )
        model_hf.update(part)

    out = OrderedDict()
    copied = dict()
    def copy(new_key, old_key):
        out[new_key] = model_hf[old_key].contiguous().half()
        copied[old_key] = True
    copy("input_embedding.weight", 'model.embed_tokens.weight')
    copy("encoder.output_layernorm.weight", 'model.norm.weight')
    copy('output_projection.weight', 'lm_head.weight')
    for lnum in range(config['num_layers']):
        hf_pfx = f"model.layers.{lnum}"
        bmt_pfx = f"encoder.layers.{lnum}"
        copy(f"{bmt_pfx}.self_att.layernorm_before_attention.weight",
            f"{hf_pfx}.input_layernorm.weight")
        copy(f"{bmt_pfx}.self_att.self_attention.project_q.weight",
            f"{hf_pfx}.self_attn.q_proj.weight")
        copy(f"{bmt_pfx}.self_att.self_attention.project_k.weight",
            f"{hf_pfx}.self_attn.k_proj.weight")
        copy(f"{bmt_pfx}.self_att.self_attention.project_v.weight",
            f"{hf_pfx}.self_attn.v_proj.weight")
        copy(f"{bmt_pfx}.self_att.self_attention.attention_out.weight",
            f"{hf_pfx}.self_attn.o_proj.weight")
        copy(f"{bmt_pfx}.ffn.layernorm_before_ffn.weight",
            f"{hf_pfx}.post_attention_layernorm.weight")
        copy(f"{bmt_pfx}.ffn.ffn.w_in.w_0.weight",
            f"{hf_pfx}.mlp.gate_proj.weight")
        copy(f"{bmt_pfx}.ffn.ffn.w_in.w_1.weight",
            f"{hf_pfx}.mlp.up_proj.weight")
        copy(f"{bmt_pfx}.ffn.ffn.w_out.weight",
            f"{hf_pfx}.mlp.down_proj.weight")

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    print('saving model ...')

    with open(os.path.join(outpath, "config.json"), 'w') as f:
        json.dump(config, f)

    copy_file(
        os.path.join(inpath, "tokenizer.model"),
        os.path.join(outpath, "tokenizer.model")
    )
    copy_file(
        os.path.join(inpath, "tokenizer.json"),
        os.path.join(outpath, "tokenizer.json")
    )
    copy_file(
        os.path.join(inpath, "tokenizer_config.json"),
        os.path.join(outpath, "tokenizer_config.json")
    )
    copy_file(
        os.path.join(inpath, "special_tokens_map.json"),
        os.path.join(outpath, "special_tokens_map.json")
    )

    all_keys = set(model_hf.keys())
    copied_keys = set(copied.keys())
    diff_keys = all_keys.difference(copied_keys)
    print('diff_keys:', diff_keys)
    torch.save(out, os.path.join(outpath, "pytorch_model.pt"))


def generate(generator, device, prompt):
    print('prompt:', prompt)
    with torch.no_grad():
        output = generator.generate([prompt])
    print(output)
    return output


def inference(model_path, **kargs):
    def get_arg(k, d=None):
        return kargs[k] if k in kargs else d
    zero_level = get_arg('zero_level', 2)
    local_rank = get_arg('local_rank')
    token_path = get_arg('token_path', model_path)
    token_path = os.path.expanduser(token_path)
    debug = get_arg('debug')

    if local_rank is not None: 
        torch.distributed.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(0, 5 * 60),
        )

    bmt.init_distributed(seed=123)
    config = LlamaConfig.from_pretrained(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(token_path)
    model = Llama(config)
    model.device = 'cuda:0'
    model.eval()
    if local_rank == 0:
        print('model loaded.')

    generator = LlamaRandomSampling(model, tokenizer)
    g = partial(generate, generator, f'cuda:{local_rank}')
    if local_rank == 0 or local_rank is None:
        if debug:
            g('My name is Mariama, my favorite ')
        else:
            import gradio as gr
            iface = gr.Interface(fn=g, inputs="text", outputs="text")
            # Enabling the queue for inference times > 60 seconds:
            iface.queue().launch(debug=True, share=True, inline=False)
    else:
        torch.distributed.barrier()


if __name__ == "__main__":
    import fire
    #fire.Fire(inference)
    fire.Fire(conv_hug2bmb)
