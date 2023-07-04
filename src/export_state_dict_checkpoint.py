import torch
from pdb import set_trace as bp
from peft import PeftModel
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig

base_model: str = "output/checkpoint-3200/"
lora_model: str = "output/adapter"

if False:
    print('loading base model ...')
    base_model = torch.load('output/checkpoint-3200/pytorch_model.bin')
    print('loading lora model ...')
    lora_model = torch.load('output/adapter/adapter_model.bin')

    new_state_dict = {}
    for k, v in base_model.items():
        k = k.replace("base_model.model.model.", "")
        new_state_dict[k] = v

    for k, v in lora_model.items():
        k = k.replace("base_model.model.model.", "")
        new_state_dict[k] = v

    torch.save(new_state_dict, "./output/merged_states")
else:
    print('loading base model (7B takes 10 min) ...')
    base_model = LlamaForCausalLM.from_pretrained(base_model,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    print('loading lora model ...')
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_model,
        torch_dtype=torch.float16
    )

    print('merging weights ...')
    for layer in lora_model.base_model.model.model.layers:
        layer.self_attn.q_proj.merge_weights = True
        layer.self_attn.v_proj.merge_weights = True
    lora_model.train(False)
    merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained('output/merged')
