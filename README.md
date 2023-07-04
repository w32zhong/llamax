```bash
conda create -n llamax python=3.10
conda activate llamax
git clone https://github.com/AetherCortex/Llama-X.git
cd Llama-X

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
python -c 'import torch; print(torch.cuda.is_available())'
python -c "import torch; print(torch.__version__)"
pip install transformers==4.29.2
pip install -r requirements.txt

# optional: to use deepspeed==0.9.5 (removed some weird warnings): 
conda install -c "nvidia/label/cuda-12.1.1" cuda 
nvcc --version
pip install deepspeed==0.9.5
```

- Training data example (e.g., [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)):
```bash
data/alpaca_data.json
```

- Convert LLaMA checkpoint to HuggingFace format:

https://gist.github.com/w32zhong/9bf98e3d9aa9b32854d1e389cdf3d666#file-llama-sh-L28

- Train LLaMA-7B on DeepSpeed Zero-3:
```bash
ps -up `nvidia-smi -q -x | grep -Po '(?<=<pid>)[0-9]+'`
./train.sh
```

- Inference
```bash
python generate.py
```
Conversion of Weights
```
>>> import torch                                                                                                                                                             
>>> m=torch.load('output/checkpoint-3200/pytorch_model.bin')                                                                                                                 
>>> n=torch.load('output/adapter/adapter_model.bin')                                                                                                                         
>>>                                                                                                                                                                          
>>> n = n['base_model.model.model.layers.31.self_attn.q_proj.lora_A.weight'].to('cpu')                                                                                       
KeyboardInterrupt                                                                                                                                                            
>>> m = m['base_model.model.model.layers.31.self_attn.q_proj.lora_A.default.weight']                                                                                         
KeyboardInterrupt                                                                                                                                                            
>>> list(filter(lambda x: '31' in x, n.keys()))                                                                                                                              
['base_model.model.model.layers.31.self_attn.q_proj.lora_A.weight', 'base_model.model.model.layers.31.self_attn.q_proj.lora_B.weight', 'base_model.model.model.layers.31.self_attn.v_proj.lora_A.weight', 'base_model.model.model.layers.31.self_attn.v_proj.lora_B.weight']                                                                              
>>> list(filter(lambda x: '31' in x, m.keys()))                                                                                                                              
['base_model.model.model.layers.31.self_attn.q_proj.weight', 'base_model.model.model.layers.31.self_attn.q_proj.lora_A.default.weight', 'base_model.model.model.layers.31.self_attn.q_proj.lora_B.default.weight', 'base_model.model.model.layers.31.self_attn.k_proj.weight', 'base_model.model.model.layers.31.self_attn.v_proj.weight', 'base_model.model.model.layers.31.self_attn.v_proj.lora_A.default.weight', 'base_model.model.model.layers.31.self_attn.v_proj.lora_B.default.weight', 'base_model.model.model.layers.31.self_attn.o_proj.weight', 'base_model.model.model.layers.31.self_attn.rotary_emb.inv_freq', 'base_model.model.model.layers.31.mlp.gate_proj.weight', 'base_model.model.model.layers.31.mlp.down_proj.weight', 'base_model.model.model.layers.31.mlp.up_proj.weight', 'base_model.model.model.layers.31.input_layernorm.weight', 'base_model.model.model.layers.31.post_attention_layernorm.weight'] 
```
