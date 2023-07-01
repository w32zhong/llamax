```bash
conda create -n llamax python=3.10
conda activate llamax
git clone https://github.com/AetherCortex/Llama-X.git
cd Llama-X

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
python -c 'import torch; print(torch.cuda.is_available())'
pip install transformers==4.29.2

conda install cuda -c nvidia/label/cuda-11.4.0
pip install -r requirements.txt
```

- Training data example (e.g., [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json)):
```bash
data/alpaca_data.json
```

- Convert LLaMA checkpoint to HuggingFace format:

https://gist.github.com/w32zhong/9bf98e3d9aa9b32854d1e389cdf3d666#file-llama-sh-L28

- Train LLaMA-7B on DeepSpeed Zero-3:
```bash
./train.sh
```

- Inference
```bash
# web demo inference
python generate.py

# batch inference
To Do
```
