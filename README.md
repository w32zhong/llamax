```bash
conda create -n llamax python=3.10
conda activate llamax
git clone https://github.com/AetherCortex/Llama-X.git
cd Llama-X

pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
python -c 'import torch; print(torch.cuda.is_available())'
python -c "import torch; print(torch.__version__)"
pip install transformers==4.29.2
pip install -r requirements.txt

conda install -c "nvidia/label/cuda-12.1.1" cuda # other versions will stuck on watgpu-100 node! Thanks Tunde!
nvcc --version
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
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
./train.sh
```

- Inference
```bash
# web demo inference
python generate.py

# batch inference
To Do
```
