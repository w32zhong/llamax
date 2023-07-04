export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_BLOCKING_WAIT=1  # Set this variable to use the NCCL backend
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export TORCH_DISTRIBUTED_DEBUG=OFF

deepspeed --include=localhost:0,1,2,3,4,5,6,7 test_llama.py ~/llama-models/13B-hgf-new

#--deepspeed configs/deepspeed_config.json
