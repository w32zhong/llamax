export NCCL_BLOCKING_WAIT=1  # Set this variable to use the NCCL backend
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_DISTRIBUTED_DEBUG=OFF

#export PYTHONPATH=`pwd`/accelerate/src/

#deepspeed --include=localhost:0 test_llama.py --world_size 1 ~/llama-models/7B-hgf-new --direct_inference --dtype fp8

#deepspeed --include=localhost:0 test_llama.py --world_size 1 ~/llama-models/13B-hgf-new --direct_inference --dtype fp8

deepspeed --include=localhost:0,1,2,3 test_llama.py --world_size 4 ~/llama-models/7B-hgf-new --direct_inference

#deepspeed --include=localhost:0 test_llama.py --world_size 1 ~/llama-models/30B-hgf --dtype fp8 --direct_inference

#deepspeed --include=localhost:0,1,2,3,4,5,6,7 test_llama.py --world_size 8 ~/llama-models/65B-hgf-new --direct_inference

#deepspeed --include=localhost:0,1,6,7 test_llama.py --world_size 8 ~/llama-models/30B-hgf --direct_inference

#deepspeed --include=localhost:0,1,2 test_llama.py --world_size 3 ~/llama-models/65B-hgf-new --dtype fp8 --direct_inference 

