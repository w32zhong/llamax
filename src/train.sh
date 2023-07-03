export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_BLOCKING_WAIT=1  # Set this variable to use the NCCL backend
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_P2P_DISABLE=1
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_DISTRIBUTED_DEBUG=OFF

#--hostfile configs/hostfile \
#--num_gpus 8 \
#--num_nodes 1 \

# Zero-s3 on Ada 6000 (49G) maximum batch sizes:
#  7B w/o Lora: batch size = 32
#  30B w/ Lora: batch size = 20

deepspeed \
	--master_port 8900 \
	train.py \
	--model_name_or_path ~/llama-models/7B-hgf/ \
	--data_path ../data/alpaca_data.json \
	--output_dir ./output/ \
	--num_train_epochs 1 \
	--model_max_length 2048 \
	--per_device_train_batch_size 12 \
	--per_device_eval_batch_size 1 \
	--gradient_accumulation_steps 1 \
	--evaluation_strategy "no" \
	--save_strategy "steps" \
	--save_steps 100 \
	--save_total_limit 2 \
	--learning_rate 2e-5 \
	--warmup_steps 2 \
	--logging_steps 2 \
	--lr_scheduler_type "cosine" \
	--report_to "tensorboard" \
	--gradient_checkpointing True \
	--deepspeed configs/deepspeed_config.json \
	--fp16 True \

	#--max_steps 1
