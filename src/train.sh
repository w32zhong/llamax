	#--hostfile configs/hostfile \
deepspeed \
	--num_gpus 8 \
	--num_nodes 1 \
	--master_port 8921 \
	train.py \
	--model_name_or_path ~/llama-models/7B-hgf/ \
	--data_path ../data/alpaca_data.json \
	--output_dir ./output/ \
	--num_train_epochs 3 \
	--model_max_length 2048 \
	--per_device_train_batch_size 64 \
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
	--fp16 True
