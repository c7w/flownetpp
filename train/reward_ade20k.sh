export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export CONTROLNET_DIR="work_dirs/finetune/Captioned_ADE20K/ft_controlnet_sd15_seg_res512_bs256_lr1e-5_warmup100_iter5k_fp16/checkpoint-5000/controlnet"
export REWARDMODEL_DIR="mmseg::upernet/upernet_r50_4xb4-160k_ade20k-512x512.py"
export OUTPUT_DIR="work_dirs/reward_model/Captioned_ADE20K/1reward_ft_controlnet_sd15_seg_res512_bs256_lr1e-5_warmup100_scale-0.5_iter5k_fp16_train0-1k_reward0-200_UperNet-R50"

export HF_ENDPOINT=https://hf-mirror.com

# export WANDB_DISABLED=1
export WANDB_API_KEY=48cd7beeac132a453ceb4db996fa8cb69bceb062
export WANDB_ENTITY=c7w
export CUDA_VISIBLE_DEVICES=6



# conda activate /data/Tsinghua/Share/miniconda3/envs/cnpp
# cd /data/Tsinghua/chihh/ControlNet_Plus_Plus 

# Download our fine-tuned weights
# You can also train a new one with command `bash train/finetune_ade20k.sh`
python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='limingcv/reward_controlnet', filename='diffusion_pytorch_model.safetensors', subfolder=f'${CONTROLNET_DIR}', local_dir='./', local_dir_use_symlinks=False); hf_hub_download(repo_id='limingcv/reward_controlnet', filename='config.json', subfolder=f'${CONTROLNET_DIR}', local_dir='./', local_dir_use_symlinks=False)"

# --dataset_name="/Node10_nvme/Captioned_ADE20K/data" \
# reward fine-tuning
accelerate launch --config_file "train/debug.yml" \
 --main_process_port=23156 train/reward_control.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --controlnet_model_name_or_path=$CONTROLNET_DIR \
 --reward_model_name_or_path=$REWARDMODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --task_name="segmentation" \
 --dataset_name="/Node10_nvme/Captioned_ADE20K/data" \
 --caption_column="prompt" \
 --conditioning_image_column="control_seg" \
 --cache_dir="data/huggingface_datasets" \
 --resolution=512 \
 --train_batch_size=8 \
 --gradient_accumulation_steps=4 \
 --learning_rate=1e-5 \
 --mixed_precision="fp16" \
 --gradient_checkpointing \
 --dataloader_num_workers=16 \
 --max_train_steps=50000 \
 --lr_scheduler="constant_with_warmup" \
 --lr_warmup_steps=200 \
 --checkpointing_steps=10000 \
 --grad_scale=0.5 \
 --use_ema \
 --validation_steps=500 \
 --timestep_sampling_start=0 \
 --timestep_sampling_end=1000 \
 --min_timestep_rewarding=0 \
 --max_timestep_rewarding=200 \
 --rflow
