export CUDA_VISIBLE_DEVICES='0,1,2,3'
CUDA='0,1,2,3'
image_folder=/home/wenhao/Datasets/med/Harvard/images

# 使用数组来处理多个数据路径
data_paths=(
    "/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/dpo/annotations/report/harvard_caption_stage1_balanced_abc.json"
)

for data_path in "${data_paths[@]}"; do
    # 根据文件名设置输出路径
    if [[ "$data_path" == *"caption"* ]]; then
        output_path='/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/outputs/med-dpo/harvard/dpo_checkpoints/harvard_llava-med-lora-caption_stage-1_balanced_abc_report'
    elif [[ "$data_path" == *"qrefVqa"* ]]; then
        output_path='/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/outputs/med-dpo/harvard/dpo_checkpoints/harvard_llava-med-lora-qrefVqa_stage-1_balanced_abc_report'
    elif [[ "$data_path" == *"vqa-merged"* ]]; then
        output_path='/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/outputs/med-dpo/harvard/dpo_checkpoints/harvard_llava-med-lora-vqa-merged_stage-1_balanced_abc_report'
    elif [[ "$data_path" == *"vqa"* ]]; then
        output_path='/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/outputs/med-dpo/harvard/dpo_checkpoints/harvard_llava-med-lora-vqa_stage-1_balanced_abc_report'
    else
        echo "Unknown data path type: $data_path"
        continue
    fi

    cd /home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/POVID || exit
    deepspeed --include localhost:$CUDA /home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/dpo/finetune/train_dpo_2stages.py \
        --model_name_or_path /home/wenhao/Project/intern/xiapeng/rein/LLaVA-Med/checkpoint/llava-med-v1.5-mistral-7b \
        --deepspeed ./scripts/zero3.json \
        --version v1 \
        --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
        --data_path $data_path \
        --image_folder $image_folder  \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir $output_path \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1\
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --learning_rate 1e-7 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --report_to wandb \
        --tf32 True \
        --model_max_length 1024 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \

done



for data_path in "${data_paths[@]}"; do
    # 根据文件名设置输出路径
    if [[ "$data_path" == *"caption"* ]]; then
        output_path='/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/outputs/med-dpo/harvard/dpo_checkpoints/harvard_llava-med-lora-caption_stage-2_balanced_abc_report'
    elif [[ "$data_path" == *"qrefVqa"* ]]; then
        output_path='/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/outputs/med-dpo/harvard/dpo_checkpoints/harvard_llava-med-lora-qrefVqa_stage-2_balanced_abc_report'
    elif [[ "$data_path" == *"vqa-merged"* ]]; then
        output_path='/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/outputs/med-dpo/harvard/dpo_checkpoints/harvard_llava-med-lora-vqa-merged_stage-2_balanced_abc_report'
    elif [[ "$data_path" == *"vqa"* ]]; then
        output_path='/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/outputs/med-dpo/harvard/dpo_checkpoints/harvard_llava-med-lora-vqa_stage-2_balanced_abc_report'
    else
        echo "Unknown data path type: $data_path"
        continue
    fi
    stage1_lora_checkpoint_path="${output_path/stage-2/stage-1}"
    data_path_stage2="${data_path/stage1/stage2}"
    echo "Modified data path (stage2): $data_path_stage2"

    cd /home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/POVID || exit
    deepspeed --include localhost:$CUDA /home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/dpo/finetune/train_dpo_2stages.py \
        --model_name_or_path /home/wenhao/Project/intern/xiapeng/rein/LLaVA-Med/checkpoint/llava-med-v1.5-mistral-7b \
        --deepspeed ./scripts/zero3.json \
        --version v1 \
        --data_path $data_path_stage2 \
        --image_folder $image_folder  \
        --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
        --lora_checkpoint_path $stage1_lora_checkpoint_path \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir $output_path \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --learning_rate 1e-7 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --report_to wandb \
        --tf32 True \
        --model_max_length 1024 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \

done

        