
export OMP_NUM_THREADS=64
CUDA="0,1,2,3"
GPU=4
model_base="/home/wenhao/Project/intern/xiapeng/rein/LLaVA-Med/checkpoint/llava-med-v1.5-mistral-7b"
model_path="/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/outputs/med-dpo/harvard/dpo_checkpoints/harvard_llava-med-lora-qrefVqa_stage-2a-change"
model_path_basename=$(basename $model_path)
image_folder=/home/wenhao/Datasets/med/Harvard/images
data_type=test
fixedK=1
# noised_image=True


noised_image_lower=$(echo "$noised_image" | tr '[:upper:]' '[:lower:]')
if [ "$noised_image_lower" = "true" ]; then
    noised_image_option="--noised_image"
    noised_suffix="noised_"
else
    noised_image_option=""
    noised_suffix=""
fi


# question_file=/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/gpt_vqa/vqa/harvard_${data_type}_vqa.jsonl
# answer_file="/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/inference/${data_type}/answer-file_${model_path_basename}_${data_type}_${noised_suffix}vqa.jsonl"


question_file=/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/retrieve/vqa/${data_type}/harvard_vqa_${data_type}_withReport_fixedK-${fixedK}.jsonl
answer_file="/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/inference/${data_type}/answer-file_${model_path_basename}_${data_type}_${noised_suffix}fixedK-${fixedK}_vqa.jsonl"

echo "question file: ${question_file}"
echo "answer_file: ${answer_file}"

# cd /home/wenhao/Project/intern/xiapeng/med-dpo/annotation/harvard/anno_scripts/open_ended/inference/parellel
CUDA_VISIBLE_DEVICES=$CUDA torchrun --nproc_per_node=$GPU --master_port=$RANDOM /home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/inference/llava-med-1.5_closed-vqa_harvard_parellel.py \
    --model-base $model_base \
    --model-path $model_path \
    --question-file $question_file \
    --image-folder $image_folder \
    --answers-file $answer_file \
    $noised_image_option





