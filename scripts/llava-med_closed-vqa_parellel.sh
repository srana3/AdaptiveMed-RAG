
export OMP_NUM_THREADS=64
CUDA="2,3"
GPU=2
model_base="/home/wenhao/Project/intern/xiapeng/rein/LLaVA-Med/checkpoint/llava-med-v1.5-mistral-7b"
image_folder=/home/wenhao/Datasets/med/Harvard/images
data_type=test
fixedK=1
noised_image=True


noised_image_lower=$(echo "$noised_image" | tr '[:upper:]' '[:lower:]')
if [ "$noised_image_lower" = "true" ]; then
    noised_image_option="--noised_image"
    noised_suffix="noised_"
else
    noised_image_option=""
    noised_suffix=""
fi


# question_file=/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/gpt_vqa/vqa/harvard_${data_type}_vqa.jsonl
# answer_file="/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/inference/${data_type}/answer-file_harvard_${data_type}_${noised_suffix}vqa.jsonl"


question_file=/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/retrieve/vqa/${data_type}/harvard_vqa_${data_type}_withReport_fixedK-${fixedK}.jsonl
answer_file="/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/inference/${data_type}/answer-file_harvard_${data_type}_withReport_fixedK-${fixedK}_${noised_suffix}vqa.jsonl"

echo "question file: ${question_file}"
echo "answer_file: ${answer_file}"

# cd /home/wenhao/Project/intern/xiapeng/med-dpo/annotation/harvard/anno_scripts/open_ended/inference/parellel
CUDA_VISIBLE_DEVICES=$CUDA torchrun --nproc_per_node=$GPU --master_port=$RANDOM /home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/inference/llava-med-1.5_closed-vqa_harvard_parellel.py \
    --model-path $model_base \
    --question-file $question_file \
    --image-folder $image_folder \
    --answers-file $answer_file \
    $noised_image_option





