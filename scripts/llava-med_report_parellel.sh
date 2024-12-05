
export OMP_NUM_THREADS=64
CUDA="2,3"
GPU=2
model_base="/home/wenhao/Project/intern/xiapeng/rein/LLaVA-Med/checkpoint/llava-med-v1.5-mistral-7b"
image_folder=/home/wenhao/Datasets/med/Harvard/images
data_type=test
# fixedK=1
# noised_image=False

store_folder=${data_type}_report
if [ "$data_type" = "test" ]; then
    num=713
else
    num=1000
fi

noised_image_lower=$(echo "$noised_image" | tr '[:upper:]' '[:lower:]')
if [ "$noised_image_lower" = "true" ]; then
    noised_image_option="--noised_image"
    noised_suffix="noised_"
else
    noised_image_option=""
    noised_suffix=""
fi


# question_file=/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/data_annotations/harvard_${data_type}_${num}.jsonl
# answer_file="/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/inference/${store_folder}/answer-file_harvard_${data_type}_${noised_suffix}report.jsonl"


question_file=/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/retrieve/report/${data_type}/harvard_report_${data_type}_withReport_fixedK-${fixedK}.jsonl
answer_file=/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/inference/${store_folder}/answer-file_harvard_${data_type}_withReport_fixedK-${fixedK}_${noised_suffix}report.jsonl

echo "question file: ${question_file}"
echo "answer_file: ${answer_file}"

# cd /home/wenhao/Project/intern/xiapeng/med-dpo/annotation/harvard/anno_scripts/open_ended/inference/parellel
CUDA_VISIBLE_DEVICES=$CUDA torchrun --nproc_per_node=$GPU --master_port=$RANDOM /home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/inference/llava-med-1.5_report_harvard_parellel.py \
    --model-path $model_base \
    --question-file $question_file \
    --image-folder $image_folder \
    --answers-file $answer_file \
    --temperature 0.2 \
    $noised_image_option





