
# config_type=verbConf

# clip_threshold=1.5
fixed_k=1
data_type=test
retrieve_modality=fundus  #fundus,pathology,radiology
retrive_mode=report #report, vqa

if [ "$data_type" = "val" ]; then
    eval_type="train"
    num=1000
elif [ "$data_type" = "test" ]; then
    eval_type="test"
    num=713
fi
output_root=/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/retrieve/report/${data_type}/


if [[ -n "$clip_threshold" && -z "$fixed_k" ]]; then
    output_path="${output_root}/${retrieve_modality}_${retrieve_mode}_${data_type}_withReport_ConfTopk_clipThreshold-${clip_threshold}.jsonl"
elif [[ -n "$clip_threshold" && -n "$fixed_k" ]]; then
    output_path="${output_root}/${retrieve_modality}_${retrieve_mode}_${data_type}_withReport_fixedK-${fixed_k}_clipThreshold-${clip_threshold}.jsonl"
elif [[ -z "$clip_threshold" && -n "$fixed_k" ]]; then
    output_path="${output_root}/${retrieve_modality}_${retrieve_mode}_${data_type}_withReport_fixedK-${fixed_k}.jsonl"
elif [[ -z "$clip_threshold" && -z "$fixed_k" ]]; then
    output_path="${output_root}/${retrieve_modality}_${retrieve_mode}_${data_type}_withReport_ConfTopk.jsonl"
fi

cd ./retrieve || exit
echo "Output path is: $output_path"


cmd="CUDA_VISIBLE_DEVICES=2 python retrieve_clip_harvard.py \
    --img_root /home/wenhao/Datasets/med/Harvard/images \
    --train_json /home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/data_annotations/harvard_train_7000.json \
    --eval_json /home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/data_annotations/harvard_${data_type}_${num}.json \
    --model_name_or_path hf-hub:thaottn/OpenCLIP-resnet50-CC12M \
    --checkpoint_path xxxx.pt \
    --output_path $output_path \
    --eval_type $eval_type "


if [[ -n "$clip_threshold" ]]; then
    cmd+=" --clip_threshold $clip_threshold"
fi

if [[ -n "$fixed_k" ]]; then
    cmd+=" --fixed_k $fixed_k"
fi

# 运行命令
eval $cmd