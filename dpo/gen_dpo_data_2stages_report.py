import json
import argparse




def generate_dpo_data_stage1(data_dict, conv_type="qrefVqa", select=False):
    clean_data = data_dict["clean_data"]
    noised_data = data_dict["noised_data"]

    # ref_list = [parse_answer(item['answer']) for item in ref_data]
    # ans_list = [parse_answer(item['answer']) for item in ans_data]
    
    # 初始化新数据列表
    
    # 生成 DPO 数据
    new_data=[]
    dpo_data = []
    # selected_phrases = ['I cannot provide', 'As an AI']

    avg_bleu_score = (sum([item["bleu_score"] for item in clean_data])+sum(item["bleu_score"] for item in noised_data)) / (len(clean_data)+len(noised_data))
    for idx, (clean_item, noised_item) in enumerate(zip(clean_data, noised_data)):
        # if select:
        #     if any(phrase in ans_item['answer'] for phrase in selected_phrases) or \
        #         any(phrase in ref_item['answer'] for phrase in selected_phrases):
        #         continue
        clean_item["correctness"] = clean_item["bleu_score"] >= avg_bleu_score
        noised_item["correctness"] = noised_item["bleu_score"] >= avg_bleu_score
        new_item = {
            "id": clean_item.get("question_id", ""),
            "image": clean_item.get("image", ""),
            "conversations": [],
            "rejected_conversations": [],
            "rejected_noised":0,
        }
        # question_with_report_prompt=ref_item["question"]
        question=clean_item["prompt"]
        gt_report=clean_item["gt_report"]
        if clean_item["correctness"] == 1 and noised_item["correctness"] == 0:
            new_item["conversations"] = [
                {"from": "human", "value": '<image>\n'+question},
                {"from": "gpt", "value": gt_report},
            ]
            new_item["rejected_conversations"] = [
                {"from": "human", "value": '<image>\n'+question},
                {"from": "gpt", "value": noised_item["answer"]},
            ]
            new_item['rejected_noised']=1
            
            
            
            new_data.append(new_item)
        
        else:
            continue
    dpo_data.extend(new_data)
    print('Actual number of stage1 data: ', len(new_data))
        

    return dpo_data
def generate_dpo_data_stage2(data_dict, conv_type="qrefVqa", select=False):
    clean_data = data_dict["clean_data"]
    noised_data = data_dict["noised_data"]
    clean_data_with_report = data_dict["clean_data_with_report"]
    noised_data_with_report = data_dict["noised_data_with_report"]
    

    # ref_list = [parse_answer(item['answer']) for item in ref_data]
    # ans_list = [parse_answer(item['answer']) for item in ans_data]
    
    # 初始化新数据列表
    
    # 生成 DPO 数据
    dpo_data = []
    # selected_phrases = ['I cannot provide', 'As an AI']
    stage2a_data=[]
    stage2b_data=[]
    stage2c_data=[]
    
    avg_bleu_score = (sum([item["bleu_score"] for item in clean_data_with_report])+sum(item["bleu_score"] for item in noised_data_with_report)) / (len(clean_data_with_report)+len(noised_data_with_report))
    # to guarantee attention on image when retrieving report
    for idx, (clean_item_with_report, noised_item_with_report) in enumerate(zip(clean_data_with_report, noised_data_with_report)):
        # if select:
        #     if any(phrase in ans_item['answer'] for phrase in selected_phrases) or \
        #         any(phrase in ref_item['answer'] for phrase in selected_phrases):
        #         continue
        clean_item_with_report["correctness"] = clean_item_with_report["bleu_score"] >= avg_bleu_score
        noised_item_with_report["correctness"] = noised_item_with_report["bleu_score"] >= avg_bleu_score
        new_item = {
            "id": clean_item_with_report.get("question_id", ""),
            "image": clean_item_with_report.get("image", ""),
            "conversations": [],
            "rejected_conversations": [],
            "rejected_noised":0,
        }
        # question_with_report_prompt=ref_item["question"]
        question_with_report_prompt=clean_item_with_report["prompt"]
        gt_report=clean_item_with_report["gt_report"]
        if clean_item_with_report["correctness"] == 1 and noised_item_with_report["correctness"] == 1:
            # print(clean_item_with_report["answer"])
            new_item["conversations"] = [
                {"from": "human", "value": '<image>\n'+question_with_report_prompt},
                {"from": "gpt", "value": gt_report},
            ]
            new_item["rejected_conversations"] = [
                {"from": "human", "value": '<image>\n'+question_with_report_prompt},
                {"from": "gpt", "value": noised_item_with_report["answer"]},
            ]
            new_item['rejected_noised']=1
            
            stage2a_data.append(new_item)
        
            
        
        else:
            continue
    print(f'Number of stage2a_data: {len(stage2a_data)}')
    # print(new_data)
    # dpo_data.extend(stage1_data)
    
    avg_bleu_score = (sum([item["bleu_score"] for item in clean_data])+sum(item["bleu_score"] for item in clean_data_with_report)) / (len(clean_data)+len(clean_data_with_report))
    for idx, (clean_item, clean_item_with_report) in enumerate(zip(clean_data, clean_data_with_report)):
        
        clean_item["correctness"] = clean_item["bleu_score"] >= avg_bleu_score
        clean_item_with_report["correctness"] = clean_item_with_report["bleu_score"] >= avg_bleu_score
        new_item = {
            "id": clean_item.get("image_id", ""),
            "image": clean_item.get("image", ""),
            "conversations": [],
            "rejected_conversations": [],
            "rejected_noised":0,
        }
        # question_with_report_prompt=ref_item["question"]
        question=clean_item["prompt"]
        question_with_report_prompt=clean_item_with_report["prompt"]
        gt_report=clean_item["gt_report"]
        if clean_item["correctness"] == 1 and clean_item_with_report["correctness"] == 0:
            new_item["conversations"] = [
                {"from": "human", "value": '<image>\n'+question_with_report_prompt}, #question?
                {"from": "gpt", "value": gt_report},
            ]
            new_item["rejected_conversations"] = [
                {"from": "human", "value": '<image>\n'+question_with_report_prompt},
                {"from": "gpt", "value": clean_item_with_report["answer"]},
            ]
            
            stage2b_data.append(new_item)
        if clean_item["correctness"] == 0 and clean_item_with_report["correctness"] == 1:
            new_item["conversations"] = [
                {"from": "human", "value": '<image>\n'+question_with_report_prompt}, #question?
                {"from": "gpt", "value": gt_report},
            ]
            new_item["rejected_conversations"] = [
                {"from": "human", "value": '<image>\n'+question_with_report_prompt},
                {"from": "gpt", "value": clean_item["answer"]},
            ]
            stage2c_data.append(new_item)
            
        
        else:
            continue
        
    import random
    random.seed(42)
    stage2a_data=(random.sample(stage2a_data, int((len(stage2b_data)+len(stage2c_data))*0.5)))
    dpo_data.extend(stage2a_data)
    dpo_data.extend(stage2b_data)
    dpo_data.extend(stage2c_data)
    
    print('Actual number of stage2a_data: ', len(stage2a_data))
    print('Actual number of stage2b_data: ', len(stage2b_data))
    print('Actual number of stage2c_data: ', len(stage2c_data))
    # print(dpo_data)
        

    return dpo_data
import json
import argparse
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu_score(answer_text, reference_text):
    reference = [reference_text.split()]
    candidate = answer_text.split()
    smoothing = SmoothingFunction().method7
    return sentence_bleu(reference, candidate, smoothing_function=smoothing)

def parse_answer(answer_text):
    if "." in answer_text:
        answer_text = answer_text.split(".")[0]
    answer_text = answer_text.replace(",", "")
    return answer_text

def get_bleu_score_inDict(data):
    
    for i,item in enumerate(data):
        answer = parse_answer(item["answer"])
        gt_answer = parse_answer(item["gt_report"])
        bleu_score = calculate_bleu_score(answer, gt_answer)
        data[i]["bleu_score"] = bleu_score
    return data
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--select", action="store_true", help="是否应用筛选条件来过滤特定答案。"
    )
    args = parser.parse_args()

    clean_file_path = '/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/inference/val_report/answer-file_harvard_val_report.jsonl'
    noised_file_path = '/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/inference/val_report/answer-file_harvard_val_noised_report.jsonl'
    clean_file_with_report_path = '/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/inference/val_report/answer-file_harvard_val_withReport_fixedK-1_report.jsonl'
    noised_file_with_report_path = '/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/inference/val_report/answer-file_harvard_val_withReport_fixedK-1_noised_report.jsonl'

    with open(clean_file_path, "r") as clean_file, open(noised_file_path, "r") as noised_file, open(clean_file_with_report_path, "r") as clean_file_with_report, open(noised_file_with_report_path, "r") as noised_file_with_report:
        clean_data = [json.loads(line) for line in clean_file]
        noised_data = [json.loads(line) for line in noised_file]
        clean_data_with_report = [json.loads(line) for line in clean_file_with_report]
        noised_data_with_report = [json.loads(line) for line in noised_file_with_report]

    assert len(clean_data) == len(noised_data) == len(clean_data_with_report) == len(noised_data_with_report), "The length of the four files should be the same."


    clean_data=get_bleu_score_inDict(clean_data)
    noised_data=get_bleu_score_inDict(noised_data)
    clean_data_with_report=get_bleu_score_inDict(clean_data_with_report)
    noised_data_with_report=get_bleu_score_inDict(noised_data_with_report)


    data_dict = {
        "clean_data": clean_data,
        "noised_data": noised_data,
        "clean_data_with_report": clean_data_with_report,
        "noised_data_with_report": noised_data_with_report,
    }

    dpo_data_stage1 = generate_dpo_data_stage1(data_dict, conv_type="caption", select=args.select)
    dpo_data_stage2 = generate_dpo_data_stage2(data_dict, conv_type="caption", select=args.select)

    if not args.select:
        output_file_json_stage1 = "/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/dpo/annotations/report/harvard_caption_stage1_balanced_abc.json"
        output_file_json_stage2 = "/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/dpo/annotations/report/harvard_caption_stage2_balanced_abc.json"
    else:
        output_file_json = "/path/to/dpo_data_vqa.json"

    with open(output_file_json_stage1, "w") as output_file_stage1:
        json.dump(dpo_data_stage1, output_file_stage1)
        print(f"已将 {len(dpo_data_stage1)} 条数据写入 {output_file_json_stage1}")

    with open(output_file_json_stage2, "w") as output_file_stage2:
        json.dump(dpo_data_stage2, output_file_stage2)
        print(f"已将 {len(dpo_data_stage2)} 条数据写入 {output_file_json_stage2}")

    
