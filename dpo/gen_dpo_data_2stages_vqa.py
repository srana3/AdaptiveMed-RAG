import json
import argparse


def parse_answer(answer_text):
    if "." in answer_text:
        answer_text = answer_text.split(".")[0]
    answer_text = answer_text.replace(",", "")
    words = answer_text.split()
    if "No" in words or "no" in words or "not" in words:
        return 0
    else:
        return 1


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

    for idx, (clean_item, noised_item) in enumerate(zip(clean_data, noised_data)):
        # if select:
        #     if any(phrase in ans_item['answer'] for phrase in selected_phrases) or \
        #         any(phrase in ref_item['answer'] for phrase in selected_phrases):
        #         continue
        
        new_item = {
            "id": clean_item.get("question_id", ""),
            "image": clean_item.get("image", ""),
            "conversations": [],
            "rejected_conversations": [],
            "rejected_noised":0,
        }
        # question_with_report_prompt=ref_item["question"]
        question=clean_item["prompt"]
        if clean_item["correctness"] == 1 and noised_item["correctness"] == 0:
            new_item["conversations"] = [
                {"from": "human", "value": '<image>\n'+question},
                {"from": "gpt", "value": clean_item["answer"]},
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
    
    # to guarantee attention on image when retrieving report
    for idx, (clean_item_with_report, noised_item_with_report) in enumerate(zip(clean_data_with_report, noised_data_with_report)):
        # if select:
        #     if any(phrase in ans_item['answer'] for phrase in selected_phrases) or \
        #         any(phrase in ref_item['answer'] for phrase in selected_phrases):
        #         continue
        
        new_item = {
            "id": clean_item_with_report.get("question_id", ""),
            "image": clean_item_with_report.get("image", ""),
            "conversations": [],
            "rejected_conversations": [],
            "rejected_noised":0,
        }
        # question_with_report_prompt=ref_item["question"]
        question_with_report_prompt=clean_item_with_report["prompt"]
        if clean_item_with_report["correctness"] == 1 and noised_item_with_report["correctness"] == 1:
            # print(clean_item_with_report["answer"])
            new_item["conversations"] = [
                {"from": "human", "value": '<image>\n'+question_with_report_prompt},
                {"from": "gpt", "value": clean_item_with_report["answer"]},
            ]
            new_item["rejected_conversations"] = [
                {"from": "human", "value": '<image>\n'+question_with_report_prompt},
                {"from": "gpt", "value": noised_item_with_report["answer"]},
            ]
            new_item['rejected_noised']=1
            
            stage2a_data.append(new_item)
            
        
        else:
            continue
    # print(new_data)
    # dpo_data.extend(stage1_data)
    
    
    for idx, (clean_item, clean_item_with_report) in enumerate(zip(clean_data, clean_data_with_report)):
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
        if clean_item["correctness"] == 1 and clean_item_with_report["correctness"] == 0:
            new_item["conversations"] = [
                {"from": "human", "value": '<image>\n'+question_with_report_prompt}, #question?
                {"from": "gpt", "value": clean_item["answer"]},
            ]
            new_item["rejected_conversations"] = [
                {"from": "human", "value": '<image>\n'+question_with_report_prompt},
                {"from": "gpt", "value": clean_item_with_report["answer"]},
            ]
            
            stage2b_data.append(new_item)
        if clean_item["correctness"] == 0 and clean_item_with_report["correctness"] == 1:
            new_item["conversations"] = [
                {"from": "human", "value": '<image>\n'+question_with_report_prompt}, #question?
                {"from": "gpt", "value": clean_item_with_report["answer"]},
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
    print('actual length of stage2a_data:',len(stage2a_data))
    dpo_data.extend(stage2b_data)
    print('actual length of stage2b_data:',len(stage2b_data))
    dpo_data.extend(stage2c_data)
    print('actual length of stage2c_data:',len(stage2c_data))
    # print(dpo_data)
        

    return dpo_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--select", action="store_true", help="是否应用筛选条件来过滤特定答案。"
    )
    args = parser.parse_args()

    # 文件路径（请根据您的环境修改）
    # ref_file_path = "/path/to/answer-file_reference.jsonl"
    # ans_file_path = "/path/to/answer-file.jsonl"
    
    clean_file_path='/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/inference/val/answer-file_harvard_val_vqa.jsonl'
    noised_file_path='/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/inference/val/answer-file_harvard_val_noised_vqa.jsonl'
    clean_file_with_report_path='/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/inference/val/answer-file_harvard_val_withReport_fixedK-1_vqa.jsonl'
    noised_file_with_report_path='/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/inference/val/answer-file_harvard_val_withReport_fixedK-1_noised_vqa.jsonl'

    # 读取输入文件
    with open(clean_file_path, "r") as clean_file, open(noised_file_path, "r") as noised_file, open(clean_file_with_report_path, "r") as clean_file_with_report, open(noised_file_with_report_path, "r") as noised_file_with_report:
        clean_data = [json.loads(line) for line in clean_file]
        noised_data = [json.loads(line) for line in noised_file]
        clean_data_with_report = [json.loads(line) for line in clean_file_with_report]
        noised_data_with_report = [json.loads(line) for line in noised_file_with_report]
        
        

    assert len(clean_data)==len(noised_data)==len(clean_data_with_report)==len(noised_data_with_report), "The length of the four files should be the same."
    for i in range(len(clean_data)):
        clean_data[i]["correctness"] = parse_answer(clean_data[i]["answer"])==parse_answer(clean_data[i]["gt_answer"])
        noised_data[i]["correctness"] = parse_answer(noised_data[i]["answer"])==parse_answer(noised_data[i]["gt_answer"])
        clean_data_with_report[i]["correctness"] = parse_answer(clean_data_with_report[i]["answer"])==parse_answer(clean_data_with_report[i]["gt_answer"])
        noised_data_with_report[i]["correctness"] = parse_answer(noised_data_with_report[i]["answer"])==parse_answer(noised_data_with_report[i]["gt_answer"])
    
    # 准备数据字典
    data_dict = {
        "clean_data": clean_data,
        "noised_data": noised_data,
        "clean_data_with_report": clean_data_with_report,
        "noised_data_with_report": noised_data_with_report,
                }

    # 生成 DPO 数据
    dpo_data_stage1 = generate_dpo_data_stage1(data_dict, conv_type="qrefVqa", select=args.select)
    dpo_data_stage2 = generate_dpo_data_stage2(data_dict, conv_type="qrefVqa", select=args.select)
    
    # 输出文件路径
    if not args.select:
        output_file_json_stage1 = "/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/dpo/annotations/harvard_qrefVqa_stage1_balanced_abc.json"
        output_file_json_stage2 = "/home/wenhao/Project/intern/kangyu/annotations+scripts+outputs/annotations+scripts/med-dpo/harvard/dpo/annotations/harvard_qrefVqa_stage2_balanced_abc.json"
        
    else:
        
        # output_file_jsonl = "/path/to/dpo_data_vqa_selected.jsonl"
        output_file_json = "/path/to/dpo_data_vqa.json"

    # 写入输出文件
    # with open(output_file_jsonl, "w") as output_file:
    #     for item in dpo_data:
    #         output_file.write(json.dumps(item) + "\n")
    # print(f"已将 {len(dpo_data)} 条数据写入 {output_file_jsonl}")

    with open(output_file_json_stage1, "w") as output_file_stage1:
        json.dump(dpo_data_stage1, output_file_stage1)
        print(f"已将 {len(dpo_data_stage1)} 条数据写入 {output_file_json_stage1}")
    with open(output_file_json_stage2, "w") as output_file_stage2:
        json.dump(dpo_data_stage2, output_file_stage2)
        print(f"已将 {len(dpo_data_stage2)} 条数据写入 {output_file_json_stage2}")

    
