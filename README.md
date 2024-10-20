# MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models

We introduce MMed-RAG, a powerful multimodal RAG system that boosts the factuality of Medical Vision-Language Models (Med-LVLMs) by up to 43.8%! 🩺 [[Paper](https://arxiv.org/abs/2410.13085)] [[X(Twitter)](https://x.com/HuaxiuYaoML/status/1847097594641584574)]

## 🚀 News
- [10/18/2024] The manuscript can be found on [arXiv](https://arxiv.org/abs/2410.13085).

## 💡 Overview
MMed-RAG enhances alignment across medical domains like radiology, pathology, and ophthalmology with a domain-aware retrieval mechanism. And it tackles three key challenges in alignment of multimodal RAG:

1️⃣ Direct Copy Homework from Others❌ Think it by Self ✅
MMed-RAG helps Med-LVLMs avoid blindly copying external information by encouraging the model to rely on its own visual reasoning when solving complex problems.

2️⃣ Cannot Solve Problems by Self❌ Learn How to Copy ✅
When Med-LVLMs are unsure, MMed-RAG teaches the model to intelligently use retrieved knowledge, pulling in the right information at the right time, boosting accuracy, and reducing errors.

3️⃣ Copied Homework is Wrong❌ Avoid Interference from Incorrect Homework ✅
MMed-RAG prevents models from being misled by incorrect retrievals, reducing the risk of generating inaccurate medical diagnoses.

## 📦 Requirements
1. Clone this repository and navigate to MMed-RAG folder
```bash
git clone https://github.com/richard-peng-xia/MMed-RAG.git
cd MMed-RAG
```

2. Install Package: Create conda environment

```Shell
conda create -n MMed-RAG python=3.10 -y
conda activate MMed-RAG
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install trl
```

3. Download the required model checkpoints [LLaVA-Med-1.5](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b) from huggingface.

## 📅 Schedule

- [ ] Release the data (VQA and report generation tasks)

- [ ] Release the training code

## 📚Citation

```bibtex
@article{xia2024mmedrag,
  title={MMed-RAG: Versatile Multimodal RAG System for Medical Vision Language Models},
  author={Xia, Peng and Zhu, Kangyu and Li, Haoran and Wang, Tianze and Shi, Weijia and Wang, Sheng and Zhang, Linjun and Zou, James and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2410.13085},
  year={2024}
}
```

## 🙏Acknowledgement
We use code from [LLaVA-Med](https://github.com/microsoft/LLaVA-Med), [RULE](https://github.com/richard-peng-xia/RULE), [CARES](https://github.com/richard-peng-xia/CARES). We thank the authors for releasing their code.

<!-- 
## Clip Finetune
```
bash ./scripts/retrieve_clip_VQA.sh
```
## DPO training

```
bash ./scripts/train_dpo_2stages_VQA.sh
```

## Inference
```

```
--> 



