# HQRS-IT-210K-and-HQRS-CLIP

This is the repository of th paper "Enhancing Remote Sensing Vision-Language Models Through MLLM and LLM-Based High-Quality Image-Text Dataset Generation".

<img width="1534" height="1228" alt="image" src="https://github.com/user-attachments/assets/3db63e1b-8b6f-4b19-ba1d-db86815ab943" />

<img width="1357" height="853" alt="image" src="https://github.com/user-attachments/assets/a10f6252-040f-49ce-9345-6974a8bfb320" />


<img width="1353" height="1189" alt="image" src="https://github.com/user-attachments/assets/d0e8bc2e-945b-4a97-b6dc-63a875c998b0" />

<img width="1377" height="588" alt="image" src="https://github.com/user-attachments/assets/7e2821d5-619c-42fa-a68d-1f52c3eae524" />

Our code, dataset and models have been released here:


## 1. paper link
Our paper link on arxiv:https://arxiv.org/pdf/2507.16716

## 2.models

Our models are based on **openCLIP**, so please install the necessary dependencies for openCLIP before using the model. You can find the instructions here: [openCLIP GitHub repository](https://github.com/mlfoundations/open_clip).

Additionally, if you want to use our **cross-modal retrieval testing script** (`retrieve_test.py`) for benchmarking or reproducing state-of-the-art (SOTA) results, please install the required dependencies mentioned in the script file. Specifically, you need to install `clip_benchmark` via:

```bash
pip install clip_benchmark
```

You will also need to download the following datasets for testing:

- **RSITMD**: [RSITMD GitHub repository](https://github.com/xiaoyuan1996/AMFMN/blob/master/RSITMD/README.md)
- **RSICD**: [RSICD GitHub repository](https://github.com/201528014227051/RSICD_optimal)
- **UCMCaption**: [UCMCaption on AIStudio](https://aistudio.baidu.com/datasetdetail/90740)



The checkpoints of our HQRS-CLIP:
   
Baidu Netdisk: https://pan.baidu.com/s/1bYPDArqxdxH-4NbytzsGKA?pwd=62v7 提取码: 62v7

The checkpoints of our HQRS-CLIP-ret3（Fine-tuned on ret3 datasets）:

Baidu Netdisk: https://pan.baidu.com/s/16UCDtu5P3iTHRguxZjXfQg?pwd=6r6i 提取码: 6r6i

## 3.HQRS-IT-210K Dataset：

Images：

Baidu Netdisk: https://pan.baidu.com/s/1BW-33ilETvS-RXmJI6KmYQ?pwd=iug7 提取码: iug7

Captions:

Baidu Netdisk: https://pan.baidu.com/s/1VjVZtJuEs4ISVSS1ooGi_w?pwd=a393 提取码: a393

## 4.Code of retrieval and selo test.

## 5.Training Code
Our model is trained by **openCLIP**, so the training code is here: [openCLIP GitHub repository](https://github.com/mlfoundations/open_clip).
