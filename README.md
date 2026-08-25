## HQRS-IT-210K-and-HQRS-CLIP

This is the official repository of the paper "MpGI: Multi-Perspective Generation and Integration for High-Quality Remote Sensing Image-Text Datasets", which has been accepted by IEEE Transactions on Geoscience and Remote Sensing (TGRS) and is now available in Early Access ( https://ieeexplore.ieee.org/abstract/document/11646842 ).

Paper Update: Our paper has undergone substantial revision during the peer-review process, including changes to the title, abstract, and manuscript content. The current arXiv manuscript corresponds to an earlier version of this work and will be updated with the accepted TGRS version.

<img width="843" height="1161" alt="7a2ecd79-b3ea-427b-8bcb-4a1e431171d7" src="https://github.com/user-attachments/assets/0c9cf19b-210a-49cd-8f48-50b3914f311b" />

<img width="987" height="512" alt="image" src="https://github.com/user-attachments/assets/e6aba6c8-e2fb-415f-b41a-fd9d15c376c5" />

<img width="1004" height="729" alt="image" src="https://github.com/user-attachments/assets/3c22e1f2-5826-4038-9bb8-90b0a05040c2" />

<img width="987" height="687" alt="image" src="https://github.com/user-attachments/assets/ab3b6ece-5c02-49dd-8ecd-4aac1cd924d4" />


Our code, dataset, and models have been released here:

## 1. Paper

Title: MpGI: Multi-Perspective Generation and Integration for High-Quality Remote Sensing Image-Text Datasets

Published in: IEEE Transactions on Geoscience and Remote Sensing (Early Access)
https://ieeexplore.ieee.org/abstract/document/11646842
DOI: 10.1109/TGRS.2026.3722225

arXiv (earlier version): https://arxiv.org/pdf/2507.16716

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

## 4. Code for Retrieval and Semantic Localization (SELO) Evaluation

For the evaluation of **cross-modal retrieval** and **semantic localization (SELO)**, our implementation follows the evaluation protocols and code from the following open-source repositories:

* **Cross-modal retrieval evaluation:** Our retrieval evaluation is based on the implementation provided by [RemoteCLIP](https://github.com/ChenDelong1999/RemoteCLIP/blob/main/retrieval.py), which evaluates both image-to-text and text-to-image retrieval using Recall@K metrics.

* **Semantic localization (SELO) evaluation:** Our semantic localization evaluation follows the implementation provided by [RS5M](https://github.com/om-ai-lab/RS5M/blob/main/inference/inference.py).

We sincerely thank the authors of **RemoteCLIP** and **RS5M** for releasing their evaluation code.

## 5.Training Code
Our model is trained by **openCLIP**, so the training code is here: [openCLIP GitHub repository](https://github.com/mlfoundations/open_clip).


## 6. License

This project is licensed under the **Apache 2.0** license.
