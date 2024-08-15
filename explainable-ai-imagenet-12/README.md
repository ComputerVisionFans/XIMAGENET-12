# **Introduction:**

<p align="center">
  <img alt="XimageNet-12" style="width: 728px; max-width: 100%; height: auto;" src="https://qiangli.de/imgs/flowchart2%20(1).png"/>
  <h1 align="center">🌟 XimageNet-12 🌟</h1> 
  <p align="center">An Explainable Visual Benchmark Dataset for Robustness Evaluation. A Dataset for Image Background Exploration!</p>
  <p align="center"><b>Blur Background, Segmented Background, AI-generated Background, Bias of Tools During Annotation, Color in Background, Random Background with Real Environment</b></p>
</p>

![](https://user-images.githubusercontent.com/18329471/234640541-a6a65fbc-d7a5-4ec3-9b65-55305b01a7aa.png)
[![license](https://img.shields.io/github/license/vietanhdev/anylabeling.svg)](https://github.com/vietanhdev/anylabeling/blob/master/LICENSE)
[![Downloads](https://img.shields.io/badge/Downloads-104-red)](https://www.kaggle.com/datasets/qianglijonas/explainable-ai-imagenet-12)
[![Documentation](https://img.shields.io/badge/Read-Documentation-green)](https://sites.google.com/view/ximagenet-12/home)
[![Follow](https://img.shields.io/badge/Follow-Qiang-blue)](https://qiangli.de/)
[![Paper](https://img.shields.io/badge/CVPRw2024-Paper-yellow)](https://openreview.net/forum?id=4tbbxtGwou)

> +⭐ Follow [Authors]((https://qiangli.de/)) for project updates.


**Website:** [XimageNet-12](https://sites.google.com/view/ximagenet-12/home)/


Here, we trying to understand how image background effect the Computer Vision ML model, on topics such as Detection and Classification, based on baseline Li et.al work on ICLR 2022: [Explainable AI: Object Recognition With Help From Background](https://iclr.cc/virtual/2022/workshop/9069), we are now trying to enlarge the dataset, and analysis the following topics: **Blur Background** / **Segmented Background** / **AI generated Background**/ **Bias of tools during annotation**/ **Color in Background** / **Dependent Factor in Background**/ **LatenSpace Distance of Foreground**/ **Random Background with Real Environment**!
Ultimately, we also define the **math equation of Robustness Scores**! So if you feel interested How would we make it or join this research project? please feel free to collaborate with us!

In this paper, we propose an explainable visual dataset, XIMAGENET-12, to evaluate the robustness of visual models. XIMAGENET-12 consists of over 200K images with 15,410 manual semantic annotations. Specifically, we deliberately selected 12 categories from ImageNet, representing objects commonly encountered in practical life. To simulate real-world situations, we incorporated six diverse scenarios, such as overexposure, blurring, and color changes, etc. We further develop a quantitative criterion for robustness assessment, allowing for a nuanced understanding of how visual models perform under varying conditions, notably in relation to the background.

# **Progress:**
- **Blur Background**-&gt; **Done**! You can find the image Generated in the corresponding folder!
- **Segmented Background** -&gt; **Done**! you can download the image and its corresponding transparent mask image!
- **Color in Background**-&gt;**Done**!~~ you can now download the image with different background color modified, and play with different color-ed images!
- **Random Background with Real Environment** -&gt; **Done**! you can also find we generated the image with the photographer's real image as a background and removed the original background of the target object, but similar to the style!
- **Bias of tools during annotation**-&gt;**Done**! for this one, you won't get a new image, because this is about math and statistics data analysis when different tools and annotators are applied!
- **AI generated Background**-&gt; **current on progress** ( 12 /12) **Done**!, So basically you can find one sample folder image we uploaded, please take a look at how real it is, and guess what LLM model we are using to generate the high-resolution background to make it so real :)


# What tool we used to generate those images?
We employed a combination of tools and methodologies to generate the images in this dataset, ensuring both efficiency and quality in the annotation and synthesis processes.

1. IoG Net: Initially, we utilized the IoG Net, which played a foundational role in our image generation pipeline.

2. Polygon Faster Labeling Tool: To facilitate the annotation process, we developed a custom Polygon Faster Labeling Tool, streamlining the labeling of objects within the images.

3. AnyLabeling Open-source Project: We also experimented with the AnyLabeling open-source project, exploring its potential for our annotation needs.

4. V7 Lab Tool: Eventually, we found that the V7 Lab Tool provided the most efficient labeling speed and delivered high-quality annotations. As a result, we standardized the annotation process using this tool.

5. Data Augmentation: For the synthesis of synthetic images, we relied on a combination of deep learning frameworks, including scikit-learn and OpenCV. These tools allowed us to augment and manipulate images effectively to create a diverse range of backgrounds and variations.

6. GenAI: Our dataset includes images generated using the Stable Diffusion XL model, along with versions 1.5 and 2.0 of the Stable Diffusion model. These generative models played a pivotal role in crafting realistic and varied backgrounds.

For a detailed breakdown of our prompt engineering and hyperparameters, we invite you to consult our upcoming paper. This publication will provide comprehensive insights into our methodologies, enabling a deeper understanding of the image generation process.


# How to use our dataset?
this dataset has been/could be downloaded via Kagglg (104+), paper with code, and hugging face!
Here are some examples of people who use it for their projects: for instance verify their CLIP model robustness on the XimageNet-12 dataset.

| Pretraining Dataset          | Blur_bg | Blur_obj | Color  | Rand_bg | Seg_img |
|------------------------------|---------|----------|--------|---------|---------|
| ViT-B-16 (ImageNet)           | 88.4    | 90.8     | 66.5   | 17.2    | 49.0    |
| ViT-B-16 (XImageNet-I2)       | 71.51   | 70.21    | 74.14  | 38.01   | 78.7    |
| CLIP-ViT-B-16 (DATACOMP)      | 98.9    | 97.5     | 98.6   | 42.4    | 95.4    |
| CLIP-ViT-L-14 (OpenAI)        | 98.9    | 98.2     | 98.3   | 52.5    | 95.7    |

*Table 17: Performance on XImageNet-I2 benchmark with ViT-B and ViT-L considering different pretraining settings. CLIP pretraining with DATACOMP is quite robust to various shifts.*
 **Aristeidis Panos et.al, Imperfect Vision Encoders: Efficient and Robust Tuning for Vision-Language Models**


| Pretrained Dataset                       | Model Name       | Parameters (M) | Blur_bg | Blur_obj | Color_g | Color_b | Color_grey | Color_r | Rand_bg | Seg_img |
|------------------------------------------|------------------|----------------|---------|----------|---------|---------|------------|---------|---------|---------|
| **ImageNet (Original images) EX1**       | ResNet50         | 25.60          | 90.97%  | 89.41%   | 84.42%  | 86.98%  | 92.13%     | 89.03%  | 22.41%  | 68.55%  |
|                                           | VGG-16           | 138.4          | 89.92%  | 89.91%   | 88.64%  | 70.46%  | 81.48%     | 80.68%  | 24.58%  | 49.62%  |
|                                           | MobileNetV2      | 3.5            | 92.30%  | 85.73%   | 88.67%  | 88.81%  | 90.17%     | 84.98%  | 25.69%  | 66.43%  |
|                                           | EfficientNetB0   | 5.3            | 91.44%  | 91.06%   | 82.45%  | 86.44%  | 83.65%     | 83.65%  | 25.29%  | 53.56%  |
|                                           | EfficientNetB3   | 12.3           | 93.37%  | 93.61%   | 87.32%  | 90.38%  | 88.83%     | 85.90%  | 25.29%  | 70.32%  |
|                                           | DenseNet121      | 8.1            | 93.77%  | 92.44%   | 87.39%  | 88.83%  | 82.31%     | 82.31%  | 26.41%  | 69.67%  |
|                                           | ViT              | 86.6           | 86.16%  | 84.49%   | 70.90%  | 72.35%  | 91.26%     | 89.18%  | 36.10%  | 78.64%  |
|                                           | Swin             | 87.76          | 80.97%  | 64.59%   | 64.59%  | 65.91%  | 69.28%     | 64.41%  | 19.46%  | 44.57%  |
| **XImageNet-12 (*Scenarios) EX2**         | ResNet50         | 25.60          | 83.52%  | 80.48%   | 74.81%  | 74.18%  | 80.40%     | 80.60%  | 53.91%  | 57.85%  |
|                                           | VGG-16           | 138.4          | 74.85%  | 63.16%   | 64.26%  | 77.58%  | 69.19%     | 70.25%  | 72.05%  | 73.27%  |
|                                           | AlexNet          | 61.1           | 55.14%  | 44.81%   | 46.92%  | 64.25%  | 69.07%     | 63.34%  | 32.39%  | 78.56%  |
|                                           | MobileNetV3      | 3.50           | 67.36%  | 62.19%   | 64.25%  | 69.64%  | 63.41%     | 63.41%  | 48.39%  | 78.85%  |
|                                           | DenseNet121      | 8.1            | 71.90%  | 69.94%   | 70.04%  | 87.37%  | 69.64%     | 71.64%  | 58.50%  | 72.56%  |
|                                           | ViT              | 86.56          | 71.54%  | 68.77%   | 68.30%  | 75.80%  | 71.14%     | 71.14%  | 36.39%  | 78.66%  |
|                                           | Swin             | 87.76          | 72.81%  | 51.02%   | 51.02%  | 51.96%  | 81.63%     | 76.42%  | 13.23%  | 80.64%  |

*Table 1: Comparison of SOTA visual models with diverse scenarios. Here all the evaluation metrics are Top-1 Accuracy.*


# Code Structure


    
    ├── showmask.py             # show the mask of the image, you can use it for showing the segmentation mask generated with fully back mask image
    ├── removebg.py             # remove the jpg image
    ├── transparent.py          # remove the background of the image and convert it into transparent form and save the output as png image
    ├── Segmentbackground.py    # generate fully transpanrent background image ( currently the one used during the demostration)
    ├── json_into_mask.py       # convert the json file into mask image
    ├── segementwithRGB.py      # the code NOT used for generating the background with real environment, bit-wise and operation cannot create optimal result
    ├── generate_background.py  # generate the background with real environment, the code used for generating the background with real environment,the addWeighted method is used for generating the optimal result and it is the best method for generating the background with real environment, it is blending the image with the weight of the image, which performs an element-wise addition of the two images with equal weights.
    └── README.md



# **Citation:**
If you find XimageNet-12 Dataset useful in your research, please consider citing:
```
@inproceedings{
li2024ximagenet,
title={{XIMAGENET}-12: An Explainable Visual Benchmark Dataset for Model Robustness Evaluation},
author={Qiang Li and Dan Zhang and Shengzhao Lei and Xun Zhao and WeiWei Li and Porawit Kamnoedboon and Junhao Dong and Shuyan Li},
booktitle={Synthetic Data for Computer Vision Workshop @ CVPR 2024},
year={2024},
url={https://openreview.net/forum?id=4tbbxtGwou}
}
