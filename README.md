# Self-Supervised Learning for OOD Detection

A Simplified Pytorch implementation of *Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty(NeurIPS 2019)*

## Introduction
A number of **Self-supervised learning** have been proposed, each exploring a different pretext task. 
![image](https://user-images.githubusercontent.com/37788686/98557609-c4d4c400-22e7-11eb-99b3-7566a9dc499e.png)

Predicting rotation requires modeling shape. Texture alone is not sufficient for determining whether the zebra is flipped, although 

**The code supports only Multi-class OOD Detection experiment(in-dist: CIFAR-10, Out-of-dist: CIFAR-100/SVHN)** 


- Command 
  - RotNet-OOD
  
    python test.py --method=rot --ood_dataset=cifar100
  
  - baseline
  
    python test.py --method=msp --ood_dataset=svhn

- Reference
  - full code(by authors): https://github.com/hendrycks/ss-ood
  - Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty(NeurIPS 2019): https://arxiv.org/abs/1906.12340
  - A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks(ICLR 2017): https://arxiv.org/abs/1610.02136

