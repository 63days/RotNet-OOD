# Self-Supervised Learning for OOD Detection

A Simplified Pytorch implementation of *Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty(NeurIPS 2019)*

## Introduction
A number of **Self-supervised learning** have been proposed, each exploring a different pretext task. 
![image](https://user-images.githubusercontent.com/37788686/98618879-6807f680-2345-11eb-8a9a-39842d581add.png)

For example, the above image is unsupervised representation learning by predicting image rotations. Predicting rotation requires modeling shape. Texture alone is not sufficient for determining whether the zebra is flipped, although it may be sufficient for classification under ideal conditions. Thus, training with self-supervised auxiliary rotations may improve robustness.

## Method
CE Loss + auxiliary self-supervised loss(predicting rotations)
![image](https://user-images.githubusercontent.com/37788686/98619112-e6fd2f00-2345-11eb-8fed-05b653d0218a.png)

## Results
![image](https://user-images.githubusercontent.com/37788686/98619356-6a1e8500-2346-11eb-9190-53e7ce039329.png)

ROT(rotation method) shows better performance than MSP(Maximum Softmax Probability). So auxiliary self-supervised loss improves model robustness.

**The code supports only Multi-class OOD Detection experiment(in-dist: CIFAR-10, Out-of-dist: CIFAR-100/SVHN)** 


- Command 
  - RotNet-OOD
  
    python test.py --method=rot --ood_dataset=cifar100
  
  - baseline
  
    python test.py --method=msp --ood_dataset=svhn

- Reference
  - full code(by authors): https://github.com/hendrycks/ss-ood
  - Unsupervised Representation Learning by Predicting Image Rotations: https://arxiv.org/abs/1803.07728
  - Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty(NeurIPS 2019): https://arxiv.org/abs/1906.12340
  - A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks(ICLR 2017): https://arxiv.org/abs/1610.02136

